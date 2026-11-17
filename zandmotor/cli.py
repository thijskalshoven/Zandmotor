"""Command-line entry point: argument parsing and run orchestration."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import pathlib
import sys
import webbrowser

import numpy as np
import requests
from rasterio.warp import Resampling
from scipy import ndimage

from zandmotor.config import CACHE, CFG, HERE, TZ, UTC, log
from zandmotor.errors import ZandmotorError
from zandmotor.flood import (
    FillDrain, compute_wet_masks, effective_level, fill_drain_series, probable_extent,
    rideable_at, step_times,
)
from zandmotor.geometry import (
    basin_mask, load_lagoon_polygon, make_grid, mask_to_ring, polygon_mask, trim_lagoon_to_water,
)
from zandmotor.html_report.render import write_html
from zandmotor.lagoon_model.calibration import fetch_sea_record, fit_lagoon_response
from zandmotor.lagoon_model.dry_override import flats_dry_override
from zandmotor.lagoon_model.fill_drain_fit import FillDrainFit, fit_fill_drain
from zandmotor.lagoon_model.runtime import (
    LagoonWater, describe_observation, latest_observation, observation_records,
    possible_fill, resting_area, wet_mask_from_area, wetness_fingerprint,
)
from zandmotor.lagoon_model.types import LakeResponse, WindResponse, model_from_dict
from zandmotor.outline import make_outline
from zandmotor.rendering import class_png, png_uri, terrain_rgb
from zandmotor.terrain import (
    coastal_zone_mask, dem_age_days, fetch_ahn, overlay_coastal_dem, permanent_sea_mask,
    prepare_terrain,
    prune_cache, read_to_grid, spill_elevation,
)
from zandmotor.tides import WaterLevel
from zandmotor.time_utils import parse_iso_datetime
from zandmotor.wind import (
    fetch_sea_temp, fetch_waves, fetch_wind, fetch_wind_history, is_daylight, rain_sum,
    sea_temp_at, wave_forcing, waves_at, wetsuit_advice,
)
from zandmotor.windows import find_windows, format_window


def _source(name, when, now, detail="", stale_days=None):
    """One row of the report's "Data sources" list: what the data is, when
    it was last pulled or captured, and whether that is old enough to act on.
    `when` is a UTC datetime, or None when there is no such data."""
    if when is None:
        return {"name": name, "when": "none", "age": "", "detail": detail, "stale": True}
    days = (now - when).days
    age = "today" if days < 1 else "yesterday" if days < 2 else f"{days} days ago"
    return {"name": name, "when": when.astimezone(TZ).strftime("%a %d %b %Y"), "age": age,
            "detail": detail, "stale": stale_days is not None and days > stale_days}


def _parse_args():
    ap = argparse.ArgumentParser(description="Zandmotor lagoon nowcast for kitesurfing")
    ap.add_argument("--hours", type=int, default=8, help="hours ahead (default 8)")
    ap.add_argument("--step", type=int, default=60, help="minutes per frame (default 60)")
    ap.add_argument("--calibrate", action="store_true",
                    help="refit the lagoon's wetness model from satellite images")
    ap.add_argument("--outline", nargs="?", const="auto", choices=["auto", "sentinel", "osm"],
                    help="trace the lagoon outline (default: auto)")
    ap.add_argument("--out", default="zandmotor_lagoon.html")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def _load_terrain_and_outline(args, grid, now, warnings, sources):
    """Terrain (raw + prepared) and the lagoon outline/mask, trimmed to what
    the elevation data actually shows as water, and the basin around it.
    Returns (z_raw, ring, lagoon, basin, terrain_note): the terrain is
    prepared later, because its basin floor comes from the fill-and-drain
    fit (see main). z_raw has the coastal DTM laid over AHN (heights for the
    flood model); the outline trim and the basin use AHN alone, whose
    no-return gaps are the evidence of where the lagoon stood - and changing
    either would invalidate the year-long wetness map."""
    if CFG["local_dem"]:
        z_raw = read_to_grid(CFG["local_dem"], grid, resampling=Resampling.max)
        terrain_note = f"local file {pathlib.Path(CFG['local_dem']).name}"
    else:
        z_raw = fetch_ahn(grid)
        terrain_note = "AHN (PDOK)"
        dem_age = dem_age_days()
        sources.append(_source("Terrain (AHN)",
                               None if dem_age is None else now - dt.timedelta(days=dem_age),
                               now, "downloaded; the survey itself is older",
                               CFG["dem_max_age_days"]))
        if dem_age is not None and dem_age > CFG["dem_max_age_days"]:
            warnings.append(f"Cached AHN terrain is {dem_age:.0f} days old. Zandmotor is built "
                            "to reshape, and this DEM decides which of the traced outline counts "
                            "as water - delete cache/ahn_dtm.tif to refetch.")
            terrain_note += f" (cached {dem_age:.0f} days ago)"
    ring, props = (None, None) if args.outline else load_lagoon_polygon()
    if ring is None:
        try:
            ring = make_outline(grid, now, args.outline or "osm",
                                z=np.nan_to_num(z_raw, nan=-4.0))
            ring, props = load_lagoon_polygon()
        except (requests.RequestException, RuntimeError, KeyError, ValueError) as e:
            if args.outline:
                sys.exit(f"Could not trace the lagoon: {e}")
            log.warning("Could not trace the lagoon automatically (%s)", e)
    if ring is None:
        ring = CFG["lagoon_polygon"]
        warnings.append("Could not trace the lagoon, so a rough placeholder outline is "
                        "used. Run with --outline, or draw it on geojson.io and save it "
                        "as lagoon.geojson.")
    elif props.get("source") == "openstreetmap":
        warnings.append("Lagoon outline comes from OpenStreetMap and may be outdated. "
                        "Run --outline sentinel for one traced from a recent satellite image.")
    elif props.get("image_time"):
        age = (now - parse_iso_datetime(props["image_time"])).days
        if age > 60:
            warnings.append(f"Lagoon outline is from a satellite image {age} days old. "
                            "Run --outline to refresh it.")
    if props and props.get("image_time"):
        sources.append(_source("Lagoon outline", parse_iso_datetime(props["image_time"]), now,
                               "satellite image it was traced from", 60))
    else:
        sources.append(_source("Lagoon outline", None, now,
                               "from OpenStreetMap" if props else "placeholder"))
    lagoon = polygon_mask(grid, ring)

    ring, lagoon, cut_ha = trim_lagoon_to_water(ring, lagoon, z_raw, grid)
    if cut_ha > 0.05:
        warnings.append(f"Cut {cut_ha:.1f} ha of dune off the traced outline (elevation data "
                        f"puts it above {CFG['lagoon_trim_land_m']:.1f} m NAP).")
    basin, found = basin_mask(grid, lagoon, z_raw)
    if not found:
        warnings.append("No lagoon basin yet, so the probable water edge can't reach beyond "
                        f"the outline. Run --outline to start {CFG['lagoon_basin_file']}.")
    z_heights, covered = overlay_coastal_dem(z_raw, grid)
    if covered:
        terrain_note += f" + {CFG['coastal_dem_label']} where it has data"
        # survey date only known to the year, so not a _source() row
        sources.append({"name": CFG["coastal_dem_label"], "when": "2026", "age": "",
                        "detail": f"heights for {covered:.0%} of the map, including the basin",
                        "stale": False})
    return z_heights, ring, lagoon, basin, terrain_note


def _demote_weak_wind(model):
    """A cached wind fit that the current wind_r_min/wind_min_n would reject,
    as the lake model built from the same record. The thresholds were
    tightened after a fit at r=0.36, n=28 picked wind mode; this applies them
    to the cache without a year-long --calibrate."""
    if (not isinstance(model, WindResponse)
            or (abs(model.r) >= CFG["wind_r_min"] and model.n >= CFG["wind_min_n"])):
        return model
    log.info("Cached wind fit (r=%.2f, n=%d) is below the wind_r_min=%.2f / wind_min_n=%d "
             "bar; using lake mode instead", model.r, model.n, CFG["wind_r_min"],
             CFG["wind_min_n"])
    return LakeResponse(r=model.sea_r, wind_r=model.r, n=model.n, fitted=model.fitted,
                        times=model.times, areas=model.areas, n_optical=model.n_optical,
                        n_sar=model.n_sar, obs=model.obs, sar_tracks=model.sar_tracks)


def _load_or_fit_lagoon_model(args, grid, lagoon, basin, ring, z_raw, now, warnings, sources):
    """Loads the cached lagoon wetness model (refitting first if --calibrate),
    verifying it was fitted against the current outline. Also refreshes the
    satellite dry-ground override for the flats, and loads (or with
    --calibrate refits) the fill-and-drain parameters. Returns (lagoon_model,
    freq, dry_override, fill_fit)."""
    freq_file = CACHE / "wetness_frequency.npy"
    confidence_file = CACHE / "wetness_confidence.npy"
    response_file = CACHE / "lagoon_response.json"
    meta_file = CACHE / "wetness_meta.json"
    fingerprint = wetness_fingerprint(lagoon, ring, basin)
    sea_record = None
    if args.calibrate:
        sea_record = fetch_sea_record(now)
        response, freq, valid_count = fit_lagoon_response(grid, lagoon, now, basin,
                                                          sea=sea_record)
        if response is not None:
            response_file.write_text(json.dumps(response.to_dict(), indent=1))
            np.save(freq_file, freq)
            np.save(confidence_file, valid_count)
            meta_file.write_text(json.dumps({"fingerprint": fingerprint,
                                             "fitted": now.isoformat()}, indent=1))

    lagoon_model, freq = None, None
    if response_file.exists() and freq_file.exists():
        # The frequency map is only valid for the outline it was fitted against.
        # Comparing shapes cannot detect a re-traced outline - the shape is fixed
        # by bbox and resolution - so the fingerprint is what actually guards this.
        stored = json.loads(meta_file.read_text()).get("fingerprint") if meta_file.exists() else None
        if stored is None:
            # A cache from before this check existed. Accept it once and record the
            # fingerprint so every later run is verified, but say out loud that this
            # one run is unverified rather than implying it was checked.
            freq = np.load(freq_file)
            if freq.shape == lagoon.shape:
                lagoon_model = model_from_dict(json.loads(response_file.read_text()))
                meta_file.write_text(json.dumps(
                    {"fingerprint": fingerprint, "adopted_unverified": now.isoformat()}, indent=1))
                warnings.append("The cached wetness map predates the outline check, so this run "
                                "could not verify it was fitted against the current outline. It "
                                "has been accepted once and fingerprinted; later runs are checked. "
                                "If you have re-traced the outline since calibrating, run "
                                "--calibrate.")
            else:
                freq = None
        elif stored != fingerprint:
            warnings.append("The lagoon outline or basin has changed since the wetness map was "
                            "fitted, so the map no longer describes this footprint and is being "
                            "ignored. Run --calibrate to refit.")
        else:
            freq = np.load(freq_file)
            if freq.shape == lagoon.shape:
                lagoon_model = model_from_dict(json.loads(response_file.read_text()))
            else:
                freq = None
    if lagoon_model is not None:
        lagoon_model = _demote_weak_wind(lagoon_model)
        mode = lagoon_model.mode
        mode_txt = {"tidal": "tidal", "wind": "wind-driven (experimental)",
                    "lake": "a slowly-varying lake"}[mode]
        age = (now - parse_iso_datetime(lagoon_model.fitted)).days
        log.info("Lagoon response model: %s (r=%.2f, n=%d, fitted %d days ago)",
                 mode_txt, lagoon_model.r, lagoon_model.n, age)
        if mode == "wind":
            n_opt, n_sar = lagoon_model.n_optical, lagoon_model.n_sar
            warnings.append(f"Lagoon wet area was fitted to wind gusts rather than tide "
                            f"(r={lagoon_model.r:.2f} over the {lagoon_model.window_h}h before "
                            f"each hour, n={lagoon_model.n} images: {n_opt} optical + {n_sar} SAR). "
                            "Treat this as unproven: an earlier version of this fit looked strong at "
                            "n=20 and collapsed to r=0.11 at n=64. Sea level showed ~no relationship "
                            f"(r={lagoon_model.sea_r:.2f}).")
        if age > 90:
            warnings.append(f"Lagoon response model ({mode_txt}, fitted {age} days ago) "
                            "may be stale; run --calibrate to refit.")
        if not lagoon_model.obs:
            warnings.append("The lagoon record predates the seen-pixel rules and the per-track "
                            "radar calibration, so its areas mix two radar tracks that read ~2 ha "
                            "apart. Run --calibrate to rebuild it.")
        # The lagoon's size only moves when new images are pulled in, so how old
        # the newest one is says how current the "Lagoon water" figure is.
        newest = latest_observation(lagoon_model, now)
        latest = dt.datetime.fromtimestamp(newest["t"], UTC) if newest else None
        stale = CFG["lagoon_obs_stale_days"]
        sources.append(_source("Satellite images (lagoon size)", latest, now,
                               (f"newest usable image ({describe_observation(newest)}); "
                                if newest else "") +
                               f"last pulled "
                               f"{parse_iso_datetime(lagoon_model.fitted).astimezone(TZ):%d %b} "
                               "with --calibrate", stale))
        if latest is not None and (now - latest).days > stale:
            warnings.append(f"The newest satellite image of the lagoon is "
                            f"{(now - latest).days} days old, so the lagoon size may be out "
                            "of date. Run --calibrate to pull in recent images.")
    else:
        warnings.append("No satellite-fitted lagoon response model yet; showing the "
                        "whole traced outline as wet until one exists. Run --calibrate "
                        "to fit one from a year of satellite images.")

    dry_override = flats_dry_override(grid, lagoon, now)
    if dry_override is None:
        warnings.append("Could not refresh the satellite dry-ground check for the "
                        "flats this run; showing the elevation-only estimate there.")

    fill_file = CACHE / CFG["fill_drain_file"]
    if args.calibrate and lagoon_model is not None and freq is not None:
        lat_c, lon_c = _map_centre()
        start = dt.datetime.fromtimestamp(sea_record[0][0], UTC)
        waves = fetch_waves(lat_c, lon_c, start=start, end=now)
        try:
            wt, _, _, wr = fetch_wind_history(lat_c, lon_c, start, now)
            rain = (wt, wr)
        except requests.RequestException as e:
            log.warning("No rain history (%s); fill-and-drain is fitted without rain.", e)
            rain = None
        fit = fit_fill_drain(grid, z_raw, lagoon, basin, freq, observation_records(lagoon_model),
                             sea_record, waves, rain, dry_override, now)
        if fit is not None:
            fit.fingerprint = fingerprint
            fill_file.write_text(json.dumps(fit.to_dict(), indent=1))
    fill_fit, why_not = FillDrainFit.load(fill_file, fingerprint)
    if why_not and lagoon_model is not None and lagoon_model.mode == "lake":
        warnings.append(f"The lagoon's fill-and-drain uses its default drain time "
                        f"({fill_fit.drain_days:g} days) because there is {why_not}. Run "
                        "--calibrate to fit it to the satellite record.")
    return lagoon_model, freq, dry_override, fill_fit


def _obs_group(o):
    """Legend group for one observation: the sensor, and for radar the track,
    since the two tracks are calibrated (and can drift) separately."""
    if o.get("sensor") == "optical":
        return "satellite photo"
    if o.get("sensor") == "landsat":
        return "Landsat photo"
    if o.get("sensor") == "sar":
        return (o.get("track") or "radar").replace("S1 orbit", "radar, orbit")
    return "satellite image"


def _lagoon_now(lagoon_model, freq, now, water, wind, warnings):
    """What the report shows for the lagoon in lake mode: the newest reliable
    observation, plus a warning when enough rain since then means the lagoon
    may have filled. Returns (lake_area, label, latest_obs)."""
    if lagoon_model is None or freq is None or lagoon_model.mode != "lake":
        return None, "", None
    latest = latest_observation(lagoon_model, now)
    if latest is None:
        return None, "", None
    when = dt.datetime.fromtimestamp(latest["t"], UTC)
    area = float(latest["area_ha"])
    err = latest.get("track_mae_ha")
    label = (f"{area:.1f} ha" + (f" ±{err:.1f}" if err else "") +
             f" · {describe_observation(latest)} · {when.astimezone(TZ):%d %b}")
    # Surges are modelled now (FillDrain), so only rain still warns: the
    # 2026-05-14 fill (13.5 ha) came with 19 mm of rain and no surge.
    fill = possible_fill(latest["t"], now.timestamp(), water.meas,
                         (wind[0], wind[6]) if wind is not None else None)
    rainy = fill["rain_mm"] is not None and fill["rain_mm"] >= CFG["fill_flag_rain_mm"]
    if rainy:
        warnings.append(f"Since the newest lagoon image ({when.astimezone(TZ):%d %b}), "
                        f"{fill['rain_mm']:.0f} mm of rain fell. The lagoon once filled to "
                        "about 13 ha after rain without a surge (May 2026), which the map "
                        "can't predict, so it may be bigger than shown. Run --calibrate to "
                        "check newer images.")
        label += " · may have filled since"
    log.info("Lagoon area (lake mode): %.2f ha, %s image of %s%s", area,
             describe_observation(latest), when.date(),
             "; may have filled since (rain)" if rainy else "")
    return area, label, latest


def _map_centre():
    """(lat, lon) of the map's centre, where weather and waves are fetched."""
    return ((CFG["bbox_wgs84"][1] + CFG["bbox_wgs84"][3]) / 2,
            (CFG["bbox_wgs84"][0] + CFG["bbox_wgs84"][2]) / 2)


def _surge_text(te, water, waves, fill_fit):
    """"Mon 06:45 (sea +1.56 m NAP, waves 1.9 m)" for the surge at te. The
    surge is found on the effective level (flood.effective_level), but sea
    and waves are what anyone can check."""
    when = dt.datetime.fromtimestamp(te, UTC).astimezone(TZ)
    txt = f"{when:%a %H:%M} (sea {water.level(te)[0]:+.2f} m NAP"
    if fill_fit.wave_k and waves is not None:
        t, _, tp, wd = waves
        compass = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                   "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        i = int(np.argmin(np.abs(t - te)))
        txt += f", waves {float(waves_at(waves, te)):.1f} m"
        if np.isfinite(tp[i]) and np.isfinite(wd[i]):
            txt += f", {tp[i]:.0f} s from {compass[int(round(wd[i] / 22.5)) % 16]}"
    return txt + ")"


def _fill_drain_label(state, lagoon_ha, latest, surge_text):
    """Sidebar label for a frame whose size comes from FillDrain rather than
    straight from the newest image, or None when it is the image's own.
    `surge_text(te)` describes the surge at te (_surge_text)."""
    if state is None:
        return None
    area, surge = state
    image = (f"image {dt.datetime.fromtimestamp(latest['t'], UTC).astimezone(TZ):%d %b}: "
             f"{latest['area_ha']:.1f} ha")
    if surge is not None:
        return (f"{lagoon_ha:.1f} ha · estimate: filled by the surge of {surge_text(surge[1])}, "
                f"draining since · {image}")
    if abs(area - latest["area_ha"]) >= 0.05:
        return f"{lagoon_ha:.1f} ha · estimate: draining since the {image}"
    return None


def _lagoon_record(lagoon_model, latest, fill_fit):
    """Every observation in the record, for the sidebar chart, grouped by
    sensor and track so a disagreement between them is visible. The caption
    says how well fill-and-drain replays the record."""
    if lagoon_model is None:
        return None
    recs = sorted(observation_records(lagoon_model), key=lambda o: o["t"])
    if not recs:
        return None
    latest_i = next((i for i, o in enumerate(recs) if latest and o["t"] == latest["t"]), -1)
    return {"times": [o["t"] for o in recs], "areas": [round(o["area_ha"], 2) for o in recs],
            "groups": [_obs_group(o) for o in recs], "latest": latest_i,
            "label": (f"Lagoon size in {len(recs)} satellite images — measured, not forecast. "
                      "The ringed point is the one shown above. " + _fit_caption(fill_fit))}


def _fit_caption(fit):
    """One sentence on the fill-and-drain parameters and how they were set."""
    if fit.fitted is None:
        return (f"Between images the lagoon fills on surges and drains back with a default "
                f"{fit.drain_days:g}-day drain time (not yet fitted: run --calibrate).")
    return (f"Between images it fills on surges and drains back ({fit.drain_days:g}-day drain "
            f"time, fitted {parse_iso_datetime(fit.fitted).astimezone(TZ):%d %b}). Replayed "
            f"from image to image, that misses the next image by {fit.mae_ha:.1f} ha on "
            f"average; holding the last image misses by {fit.hold_mae_ha:.1f} ha.")


def _build_frames(args, now, grid, z, lagoon, basin, lagoon_model, freq, dry_override,
                  fill_fit, warnings):
    """Fetches tide/wind/sea-temp and builds the per-hour frames.
    Returns (frames, times, water, wind, daylight, lagoon_obs, lagoon_label,
    lagoon_water, lagoon_trend)."""
    F = spill_elevation(z)
    coastal_zone = coastal_zone_mask(z, lagoon | basin)

    lat_c, lon_c = _map_centre()
    # Reach back to the newest lagoon image, so the "may have filled since"
    # check sees the measured sea level and the rain since then.
    newest = (latest_observation(lagoon_model, now)
              if lagoon_model is not None and freq is not None else None)
    back_days = 3
    if newest is not None:
        back_days = min(max(math.ceil((now.timestamp() - newest["t"]) / 86400) + 1, 3),
                        CFG["fill_flag_max_days"] + 1)
    # Tide data out to the end of the wind forecast, not just the slider range:
    # the tidal lagoon mode needs it for the week-long window search, and the
    # map draws the wet beach at every frame.
    # The size-over-time chart replays the last trend_days_back, so the
    # measured sea (and waves) go back that far too.
    history_days = max(back_days, CFG["trend_days_back"] + 1)
    water = WaterLevel(now, now + dt.timedelta(days=8), history_days=history_days)
    wind = fetch_wind(lat_c, lon_c, past_days=history_days)
    sea_temp = fetch_sea_temp(lat_c, lon_c)
    # waves only enter through the fitted flood level; still water when it has no wave term
    waves = (fetch_waves(lat_c, lon_c, past_days=history_days, forecast_days=8)
             if fill_fit.wave_k else None)
    if fill_fit.wave_k and waves is None and lagoon_model is not None:
        warnings.append("No wave heights this run, so the lagoon's surge fill is tested on "
                        "still water and may miss a fill driven by waves.")
    wind_hist = (wind[0], wind[2]) if wind is not None else None
    daylight = wind[5] if wind is not None else []

    frames = []
    n = int(args.hours * 60 / args.step) + 1
    times = [now + dt.timedelta(minutes=k * args.step) for k in range(n)]

    # The open sea, to draw it apart from the wet beach. Neither is counted.
    datum = water.low_water_datum()
    sea = permanent_sea_mask(z, lagoon, datum)
    log.info("Open sea below the %.2f m NAP low-water datum: %.0f ha of the %.0f ha map "
             "(drawn, not counted)", datum, sea.sum() * grid.cell_area_m2 / 1e4,
             grid.width * grid.height * grid.cell_area_m2 / 1e4)

    lake_area, lagoon_label, latest = _lagoon_now(lagoon_model, freq, now, water, wind, warnings)
    lagoon_obs = _lagoon_record(lagoon_model, latest, fill_fit)
    rain = (wind[0], wind[6]) if wind is not None else None

    def flood_level(te):
        lvl, src = water.level(te)
        return effective_level(lvl, wave_forcing(waves, te, fill_fit.wave_form), fill_fit.wave_k,
                               fill_fit.flood_offset_m,
                               rain_sum(rain, te, CFG["fill_rain_window_h"]), fill_fit.rain_k), src

    def surge_text(te):
        return _surge_text(te, water, waves, fill_fit)

    fill_drain = None
    if lake_area is not None:
        # from the newest image on, the lagoon fills on surges and drains back
        # between them (see FillDrain); the replay starts at the image itself
        def fill_drain(base_wet):
            # surges are flooded onto the lagoon size the fit used (base_ha),
            # so the report counts the surges the fit was scored on
            if fill_fit.base_ha:
                base_wet = wet_mask_from_area(freq, basin | lagoon, fill_fit.base_ha,
                                              grid.cell_area_m2)
            return FillDrain(latest["t"], lake_area, resting_area(lagoon_model, latest["t"]),
                             (now + dt.timedelta(days=8)).timestamp(), flood_level,
                             lambda lvl: compute_wet_masks(z, F, lvl, coastal_zone, lagoon, sea,
                                                           dry_override, grid),
                             sea, basin, base_wet, grid.cell_area_m2,
                             drain_days=fill_fit.drain_days)
    lagoon_water = LagoonWater(lagoon, grid, lagoon_model, freq, water.level, wind_hist,
                               lake_area, basin=basin, fill_drain=fill_drain,
                               range_fn=fill_fit.range_ha if lake_area is not None else None)
    # the drawn edge follows the flood a little past the basin, towards the sea
    reach = (ndimage.distance_transform_edt(~basin) * CFG["grid_res_m"]
             <= CFG["probable_reach_m"])

    for t in times:
        te = t.timestamp()
        sea_level, src = water.level(te)
        lagoon_ha, usable_ha, w, quiver, verdict, notes, wind_txt, lagoon_wet = \
            rideable_at(te, lagoon_water, wind)
        beach, uncounted = compute_wet_masks(z, F, sea_level, coastal_zone, lagoon, sea,
                                             dry_override, grid)
        # drawn as a line, never counted - see probable_extent
        probable = probable_extent(lagoon_wet, beach, uncounted, sea, reach)
        probable_ha = float(probable.sum() * grid.cell_area_m2 / 1e4)
        probable_ring = (mask_to_ring(probable, grid)
                         if probable_ha - lagoon_ha >= CFG["probable_min_extra_ha"] else None)
        frame_label = _fill_drain_label(lagoon_water.state(te), lagoon_ha, latest, surge_text)
        lagoon_range = lagoon_water.range_at(te)
        dark = not is_daylight(daylight, te)
        if dark:
            verdict = "no"
            notes = notes + ["dark"]
        water_c = sea_temp_at(sea_temp, te)
        suit, suit_note = wetsuit_advice(water_c, w)
        tl = t.astimezone(TZ)
        frames.append({"time": tl.strftime("%H:%M"), "day": tl.strftime("%A %d %B"), "te": te,
                       "sea": sea_level, "lagoon_ha": lagoon_ha, "src": src,
                       "lagoon_label": frame_label,
                       "lagoon_range": (None if lagoon_range is None
                                        else [round(lagoon_range[0], 1), round(lagoon_range[1], 1)]),
                       "usable_ha": usable_ha,
                       "wind": wind_txt, "verdict": verdict, "notes": notes,
                       # raw numbers for the map's compass rose (None without a forecast)
                       "wind_deg": None if w is None else round(w[2], 1),
                       "wind_kn": None if w is None else round(w[0], 1),
                       "gust_kn": None if w is None else round(w[1], 1),
                       "dark": dark, "water_c": water_c,
                       "wetsuit": suit, "wetsuit_note": suit_note,
                       "quiver": quiver,
                       "png": class_png(lagoon_wet, beach, uncounted),
                       "probable": (None if probable_ring is None
                                    else [[lat, lon] for lon, lat in probable_ring]),
                       "probable_ha": round(probable_ha, 2)})
        log.info("%s  sea %+.2f  lagoon %4.1f ha  biggest patch %4.1f ha  wet beach %5.1f ha "
                 "probable edge %4.1f ha (not counted)  %-5s %s [%s]", tl.strftime("%a %H:%M"),
                 sea_level, lagoon_ha, usable_ha, beach.sum() * grid.cell_area_m2 / 1e4,
                 probable_ha, verdict, "dark" if dark else "    ", src)

    lagoon_trend = _lagoon_trend(lagoon_model, lagoon_water, fill_fit, flood_level, now, latest,
                                 surge_text)
    return (frames, times, water, wind, daylight, lagoon_obs, lagoon_label, lagoon_water,
            lagoon_trend)


def _lagoon_trend(lagoon_model, lagoon_water, fill_fit, flood_level, now, latest, surge_text):
    """Data for the size-over-time chart: the satellite images of the last
    trend_days_back; between each and the next, the fill-and-drain replay the
    fit was scored on (so the jump at the next image is the model's miss);
    and from the newest image on, the report's own estimate into the
    forecast, with every surge that reaches the basin marked. Lake mode
    only (None otherwise), where the size is an estimate between images."""
    if latest is None or lagoon_model is None or lagoon_model.mode != "lake":
        return None
    t0 = now.timestamp() - CFG["trend_days_back"] * 86400
    t1 = now.timestamp() + CFG["trend_days_ahead"] * 86400
    recs = sorted((o for o in observation_records(lagoon_model)
                   if t0 <= o["t"] <= latest["t"]
                   and o["area_ha"] >= CFG["lake_min_plausible_ha"]), key=lambda o: o["t"])
    # start shortly before the first image instead of on an empty stretch
    t0 = max(t0, recs[0]["t"] - 2 * 86400) if recs else t0
    segments, surges = [], {}

    def add(kind, start, ts, areas, events):
        # hourly points are plenty for a 300 px wide chart
        pts = [[float(start["t"]), round(start["area_ha"], 2)]] + [
            [float(t), round(float(a), 2)] for t, a in zip(ts, areas) if t % 3600 == 0 and t <= t1]
        segments.append({"kind": kind, "pts": pts})
        # one marker per surge, at its peak: the start te identifies the surge
        # and its event only changes while the peak still rises
        for e in events:
            if e is not None and t0 <= e[1] <= t1:
                surges[e[2]] = e

    fill_ha = fill_fit.fill_ha()
    if fill_ha is not None:
        for o0, o1 in zip(recs[:-1], recs[1:]):
            ts = step_times(o0["t"], o1["t"])
            ts = ts[ts <= o1["t"]]
            levels = np.array([flood_level(t)[0] for t in ts], float)
            levels = np.round(np.where(np.isfinite(levels), levels, -99.0), 2)
            rest = resting_area(lagoon_model, o0["t"])
            areas, events, _ = fill_drain_series(o0["t"], o0["area_ha"],
                                                 o0["area_ha"] if rest is None else rest,
                                                 ts, levels, fill_ha, fill_fit.drain_days)
            add("replay", o0, ts, areas, events)
    fd = lagoon_water.fill_drain
    if fd is not None:
        add("estimate", latest, fd.ts, fd.area, fd.event)
    else:
        add("estimate", latest, [t1 - t1 % 3600], [latest["area_ha"]], [])
    # the uncertainty band around the estimate (fit.ranges, out-of-sample errors)
    band = []
    for t, a in segments[-1]["pts"]:
        r = lagoon_water.range_at(t, area_ha=a)
        if r is not None:
            band.append([t, round(r[0], 2), round(r[1], 2)])
    return {"t0": t0, "t1": t1, "now": now.timestamp(),
            "images": [{"t": o["t"], "ha": round(o["area_ha"], 2), "group": _obs_group(o),
                        "partial": o.get("seen_basin", 1.0) < CFG["fill_fit_full_view"]}
                       for o in recs],
            "segments": segments, "band": band,
            "surges": [{"t": e[1], "text": "Surge reaches the basin, peaking " + surge_text(e[1])}
                       for _, e in sorted(surges.items())]}


def _search_windows_and_tides(now, water, wind, daylight, lagoon_water, end):
    """Rideable windows over the whole wind horizon, and the next tides."""
    # Only the tidal lagoon mode reads the sea level; the others can search the
    # whole wind horizon even where the tide data runs out.
    model = lagoon_water.model
    until = (water.covers_until() if model is not None and model.mode == "tidal"
             else float("inf"))
    wins = find_windows(now, lagoon_water, wind, daylight, until, step_min=60)
    windows = {"list": [{"when": format_window(win).split(" · ")[0],
                         "detail": " · ".join(format_window(win).split(" · ")[1:]),
                         "verdict": win["verdict"]} for win in wins],
               "none": ("No rideable window in the forecast" if wind is not None
                        else "No wind forecast, so no windows could be found"),
               "caveat": ("" if not lagoon_water.constant else
                          "The lagoon is held at its newest satellite image, so these "
                          "windows follow wind and daylight only." if model is not None else
                          "No lagoon model yet, so the whole outline counts as water and "
                          "these windows follow wind and daylight only.")}
    for win in wins:
        log.info("Window: %s", format_window(win))
    if not wins:
        log.info("No rideable window found in the forecast horizon.")

    horizon_end = (float(wind[0][-1]) if wind is not None
                   else (end + dt.timedelta(days=2)).timestamp())
    tides = [{"kind": f"{x['kind']} water",
              "when": dt.datetime.fromtimestamp(x["t"], UTC).astimezone(TZ).strftime("%a %H:%M"),
              "level": f"{x['level']:+.2f} m"}
             for x in water.extremes(now.timestamp(), horizon_end, max_n=4)]
    return windows, tides


def _render(args, frames, grid, ring, water, terrain_note, warnings, z, windows, tides,
            lagoon_obs, lagoon_label, lagoon_varies, sources, lagoon_trend):
    out = HERE / args.out
    write_html(out, frames, grid, ring, water.location, terrain_note, warnings,
               png_uri(terrain_rgb(z)), windows, tides, lagoon_obs, lagoon_label,
               lagoon_varies, sources, lagoon_trend)
    log.info("Written %s (%.2f MB)", out, out.stat().st_size / 1e6)
    if not args.no_open:
        webbrowser.open(out.as_uri())


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")

    if args.calibrate and not (os.environ.get("CDSE_CLIENT_ID") and os.environ.get("CDSE_CLIENT_SECRET")):
        sys.exit("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET for --calibrate (see top of file).")

    now = dt.datetime.now(UTC)
    now = now.replace(minute=(now.minute // args.step) * args.step if args.step < 60 else 0,
                      second=0, microsecond=0)
    end = now + dt.timedelta(hours=args.hours)
    grid = make_grid(CFG["bbox_wgs84"], CFG["grid_res_m"])
    warnings = []
    sources = []
    CACHE.mkdir(exist_ok=True)
    prune_cache()

    try:
        z_raw, ring, lagoon, basin, terrain_note = _load_terrain_and_outline(
            args, grid, now, warnings, sources)
        lagoon_model, freq, dry_override, fill_fit = _load_or_fit_lagoon_model(
            args, grid, lagoon, basin, ring, z_raw, now, warnings, sources)
        z = prepare_terrain(z_raw, lagoon, basin, fill_fit.basin_gap_fill_m)
        frames, times, water, wind, daylight, lagoon_obs, lagoon_label, lagoon_water, \
            lagoon_trend = _build_frames(args, now, grid, z, lagoon, basin, lagoon_model, freq, dry_override,
                          fill_fit, warnings)
        windows, tides = _search_windows_and_tides(now, water, wind, daylight, lagoon_water, end)
        sources.append(_source("Tide and wind forecasts", dt.datetime.now(UTC), now,
                               "pulled fresh on every run"))
        _render(args, frames, grid, ring, water, terrain_note, warnings, z, windows, tides,
                lagoon_obs, lagoon_label, not lagoon_water.constant, sources, lagoon_trend)
    except ZandmotorError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
