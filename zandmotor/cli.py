"""Command-line entry point: argument parsing and run orchestration."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import pathlib
import sys
import webbrowser

import numpy as np
import requests
from rasterio.warp import Resampling

from zandmotor.config import CACHE, CFG, HERE, TZ, UTC, log
from zandmotor.errors import ZandmotorError
from zandmotor.flood import compute_wet_masks, largest_patch_ha, rideable_at, rideable_curve
from zandmotor.geometry import load_lagoon_polygon, make_grid, polygon_mask, trim_lagoon_to_water
from zandmotor.html_report.render import write_html
from zandmotor.kiting import kite_advice, quiver_advice
from zandmotor.lagoon_model.calibration import fit_lagoon_response
from zandmotor.lagoon_model.dry_override import flats_dry_override
from zandmotor.lagoon_model.runtime import (
    lagoon_area_from_model, lake_area_summary, lake_recent_observations,
    wet_mask_from_area, wetness_fingerprint,
)
from zandmotor.lagoon_model.types import model_from_dict
from zandmotor.outline import make_outline
from zandmotor.rendering import class_png, png_uri, terrain_rgb
from zandmotor.terrain import (
    coastal_zone_mask, dem_age_days, fetch_ahn, permanent_sea_mask, prepare_terrain,
    prune_cache, read_to_grid, spill_elevation,
)
from zandmotor.tides import WaterLevel
from zandmotor.time_utils import parse_iso_datetime
from zandmotor.wind import fetch_sea_temp, fetch_wind, is_daylight, sea_temp_at, wetsuit_advice, wind_at
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
    the elevation data actually shows as water. Returns (z, ring, lagoon,
    terrain_note)."""
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
        warnings.append(f"Cut {cut_ha:.1f} ha of dry land off the traced outline "
                        "(elevation data showed it wasn't water).")
    z = prepare_terrain(z_raw, lagoon)
    return z, ring, lagoon, terrain_note


def _load_or_fit_lagoon_model(args, grid, lagoon, ring, now, warnings, sources):
    """Loads the cached lagoon wetness model (refitting first if --calibrate),
    verifying it was fitted against the current outline. Also refreshes the
    satellite dry-ground override for the flats. Returns (lagoon_model, freq,
    dry_override)."""
    freq_file = CACHE / "wetness_frequency.npy"
    confidence_file = CACHE / "wetness_confidence.npy"
    response_file = CACHE / "lagoon_response.json"
    meta_file = CACHE / "wetness_meta.json"
    fingerprint = wetness_fingerprint(lagoon, ring)
    if args.calibrate:
        response, freq, valid_count = fit_lagoon_response(grid, lagoon, now)
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
            warnings.append("The lagoon outline has changed since the wetness map was fitted, so "
                            "the map no longer describes this footprint and is being ignored. "
                            "Run --calibrate to refit.")
        else:
            freq = np.load(freq_file)
            if freq.shape == lagoon.shape:
                lagoon_model = model_from_dict(json.loads(response_file.read_text()))
            else:
                freq = None
    if lagoon_model is not None:
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
        # The lagoon's size only moves when new images are pulled in, so how old
        # the newest one is says how current the "Lagoon water" figure is.
        latest = (dt.datetime.fromtimestamp(max(lagoon_model.times), UTC)
                  if lagoon_model.times else None)
        stale = CFG["lagoon_obs_stale_days"]
        sources.append(_source("Satellite images (lagoon size)", latest, now,
                               f"newest image; last pulled "
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
    return lagoon_model, freq, dry_override


def _build_frames(args, now, grid, z, lagoon, lagoon_model, freq, dry_override, warnings):
    """Fetches tide/wind/sea-temp, precomputes the rideable-area curve, and
    builds the per-hour frames. Returns (frames, times, curve, water, wind,
    daylight, lagoon_obs, lagoon_label, lake_area)."""
    F = spill_elevation(z)
    coastal_zone = coastal_zone_mask(z, lagoon)

    lat_c = (CFG["bbox_wgs84"][1] + CFG["bbox_wgs84"][3]) / 2
    lon_c = (CFG["bbox_wgs84"][0] + CFG["bbox_wgs84"][2]) / 2
    # Ask for tide data out to the end of the wind forecast, not just the slider
    # range: the window search walks the whole week, and without coverage
    # WaterLevel.level falls through to the harmonic fit (which exits if utide
    # isn't installed).
    water = WaterLevel(now, now + dt.timedelta(days=8))
    wind = fetch_wind(lat_c, lon_c)
    sea_temp = fetch_sea_temp(lat_c, lon_c)
    wind_hist = (wind[0], wind[2]) if wind is not None else None
    daylight = wind[5] if wind is not None else []

    frames = []
    n = int(args.hours * 60 / args.step) + 1
    times = [now + dt.timedelta(minutes=k * args.step) for k in range(n)]

    # Lagoon area first: in lake mode it is one measured value for the whole run
    # (nothing in the record predicts it), so it also fixes what the window
    # search should assume about water.
    lagoon_obs, lake_area = None, None
    lagoon_label = ""
    if lagoon_model is not None and freq is not None and lagoon_model.mode == "lake":
        med, lo, hi, as_of, n_used = lake_area_summary(lagoon_model, now)
        if med is not None:
            lake_area = med
            lagoon_label = (f"{med:.1f} ha ({lo:.1f}–{hi:.1f}), "
                            f"last seen {as_of.astimezone(TZ):%d %b}")
            all_t, all_a = np.array(lagoon_model.times), np.array(lagoon_model.areas)
            used_t, _ = lake_recent_observations(lagoon_model, now)
            used_set = set(used_t.tolist()) if used_t is not None else set()
            lagoon_obs = {
                "times": all_t.tolist(), "areas": [round(a, 2) for a in all_a.tolist()],
                "used": [bool(t in used_set) for t in all_t.tolist()],
                "median": round(med, 2), "lo": round(lo, 2), "hi": round(hi, 2),
                "label": (f"Lagoon, {n_used} of {len(all_t)} satellite observations used "
                          f"(median and middle half) — measured, not forecast")}
            log.info("Lagoon area (lake mode): median %.2f ha, p25-p75 %.2f-%.2f, "
                     "from %d of %d observations, newest %s",
                     med, lo, hi, n_used, len(all_t), as_of.date())

    # Wet-area-vs-sea-level curve, so the week-long window search doesn't rerun
    # the flood model at every candidate hour.
    sea_levels = np.array([water.level(t.timestamp())[0] for t in times])
    datum = water.low_water_datum()
    sea = permanent_sea_mask(z, lagoon, datum)
    log.info("Open sea below the %.2f m NAP low-water datum (excluded from every area "
             "figure): %.0f ha of the %.0f ha map", datum,
             sea.sum() * grid.cell_area_m2 / 1e4,
             grid.width * grid.height * grid.cell_area_m2 / 1e4)
    # span the whole tidal range, not just this run's, so the window search can
    # interpolate anywhere the week goes
    curve_levels = np.linspace(min(float(sea_levels.min()), datum) - 0.3,
                               float(sea_levels.max()) + 1.0, 61)
    if lagoon_model is not None and freq is not None:
        curve_lagoon_wet = wet_mask_from_area(
            freq, lagoon,
            lake_area if lake_area is not None
            else lagoon_area_from_model(lagoon_model, water.level, now.timestamp(), wind_hist),
            grid.cell_area_m2)
    else:
        curve_lagoon_wet = lagoon.copy()
    curve = (curve_levels,) + rideable_curve(z, F, coastal_zone, lagoon, sea,
                                             curve_lagoon_wet, dry_override, grid, curve_levels)

    for t in times:
        te = t.timestamp()
        sea_level, src = water.level(te)
        if lagoon_model is not None and freq is not None:
            lagoon_ha = lagoon_area_from_model(lagoon_model, water.level, te, wind_hist,
                                               lake_area=lake_area)
            lagoon_wet = wet_mask_from_area(freq, lagoon, lagoon_ha, grid.cell_area_m2)
        else:
            lagoon_wet = lagoon.copy()
            lagoon_ha = float(lagoon.sum() * grid.cell_area_m2 / 1e4)
        # The open sea and the isolated dune ponds are drawn but never counted:
        # ~610 ha and ~21 ha against a ~2.8 ha lagoon. Folded into one total they
        # made the "enough water" test impossible to fail.
        rideable, uncounted = compute_wet_masks(z, F, sea_level, coastal_zone, lagoon, sea,
                                                lagoon_wet, dry_override, grid)
        # report the areas actually drawn, after small-patch removal
        lagoon_ha = float((rideable & lagoon).sum() * grid.cell_area_m2 / 1e4)
        shallows_ha = float((rideable & ~lagoon).sum() * grid.cell_area_m2 / 1e4)
        rideable_ha = lagoon_ha + shallows_ha
        usable_ha = largest_patch_ha(rideable, grid.cell_area_m2)

        w = wind_at(wind, te)
        quiver = quiver_advice(w, CFG["rider"])
        wind_txt, verdict, notes = kite_advice(w, usable_ha, quiver)
        dark = not is_daylight(daylight, te)
        if dark:
            verdict = "no"
            notes = notes + ["dark"]
        water_c = sea_temp_at(sea_temp, te)
        suit, suit_note = wetsuit_advice(water_c, w)
        tl = t.astimezone(TZ)
        frames.append({"time": tl.strftime("%H:%M"), "day": tl.strftime("%A %d %B"),
                       "sea": sea_level, "lagoon_ha": lagoon_ha, "src": src,
                       "shallows_ha": shallows_ha, "rideable_ha": rideable_ha,
                       "usable_ha": usable_ha,
                       "wind": wind_txt, "verdict": verdict, "notes": notes,
                       # raw numbers for the map's compass rose (None without a forecast)
                       "wind_deg": None if w is None else round(w[2], 1),
                       "wind_kn": None if w is None else round(w[0], 1),
                       "gust_kn": None if w is None else round(w[1], 1),
                       "dark": dark, "water_c": water_c,
                       "wetsuit": suit, "wetsuit_note": suit_note,
                       "quiver": quiver,
                       "png": class_png(rideable, uncounted)})
        log.info("%s  sea %+.2f  lagoon %4.1f ha  shallows %5.1f ha  rideable %5.1f ha  "
                 "biggest patch %5.1f ha  %-5s %s [%s]", tl.strftime("%a %H:%M"), sea_level,
                 lagoon_ha, shallows_ha, rideable_ha, usable_ha, verdict,
                 "dark" if dark else "    ", src)

    # Both paths now run through the same compute_wet_masks, so this is no
    # longer a check for two divergent implementations - but the window search
    # still reads areas off a sea-level-sampled, interpolated curve rather
    # than the frame loop's exact per-hour computation, and that
    # approximation could still disagree near a threshold. What matters is
    # not that the hectares match exactly - the curve interpolates a steep,
    # patch-merging relationship and will be off by some margin mid-tide - but
    # that they never reach a different VERDICT for the same hour, which is
    # what the sidebar and the "next window" line would then disagree about.
    clashes = []
    for f, t in zip(frames, times):
        _, _, _, _, curve_verdict, _, _, _ = rideable_at(t.timestamp(), water, wind, curve)
        if not is_daylight(daylight, t.timestamp()):
            curve_verdict = "no"
        if curve_verdict != f["verdict"]:
            clashes.append(f'{f["time"]}: frame {f["verdict"]} vs window search {curve_verdict}')
    if clashes:
        log.warning("Window search disagrees with the hourly frames on %d hour(s): %s",
                    len(clashes), "; ".join(clashes))
    else:
        log.debug("Window search agrees with all %d hourly frames", len(frames))

    return frames, times, curve, water, wind, daylight, lagoon_obs, lagoon_label, lake_area


def _search_windows_and_tides(now, water, wind, daylight, curve, lake_area, end):
    """Rideable windows over the whole wind horizon, and the next tides."""
    wins = find_windows(now, water, wind, curve, daylight, step_min=60)
    windows = {"list": [{"when": format_window(win).split(" · ")[0],
                         "detail": " · ".join(format_window(win).split(" · ")[1:]),
                         "verdict": win["verdict"]} for win in wins],
               "none": ("No rideable window in the forecast" if wind is not None
                        else "No wind forecast, so no windows could be found"),
               "caveat": ("Lagoon area is a constant in lake mode, so these windows follow "
                          "wind, tide and daylight — not the lagoon."
                          if lake_area is not None else "")}
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
           lagoon_obs, lagoon_label, sources):
    out = HERE / args.out
    write_html(out, frames, grid, ring, water.location, terrain_note, warnings,
               png_uri(terrain_rgb(z)), windows, tides, lagoon_obs, lagoon_label, sources)
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
        z, ring, lagoon, terrain_note = _load_terrain_and_outline(args, grid, now, warnings, sources)
        lagoon_model, freq, dry_override = _load_or_fit_lagoon_model(
            args, grid, lagoon, ring, now, warnings, sources)
        frames, times, curve, water, wind, daylight, lagoon_obs, lagoon_label, lake_area = \
            _build_frames(args, now, grid, z, lagoon, lagoon_model, freq, dry_override, warnings)
        windows, tides = _search_windows_and_tides(now, water, wind, daylight, curve, lake_area, end)
        sources.append(_source("Tide and wind forecasts", dt.datetime.now(UTC), now,
                               "pulled fresh on every run"))
        _render(args, frames, grid, ring, water, terrain_note, warnings, z, windows, tides,
               lagoon_obs, lagoon_label, sources)
    except ZandmotorError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
