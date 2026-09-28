#!/usr/bin/env python3
"""
Zandmotor lagoon nowcast for kitesurfing
========================================

Builds an interactive HTML map (with a time slider) of the Zandmotor lagoon for
the current hour and the next few hours: where there is water, plus wind and
kite-size advice. Depth isn't modelled or shown - kiting only needs a surface
to ride on, not a particular depth, so the only question that matters is
whether a spot is wet.

How it works
------------
1. Terrain   : AHN elevation (PDOK WCS, m NAP) for the beach and dunes, which
               AHN can measure directly. The lagoon's bed is underwater and
               was never reliably measurable this way, so it isn't modelled
               at all - see "Lagoon" below.
2. Water     : Rijkswaterstaat WaterWebservices (ddapi20): measured level,
               official forecast (incl. surge) and astronomical tide, with
               fallbacks (astro + decaying surge, or harmonic fit via utide).
               Drives tidal flooding of the beach/dunes.
3. Lagoon    : tracked by presence, not elevation. --calibrate tests how much
               of the lagoon is wet against the sea (tidal) and against wind
               gusts, using a year of Sentinel-2 (optical) and Sentinel-1
               (SAR, not cloud-limited) images. On the current record neither
               explains it (see "What the calibration actually found" below),
               so the lagoon is treated as a slowly-varying lake: its area is
               the median of recent satellite observations, reported with the
               spread of those observations rather than as a false forecast.
               The same images also build a per-pixel wetness frequency map,
               so an area can be turned back into an actual shape: the cells
               most often observed wet fill in first.
4. Wind      : Open-Meteo forecast (free, no key), a week ahead, plus
               sunrise/sunset so dark hours aren't offered as sessions.
5. Water temp: Open-Meteo marine API, turned into a wetsuit call.
6. Output    : zandmotor_lagoon.html, open it in any browser.

What counts as rideable
-----------------------
The lagoon plus the intertidal beach strip - never the open sea, which fills
most of the map area and would otherwise dominate every total (it measured
~620 ha against a ~2.8 ha lagoon, which silently disabled the
"is there enough water" test altogether). The sea is still drawn, in a paler
tone, so it's obvious it isn't being counted.

Kite advice comes from the quiver in CFG below: the question is whether a kite
you actually own works right now, capped by rider["skill_ceiling_kn"]. Upper
limits are tested against the gust, not the mean, since that's what you get
hit by.

Install
-------
    pip install requests numpy rasterio pyproj scikit-image pillow scipy
    pip install utide            # optional, extra fallback for the tide

Run
---
    python zandmotor_lagoon.py                  # map for now + next 8 hours
    python zandmotor_lagoon.py --hours 12 --step 30
    python zandmotor_lagoon.py --outline        # trace the lagoon from satellite
                                                # (or OpenStreetMap without account)
    python zandmotor_lagoon.py --calibrate      # refit the lagoon's wetness model
                                                # (needs CDSE credentials, below)

Satellite calibration (optional)
--------------------------------
Create a free account at https://dataspace.copernicus.eu, then in the dashboard
create an OAuth client (Sentinel Hub). Set:
    export CDSE_CLIENT_ID=...
    export CDSE_CLIENT_SECRET=...

Lagoon outline
--------------
--outline traces the lagoon automatically and writes lagoon.geojson plus a
check image (lagoon_outline_check.png, north up) so you can verify it:
  * sentinel : picks a recent cloud-free Sentinel-2 image taken near high
               water, detects water (NDWI) and separates the lagoon from the
               open sea and the dune lake. Needs CDSE credentials (below).
  * osm      : uses the water polygons in OpenStreetMap. No account needed,
               but may be outdated.
  * auto     : sentinel if credentials are set, otherwise osm (default).
If the wrong water body is picked, set CFG["lagoon_hint"] to a (lon, lat) point
inside the lagoon and run again. You can also draw the outline yourself on
https://geojson.io and save it as lagoon.geojson.

What the calibration actually found
-----------------------------------
Current fit (cache/lagoon_response.json, n=64 images: 20 optical + 44 SAR):

    mode = lake      sea level r = -0.05      wind gust r = 0.11

That is: neither the tide nor the wind explains the lagoon's wet area. It sits
around 2.6 ha (observed range 0.02-4.35 ha over a year) and drifts slowly.

An earlier run on a much smaller sample (n=20) found what looked like a strong
wind-gust relationship (12h trailing max gust, r=0.60, p=0.005) and this file
used to describe it as validated. It did not survive the larger sample - at
n=64 the same correlation is r=0.11. Treat the earlier finding as a false
positive from a small sample, not as a suppressed effect. It is recorded here
only so the same lead isn't chased twice.

The SAR wind-gating below is a separate matter and does still hold: a strong
gust shortly before acquisition makes water look like land to radar, so those
scenes are skipped rather than believed.

Not a safety tool. Always check conditions on the spot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import pathlib
import sys
import time
import webbrowser

import numpy as np
import requests
from rasterio.warp import Resampling
from scipy import ndimage

# ----------------------------------------------------------------------------
# Configuration: everything you might want to tweak is here
# ----------------------------------------------------------------------------
from zandmotor.config import CFG, HERE, CACHE, TZ, UTC, log


# ----------------------------------------------------------------------------
# Grid (Web Mercator, so the map overlay lines up exactly in Leaflet)
# ----------------------------------------------------------------------------
from zandmotor.geometry import (
    Grid, make_grid, load_lagoon_polygon, polygon_mask, trim_lagoon_to_water,
    mask_to_ring,
)


# ----------------------------------------------------------------------------
# Terrain
# ----------------------------------------------------------------------------
from zandmotor.terrain import (
    _is_tiff, prune_cache, dem_age_days, fetch_ahn, read_to_grid,
    prepare_terrain, spill_elevation, open_sea_mask, coastal_zone_mask,
    permanent_sea_mask,
)


# ----------------------------------------------------------------------------
# Water level (Rijkswaterstaat ddapi20)
# ----------------------------------------------------------------------------
from zandmotor.time_utils import parse_iso_datetime
from zandmotor.tides import rws_series, WaterLevel


# ----------------------------------------------------------------------------
# Wind (Open-Meteo)
# ----------------------------------------------------------------------------
from zandmotor.wind import (
    fetch_wind, wind_at, is_daylight, fetch_sea_temp, sea_temp_at,
    wetsuit_advice, compass, shore_class, fetch_wind_history, max_gust_window,
)


# ----------------------------------------------------------------------------
# Satellite calibration (Copernicus Data Space, Sentinel-1/2)
# ----------------------------------------------------------------------------
from zandmotor.satellite.client import cdse_token
from zandmotor.satellite.optical import cdse_scenes, cdse_image
from zandmotor.satellite.sar import cdse_sar_scenes, cdse_sar_image, sar_wet_dry


from zandmotor.lagoon_model.calibration import fit_lagoon_response
from zandmotor.lagoon_model.runtime import (
    lake_recent_observations, lake_area_summary, lagoon_area_from_model,
    wetness_fingerprint, wet_mask_from_area,
)
from zandmotor.lagoon_model.dry_override import flats_dry_override


# ----------------------------------------------------------------------------
# Lagoon outline: from satellite (Sentinel-2) or OpenStreetMap
# ----------------------------------------------------------------------------
from zandmotor.rendering import terrain_rgb, png_uri, outline_preview
from zandmotor.outline import (
    lonlat_to_rc, lagoon_candidates, pick_lagoon, save_outline,
    outline_from_sentinel, outline_from_osm, make_outline,
)


# ----------------------------------------------------------------------------
# Water, depth and kite classes
# ----------------------------------------------------------------------------
from zandmotor.flood import flats_wet_map, remove_small_patches
from zandmotor.rendering import class_png
from zandmotor.kiting import kite_advice, quiver_image_uri, kite_wind_range, quiver_advice


# ----------------------------------------------------------------------------
# Rideable windows over the week ahead
# ----------------------------------------------------------------------------
from zandmotor.flood import largest_patch_ha, rideable_curve, rideable_at


from zandmotor.windows import find_windows, format_window


# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------
from zandmotor.html_report.render import write_html


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
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
    args = ap.parse_args()
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
    CACHE.mkdir(exist_ok=True)
    prune_cache()

    # --- terrain (raw) and lagoon outline
    if CFG["local_dem"]:
        z_raw = read_to_grid(CFG["local_dem"], grid, resampling=Resampling.max)
        terrain_note = f"local file {pathlib.Path(CFG['local_dem']).name}"
    else:
        z_raw = fetch_ahn(grid)
        terrain_note = "AHN (PDOK)"
        dem_age = dem_age_days()
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
    lagoon = polygon_mask(grid, ring)

    ring, lagoon, cut_ha = trim_lagoon_to_water(ring, lagoon, z_raw, grid)
    if cut_ha > 0.05:
        warnings.append(f"Cut {cut_ha:.1f} ha of dry land off the traced outline "
                        "(elevation data showed it wasn't water).")
    z = prepare_terrain(z_raw, lagoon)

    # The lagoon's water is tracked by presence, not depth (kiting only needs a
    # surface to ride on) - so there's no bed elevation to calibrate here, just a
    # wetness frequency map built straight from the satellite record: how often
    # each pixel was actually observed wet.
    freq_file = CACHE / "wetness_frequency.npy"
    confidence_file = CACHE / "wetness_confidence.npy"
    response_file = CACHE / "lagoon_response.json"
    meta_file = CACHE / "wetness_meta.json"
    fingerprint = wetness_fingerprint(lagoon, ring)
    if args.calibrate:
        response, freq, valid_count = fit_lagoon_response(grid, lagoon, now)
        if response is not None:
            response_file.write_text(json.dumps(response, indent=1))
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
                lagoon_model = json.loads(response_file.read_text())
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
                lagoon_model = json.loads(response_file.read_text())
            else:
                freq = None
    if lagoon_model is not None:
        mode = lagoon_model["mode"]
        mode_txt = {"tidal": "tidal", "wind": "wind-driven (experimental)",
                    "lake": "a slowly-varying lake"}[mode]
        age = (now - parse_iso_datetime(lagoon_model["fitted"])).days
        log.info("Lagoon response model: %s (r=%.2f, n=%d, fitted %d days ago)",
                 mode_txt, lagoon_model["r"], lagoon_model["n"], age)
        if mode == "wind":
            n_opt, n_sar = lagoon_model.get("n_optical", "?"), lagoon_model.get("n_sar", "?")
            warnings.append(f"Lagoon wet area was fitted to wind gusts rather than tide "
                            f"(r={lagoon_model['r']:.2f} over the {lagoon_model['window_h']}h before "
                            f"each hour, n={lagoon_model['n']} images: {n_opt} optical + {n_sar} SAR). "
                            "Treat this as unproven: an earlier version of this fit looked strong at "
                            "n=20 and collapsed to r=0.11 at n=64. Sea level showed ~no relationship "
                            f"(r={lagoon_model.get('sea_r', 0):.2f}).")
        if age > 90:
            warnings.append(f"Lagoon response model ({mode_txt}, fitted {age} days ago) "
                            "may be stale; run --calibrate to refit.")
    else:
        warnings.append("No satellite-fitted lagoon response model yet; showing the "
                        "whole traced outline as wet until one exists. Run --calibrate "
                        "to fit one from a year of satellite images.")

    dry_override = flats_dry_override(grid, lagoon, now)
    if dry_override is None:
        warnings.append("Could not refresh the satellite dry-ground check for the "
                        "flats this run; showing the elevation-only estimate there.")

    F = spill_elevation(z)
    coastal_zone = coastal_zone_mask(z, lagoon)

    # --- water & wind
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

    # --- frames
    frames = []
    n = int(args.hours * 60 / args.step) + 1
    times = [now + dt.timedelta(minutes=k * args.step) for k in range(n)]

    # Lagoon area first: in lake mode it is one measured value for the whole run
    # (nothing in the record predicts it), so it also fixes what the window
    # search should assume about water.
    lagoon_obs, lake_area = None, None
    lagoon_label = ""
    if lagoon_model is not None and freq is not None and lagoon_model["mode"] == "lake":
        med, lo, hi, as_of, n_used = lake_area_summary(lagoon_model, now)
        if med is not None:
            lake_area = med
            lagoon_label = (f"{med:.1f} ha ({lo:.1f}–{hi:.1f}), "
                            f"last seen {as_of.astimezone(TZ):%d %b}")
            all_t, all_a = np.array(lagoon_model["times"]), np.array(lagoon_model["areas"])
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
        tidal_wet, ponded_wet = flats_wet_map(z, F, sea_level, coastal_zone, lagoon)
        wet = remove_small_patches(tidal_wet | lagoon_wet, grid)
        if dry_override is not None:
            wet &= ~dry_override
        # The open sea and the isolated dune ponds are drawn but never counted:
        # ~610 ha and ~21 ha against a ~2.8 ha lagoon. Folded into one total they
        # made the "enough water" test impossible to fail.
        rideable = wet & ~sea
        uncounted = (wet & sea) | remove_small_patches(ponded_wet & ~sea, grid)
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
                       "dark": dark, "water_c": water_c,
                       "wetsuit": suit, "wetsuit_note": suit_note,
                       "quiver": quiver,
                       "png": class_png(rideable, uncounted)})
        log.info("%s  sea %+.2f  lagoon %4.1f ha  shallows %5.1f ha  rideable %5.1f ha  "
                 "biggest patch %5.1f ha  %-5s %s [%s]", tl.strftime("%a %H:%M"), sea_level,
                 lagoon_ha, shallows_ha, rideable_ha, usable_ha, verdict,
                 "dark" if dark else "    ", src)

    # The window search reads areas off the precomputed curve rather than
    # rebuilding every mask, so the two paths could drift. What matters is not
    # that the hectares match exactly - the curve interpolates a steep,
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

    # --- rideable windows over the whole wind horizon, and the next tides
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

    out = HERE / args.out
    write_html(out, frames, grid, ring, water.location, terrain_note, warnings,
               png_uri(terrain_rgb(z)), windows, tides, lagoon_obs, lagoon_label)
    log.info("Written %s (%.2f MB)", out, out.stat().st_size / 1e6)
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
