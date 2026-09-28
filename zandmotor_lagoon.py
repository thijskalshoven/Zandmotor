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
import base64
import datetime as dt
import io
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
from PIL import Image
from pyproj import Transformer
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
def lonlat_to_rc(grid, lon, lat):
    x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
    col, row = ~grid.transform * (x, y)
    return int(row), int(col)


def lagoon_candidates(wet, grid):
    """Split the water mask into separate water bodies. The lagoon is often
    joined to the sea by a narrow channel, so we erode until the sea and the
    lagoon come apart, then grow each part back (watershed).
    Returns a list of (mask, joined_to_sea) for every body except the sea."""
    from skimage.segmentation import watershed
    wet = ndimage.binary_opening(wet, iterations=1)
    min_px = CFG["outline_min_ha"] * 1e4 / grid.cell_area_m2
    lab0, _ = ndimage.label(wet)
    sea0 = set(np.unique(np.r_[lab0[0], lab0[-1], lab0[:, 0], lab0[:, -1]])) - {0}
    dist = ndimage.distance_transform_edt(wet)
    for k in range(1, 16):
        er = dist > k
        el, n = ndimage.label(er)
        if n == 0:
            break
        edge = set(np.unique(np.r_[el[0], el[-1], el[:, 0], el[:, -1]])) - {0}
        inner = [i for i in range(1, n + 1) if i not in edge]
        if not inner:
            continue
        regions = watershed(-dist, el, mask=wet)
        out = []
        for i in inner:
            m = regions == i
            if m.sum() >= min_px:
                joined = bool(set(np.unique(lab0[m])) & sea0)
                out.append((ndimage.binary_fill_holes(m), joined))
        # also water bodies that never touched the sea (closed-off lagoon, lake)
        for lab_id in set(range(1, lab0.max() + 1)) - sea0:
            m = lab0 == lab_id
            if m.sum() >= min_px and not any((m & c).any() for c, _ in out):
                out.append((ndimage.binary_fill_holes(m), False))
        if out:
            return out
    return []


def pick_lagoon(cands, grid):
    if not cands:
        return None
    for m, joined in cands:
        ys, xs = np.nonzero(m)
        x, y = grid.transform * (xs.mean(), ys.mean())
        lon, lat = Transformer.from_crs(3857, 4326, always_xy=True).transform(x, y)
        log.info("  water body: %.1f ha at %.5f, %.5f%s", m.sum() * grid.cell_area_m2 / 1e4,
                 lon, lat, " (joined to sea)" if joined else "")
    hint = CFG["lagoon_hint"]
    if hint:
        r, c = lonlat_to_rc(grid, *hint)
        for m, _ in cands:
            if 0 <= r < m.shape[0] and 0 <= c < m.shape[1] and m[r, c]:
                return m
        log.warning("lagoon_hint is not inside any water body; picking automatically")
    joined = [m for m, j in cands if j]
    pool = joined or [m for m, _ in cands]
    return max(pool, key=lambda m: m.sum())


def save_outline(ring, props):
    gj = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": props,
        "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in ring]]}}]}
    (HERE / "lagoon.geojson").write_text(json.dumps(gj, indent=1))
    log.info("Written lagoon.geojson (%s)", props.get("source"))


def terrain_rgb(z):
    """Colour the terrain: sea, wet sand, beach, dunes, with hillshade."""
    stops = [(-8, (52, 98, 128)), (-2, (86, 138, 166)), (-0.4, (140, 184, 200)),
             (0.3, (214, 203, 170)), (2.0, (233, 222, 190)), (4.0, (214, 208, 164)),
             (6.0, (166, 178, 132)), (12.0, (122, 142, 104))]
    zs = [a for a, _ in stops]
    rgb = np.stack([np.interp(z, zs, [c[i] for _, c in stops]) for i in range(3)], -1)
    gy, gx = np.gradient(z, CFG["grid_res_m"])
    # sun from the north-west, 45 degrees high
    slope = np.arctan(np.hypot(gx, gy) * 2.0)
    aspect = np.arctan2(-gx, gy)
    az, alt = math.radians(315), math.radians(45)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(0.55 + 0.6 * shade, 0.55, 1.15)
    land = z > 0.3
    rgb[land] *= shade[land, None]
    return np.clip(rgb, 0, 255).astype("uint8")


def png_uri(rgb_or_rgba):
    mode = "RGBA" if rgb_or_rgba.shape[-1] == 4 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(rgb_or_rgba, mode).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def outline_preview(rgb, ring, grid, title, path):
    """North-up check image: background, outline, north arrow, scale bar."""
    from PIL import ImageDraw, ImageFont
    scale = max(1, int(math.ceil(1100 / grid.width)))
    img = Image.fromarray(rgb, "RGB").resize((grid.width * scale, grid.height * scale),
                                             Image.NEAREST if scale > 1 else Image.BILINEAR)
    d = ImageDraw.Draw(img)
    to_merc = Transformer.from_crs(4326, 3857, always_xy=True)
    inv = ~grid.transform
    pts = []
    for lon, lat in ring:
        c, r = inv * to_merc.transform(lon, lat)
        pts.append((c * scale, r * scale))
    d.line(pts, fill=(240, 127, 30), width=4, joint="curve")
    try:
        font = ImageFont.load_default(size=22)
        small = ImageFont.load_default(size=16)
    except TypeError:
        font = small = ImageFont.load_default()
    W, H = img.size
    # north arrow
    ax, ay = W - 50, 30
    d.polygon([(ax, ay), (ax - 14, ay + 40), (ax, ay + 30), (ax + 14, ay + 40)],
              fill=(16, 41, 58), outline=(255, 255, 255))
    d.text((ax - 7, ay + 44), "N", fill=(16, 41, 58), font=font,
           stroke_width=3, stroke_fill=(255, 255, 255))
    # scale bar 500 m
    px = 500 / CFG["grid_res_m"] * scale
    x0, y0 = 30, H - 40
    d.rectangle([x0 - 6, y0 - 26, x0 + px + 50, y0 + 14], fill=(255, 255, 255))
    d.rectangle([x0, y0, x0 + px, y0 + 6], fill=(16, 41, 58))
    d.text((x0, y0 - 22), "500 m", fill=(16, 41, 58), font=small)
    d.rectangle([0, 0, W, 34], fill=(255, 255, 255))
    d.text((12, 6), title, fill=(16, 41, 58), font=small)
    img.save(path)
    log.info("Written %s", path.name)


def outline_from_sentinel(grid, now):
    token = cdse_token()
    days = CFG["outline_days_back"]
    scenes = cdse_scenes(token, now, days_back=days, max_scenes=40)
    if not scenes:
        raise RuntimeError(f"no cloud-free Sentinel-2 images in the last {days} days")
    wl = WaterLevel(now, now, history_days=days + 2)
    lag = CFG["lagoon_lag_min"] * 60
    ranked = []
    for t in scenes:
        h, src = wl.level(t.timestamp() - lag)
        if src == "measured":
            ranked.append((h, t))
    ranked.sort(reverse=True)          # fullest lagoon first
    for h, t in ranked[:6]:
        log.info("Trying image %s (water level %.2f m NAP)", t.strftime("%Y-%m-%d %H:%M"), h)
        ndwi, scl, rgb = cdse_image(token, grid, t)
        clear = np.isin(scl.round(), [4, 5, 6, 7, 11])
        if clear.mean() < 0.8:
            log.info("  too cloudy over the area, next")
            continue
        m = pick_lagoon(lagoon_candidates((ndwi > CFG["ndwi_wet"]) & clear, grid), grid)
        if m is None:
            continue
        ring = mask_to_ring(m, grid)
        stretched = np.clip(rgb / 0.25 * 255, 0, 255).astype("uint8")
        title = (f"Sentinel-2 {t.astimezone(TZ):%d %b %Y %H:%M}, water {h:+.2f} m NAP, "
                 f"lagoon {m.sum() * grid.cell_area_m2 / 1e4:.1f} ha")
        outline_preview(stretched, ring, grid, title, HERE / "lagoon_outline_check.png")
        return ring, {"source": "sentinel-2", "image_time": t.isoformat(),
                      "water_level_m_nap": round(h, 2)}
    raise RuntimeError("no usable satellite image found (clouds or no lagoon detected)")


def outline_from_osm(grid):
    s_, w_, n_, e_ = (CFG["bbox_wgs84"][1], CFG["bbox_wgs84"][0],
                      CFG["bbox_wgs84"][3], CFG["bbox_wgs84"][2])
    bb = f"({s_},{w_},{n_},{e_})"
    q = (f'[out:json][timeout:60];(way["natural"="water"]{bb};way["water"="lagoon"]{bb};'
         f'way["natural"="bay"]{bb};relation["natural"="water"]{bb};);out geom;')
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=90,
                       headers={"User-Agent": "zandmotor-lagoon-nowcast/1.0 (kitesurf forecast tool)"})
    r.raise_for_status()
    to_merc = Transformer.from_crs(4326, 3857, always_xy=True)
    cands = []
    for el in r.json().get("elements", []):
        rings = []
        if el["type"] == "way" and el.get("geometry"):
            rings = [el["geometry"]]
        elif el["type"] == "relation":
            rings = [m["geometry"] for m in el.get("members", [])
                     if m.get("role") == "outer" and m.get("geometry")]
        for g in rings:
            ring = [(p["lon"], p["lat"]) for p in g]
            if len(ring) < 4 or ring[0] != ring[-1]:
                continue
            xy = np.array([to_merc.transform(*p) for p in ring])
            k = math.cos(math.radians(ring[0][1])) ** 2
            area_ha = 0.5 * abs(np.dot(xy[:-1, 0], xy[1:, 1]) -
                                np.dot(xy[1:, 0], xy[:-1, 1])) * k / 1e4
            tags = el.get("tags", {})
            cands.append((ring, area_ha, tags))
            log.info("  OSM water: %-25s %5.1f ha  %s", tags.get("name", "(no name)"), area_ha,
                     tags.get("water", ""))
    cands = [c for c in cands if c[1] >= CFG["outline_min_ha"]]
    if not cands:
        raise RuntimeError("no water polygons found in OpenStreetMap for this area")
    hint = CFG["lagoon_hint"]
    if hint:
        for ring, a, t in cands:
            m = polygon_mask(grid, ring)
            r_, c_ = lonlat_to_rc(grid, *hint)
            if 0 <= r_ < m.shape[0] and 0 <= c_ < m.shape[1] and m[r_, c_]:
                return ring, {"source": "openstreetmap", "name": t.get("name", "")}
    def score(c):
        ring, a, t = c
        name = t.get("name", "").lower()
        return (t.get("water") == "lagoon" or "lagune" in name or "lagoon" in name,
                "meer" not in name and t.get("water") != "lake", a)
    ring, a, t = max(cands, key=score)
    return ring, {"source": "openstreetmap", "name": t.get("name", "")}


def make_outline(grid, now, method, z=None):
    """Returns (ring, props). Writes lagoon.geojson and a check image."""
    if method == "auto":
        method = "sentinel" if os.environ.get("CDSE_CLIENT_ID") else "osm"
    log.info("Tracing lagoon outline (%s)...", method)
    if method == "sentinel":
        ring, props = outline_from_sentinel(grid, now)
    else:
        ring, props = outline_from_osm(grid)
        if z is not None:
            outline_preview(terrain_rgb(z), ring, grid,
                            f"OpenStreetMap outline on terrain model ({props.get('name') or 'lagoon'})",
                            HERE / "lagoon_outline_check.png")
    save_outline(ring, props)
    return ring


# ----------------------------------------------------------------------------
# Water, depth and kite classes
# ----------------------------------------------------------------------------
def flats_wet_map(z, F, sea_level, coastal_zone, lagoon):
    """Which cells outside the lagoon hold water right now, split into
    (tidal, ponded).

    `tidal` is water connected to the sea - the intertidal beach, which is what
    "beach shallows" means and which counts as rideable. `ponded` is standing
    water with no tidal route to the sea: dune lakes, ditches, wet dune
    valleys. Those are drawn but never counted, for two reasons - they are not
    beach, and they don't change with the tide, so including them put a
    constant ~21 ha in front of the "enough water" test and made it pass every
    hour regardless of conditions. Same failure as counting the open sea.
    Depth doesn't matter for kiting, only presence of water, so this is
    boolean, not a depth value - but a cell only counts as flooded if the
    lowest point on its route to the sea is at least flood_margin_m below
    the water level (a thin, averaged-down sand ridge shouldn't leak) and
    the resulting film is at least min_visible_depth_m deep (anything
    thinner is a 5 m-grid artefact, not real standing water). Restricted to
    coastal_zone (open sea + lagoon only, see coastal_zone_mask): elevation
    alone can't tell a real dune-backed beach from a diked, pumped-dry town
    that happens to sit below sea level too.

    The beach and dunes are normal dry land, occasionally wet - AHN's laser
    measures them directly, so this elevation-based bathtub model is right
    for them. It's deliberately NOT used for the lagoon: its bed is
    underwater and was never reliably measurable this way (see
    wet_mask_from_area for how the lagoon is handled instead)."""
    flats = coastal_zone & ~lagoon
    connected = flats & (F < sea_level - CFG["flood_margin_m"])
    trapped = flats & ~connected & (z < F - CFG["pond_min_depth"])
    thin = CFG["min_visible_depth_m"]
    tidal = connected & (sea_level - z >= thin)
    ponded = trapped & (F - CFG["pond_drawdown"] - z >= thin)
    return tidal, ponded


WATER_COLOR = (56, 126, 168, 220)     # rideable: lagoon + intertidal beach
SEA_COLOR = (56, 126, 168, 70)        # open sea: drawn, but never counted


def remove_small_patches(wet, grid):
    """Drop wet blobs smaller than min_patch_ha: isolated puddles left by
    diagonal leaks or DEM/observation noise, none of it rideable anyway."""
    struct = ndimage.generate_binary_structure(2, 1)      # 4-connected
    lab, n = ndimage.label(wet, structure=struct)
    if n == 0:
        return wet
    min_px = CFG["min_patch_ha"] * 1e4 / grid.cell_area_m2
    counts = ndimage.sum(wet, lab, index=np.arange(1, n + 1))
    small_ids = np.nonzero(counts < min_px)[0] + 1
    if len(small_ids) == 0:
        return wet
    wet = wet.copy()
    wet[np.isin(lab, small_ids)] = False
    return wet


def class_png(rideable, sea):
    """Two-tone water overlay: what counts as rideable in full colour, the
    open sea in a paler tone. They used to be one mask summed into one
    number, which let ~620 ha of North Sea stand in for a 2.8 ha lagoon."""
    rgba = np.zeros(rideable.shape + (4,), "uint8")
    rgba[sea] = SEA_COLOR
    rgba[rideable] = WATER_COLOR
    return png_uri(rgba)


def kite_advice(w, usable_ha, quiver):
    """Verdict for one hour: is there water, does a kite you own fit, and is
    anything about the wind unpleasant.

    usable_ha is the largest CONNECTED wet patch (see largest_patch_ha), not a
    total - the total is what let ~620 ha of North Sea, and later 250 ha of
    scattered beach, stand in for a rideable surface.

    The kite question is answered by the quiver (see quiver_advice), not by an
    abstract wind table. There used to be both, and they contradicted each
    other - the table called 26 kn "too strong for a beginner" while the
    quiver offered a 7 m as good to 37 kn.
    Returns (wind_txt, verdict, notes)."""
    notes = []
    water_ok = usable_ha >= CFG["min_rideable_ha"]
    if not water_ok:
        notes.append(f"largest patch of water only {usable_ha:.1f} ha "
                     f"(want {CFG['min_rideable_ha']:.0f})")
    if w is None:
        return "No wind data", "no", notes + ["no wind forecast"]

    spd, gst, deg = w[0], w[1], w[2]
    dir_cls = shore_class(deg)
    wind_txt = f"{spd:.0f} kn, gusts {gst:.0f} kn, {compass(deg)} ({dir_cls})"
    kite_ok = quiver["best"] is not None
    if not kite_ok:
        notes.append(quiver["conclusion"].lower())
    if quiver.get("over_ceiling"):
        notes.append(f"gusts past your {CFG['rider']['skill_ceiling_kn']:.0f} kn ceiling")
    if dir_cls == "offshore":
        notes.append("offshore wind, gusty over the dunes")
    if gst - spd >= CFG["gusty_delta_kn"]:
        notes.append("gusty")

    if not (water_ok and kite_ok) or quiver.get("over_ceiling"):
        verdict = "no"
    elif notes or quiver["best_status"] == "marginal":
        verdict = "maybe"
    else:
        verdict = "go"
    return wind_txt, verdict, notes


def quiver_image_uri(filename, warnings):
    """Embed a kite photo from CFG['kite_images_dir'] as a data URI, so the
    output HTML stays a single portable file. Swap in a replacement photo by
    keeping the same filename; a missing/renamed file degrades to a blank
    thumbnail plus a warning instead of a broken page.

    Downscaled to kite_thumb_px before embedding: the originals are up to
    1.6 MB each and are displayed at 56 px, which put 3.4 MB of invisible
    detail into a 4.6 MB page."""
    path = HERE / CFG["kite_images_dir"] / filename
    try:
        img = Image.open(io.BytesIO(path.read_bytes()))
    except (OSError, ValueError):
        warnings.append(f"Kite photo '{filename}' not found or unreadable in "
                        f"{CFG['kite_images_dir']}/; showing a blank thumbnail.")
        return ""
    n = CFG["kite_thumb_px"]
    img = img.convert("RGB")
    img.thumbnail((n, n), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def kite_wind_range(kite, rider):
    """This kite's usable wind range for this rider, in knots.

    Manufacturer charts assume a 75 kg rider. Shift the range's CENTRE by
    weight/75 (for a fixed kite size the usable wind scales with rider weight,
    per size_m2 = weight_kg*2.2/wind_kn) but keep its WIDTH - how wide a
    window a kite has is a property of its depower and the rider's skill, and
    does not stretch because the rider is heavier. Scaling both endpoints, as
    this used to, inflated an 11-25 kn chart into 12.9-29.3 and turned a
    formula-estimated 7 m into "good to 37 kn".

    The top is then capped at skill_ceiling_kn - what a kite can technically
    hold and what you want to be out in are different numbers.
    Returns (lo, hi, estimated, capped)."""
    if kite["wind_range_75kg"] is not None:
        lo75, hi75 = kite["wind_range_75kg"]
        estimated = False
    else:
        centre75 = 75 * 2.2 / kite["size_m2"]
        f = CFG["kite_est_range_frac"]
        lo75, hi75 = centre75 * (1 - f), centre75 * (1 + f)
        estimated = True
    half = (hi75 - lo75) / 2
    centre = (lo75 + hi75) / 2 * rider["weight_kg"] / 75.0
    lo, hi = max(centre - half, 0.0), centre + half
    ceiling = rider.get("skill_ceiling_kn")
    capped = ceiling is not None and hi > ceiling
    if capped:
        hi = float(ceiling)
    return min(lo, hi), hi, estimated, capped


def quiver_advice(w, rider):
    """Given the current wind and the rider's own quiver (CFG['quiver']),
    work out which owned kite fits best right now.

    Ranges come from kite_wind_range (see there for the weight shift and the
    skill ceiling). The LOWER bound is tested against the mean wind and the
    UPPER against the gust: the gust is what actually overpowers you, so a
    10 kn mean gusting to 30 is not a 10 kn day for kite choice.

    A kite is "ideal" when the wind sits inside its range, "marginal" within a
    further kite_marginal_frac buffer, else "off". The buffer never extends
    past the skill ceiling - that bound is meant to be hard."""
    kites = []
    for k in CFG["quiver"]:
        lo, hi, estimated, capped = kite_wind_range(k, rider)
        kites.append({"name": k["name"], "size_m2": k["size_m2"],
                      "range": (round(lo, 1), round(hi, 1)),
                      "estimated": estimated, "capped": capped})

    if w is None:
        for k in kites:
            k["status"] = "off"
        return {"kites": kites, "best": None, "best_status": None,
                "over_ceiling": False, "conclusion": "No wind data"}

    spd, gst = w[0], w[1]
    ceiling = rider.get("skill_ceiling_kn")
    over_ceiling = ceiling is not None and gst > ceiling
    m = CFG["kite_marginal_frac"]
    for k in kites:
        lo, hi = k["range"]
        hi_buf = hi * (1 + m)
        if ceiling is not None:
            hi_buf = min(hi_buf, float(ceiling))
        if lo <= spd and gst <= hi:
            k["status"] = "ideal"
        elif lo * (1 - m) <= spd and gst <= hi_buf:
            k["status"] = "marginal"
        else:
            k["status"] = "off"

    ideal = [k for k in kites if k["status"] == "ideal"]
    marginal = [k for k in kites if k["status"] == "marginal"]

    def center_dist(k):
        lo, hi = k["range"]
        return abs(spd - (lo + hi) / 2)

    best_status = None
    if ideal:
        best = min(ideal, key=center_dist)
        best_status = "ideal"
        conclusion = f"{best['size_m2']}m² {best['name']} — right in its sweet spot at {spd:.0f} kn"
    elif marginal:
        best = min(marginal, key=center_dist)
        best_status = "marginal"
        side = "underpowered" if spd < sum(best["range"]) / 2 else "overpowered"
        conclusion = (f"{best['size_m2']}m² {best['name']} is closest, but a little {side} "
                      f"at {spd:.0f} kn gusting {gst:.0f}")
    else:
        best = None
        biggest = max(kites, key=lambda k: k["size_m2"])
        if over_ceiling:
            conclusion = (f"Gusting {gst:.0f} kn, past your {ceiling:.0f} kn ceiling — "
                          "not a day for any of them")
        elif spd < biggest["range"][0]:
            conclusion = f"Too light for all three kites right now ({spd:.0f} kn)"
        else:
            conclusion = (f"Too strong for all three kites right now "
                          f"({spd:.0f} kn gusting {gst:.0f})")

    return {"kites": kites, "best": best["name"] if best else None,
            "best_status": best_status, "over_ceiling": over_ceiling,
            "conclusion": conclusion}


# ----------------------------------------------------------------------------
# Rideable windows over the week ahead
# ----------------------------------------------------------------------------
def largest_patch_ha(wet, cell_area_m2):
    """Area of the biggest connected wet patch.

    This, not the total, is what "enough water to ride" means: 250 ha spread
    over disconnected puddles is not a session, and one continuous sheet of
    4 ha is. Using the total made the test unfailable once the intertidal
    beach was included."""
    lab, n = ndimage.label(wet, structure=ndimage.generate_binary_structure(2, 1))
    if n == 0:
        return 0.0
    counts = ndimage.sum(wet, lab, index=np.arange(1, n + 1))
    return float(counts.max() * cell_area_m2 / 1e4)


def rideable_curve(z, F, coastal_zone, lagoon, sea, lagoon_wet, dry_override, grid, levels):
    """(total_ha, largest_patch_ha) at each sea level in `levels`.

    Precomputed so the week-long window search doesn't rerun the flood model at
    every candidate hour. Mirrors the frame loop step for step, so the sidebar
    and the "next window" line cannot disagree. Wet area is monotonic in sea
    level, so interpolating between samples is safe.

    Assumes the lagoon's own wet mask is fixed across the run - exactly true in
    lake mode (its area is one measured constant); an approximation in the tidal
    and wind modes, where it only affects the window search, not the frames."""
    tot, big = [], []
    for lv in levels:
        tidal, _ = flats_wet_map(z, F, float(lv), coastal_zone, lagoon)
        wet = remove_small_patches(tidal | lagoon_wet, grid)
        if dry_override is not None:
            wet &= ~dry_override
        wet &= ~sea
        tot.append(wet.sum() * grid.cell_area_m2 / 1e4)
        big.append(largest_patch_ha(wet, grid.cell_area_m2))
    return np.asarray(tot, float), np.asarray(big, float)


def rideable_at(te, water, wind, curve):
    """Everything one instant needs, shared by the frame loop and the window
    search. `curve` is (levels, total_ha, largest_ha) from rideable_curve."""
    levels, tot, big = curve
    sea_level, _ = water.level(te)
    total_ha = float(np.interp(sea_level, levels, tot))
    usable_ha = float(np.interp(sea_level, levels, big))
    w = wind_at(wind, te)
    quiver = quiver_advice(w, CFG["rider"])
    wind_txt, verdict, notes = kite_advice(w, usable_ha, quiver)
    return total_ha, usable_ha, w, quiver, verdict, notes, wind_txt, sea_level


def find_windows(now, water, wind, curve, daylight, max_windows=3, step_min=60):
    """Runs of consecutive daylight hours whose verdict is go or maybe.

    This is the question the tool is actually for: most hours are "no", so an
    8-hour slider showing thirteen refusals answers nothing. Searches the whole
    wind forecast horizon.

    Caveat worth remembering when reading the output: with the lagoon in lake
    mode its area is a constant, so what varies across the week is wind, tide
    (via the intertidal strip) and daylight - not the lagoon."""
    if wind is None:
        return []
    t_end = min(float(wind[0][-1]), water.covers_until())
    windows, cur = [], None
    te = now.timestamp()
    while te <= t_end:
        ok = is_daylight(daylight, te)
        if ok:
            _, _, w, quiver, verdict, _, _, _ = rideable_at(te, water, wind, curve)
            ok = verdict in ("go", "maybe")
        if ok:
            if cur is None:
                cur = {"start": te, "end": te, "verdict": verdict,
                       "kite": quiver["best"], "size": next(
                           (k["size_m2"] for k in quiver["kites"]
                            if k["name"] == quiver["best"]), None),
                       "spd": [w[0]], "gst": [w[1]], "deg": [w[2]]}
            else:
                cur["end"] = te
                cur["spd"].append(w[0])
                cur["gst"].append(w[1])
                cur["deg"].append(w[2])
                if verdict == "go":
                    cur["verdict"] = "go"
        elif cur is not None:
            windows.append(cur)
            cur = None
            if len(windows) >= max_windows:
                break
        te += step_min * 60
    if cur is not None and len(windows) < max_windows:
        windows.append(cur)
    for win in windows:
        win["mean_kn"] = float(np.mean(win["spd"]))
        win["gust_kn"] = float(np.max(win["gst"]))
        rad = np.radians(win["deg"])
        win["dir_deg"] = math.degrees(math.atan2(float(np.mean(np.sin(rad))),
                                                 float(np.mean(np.cos(rad))))) % 360
        for k in ("spd", "gst", "deg"):
            win.pop(k)
    return windows


def format_window(win, step_min=60):
    """One human line for a rideable window, in local time.

    The end is extended by one step: an hour that qualifies is rideable for
    that hour, so a single qualifying hour reads "15:00-16:00" rather than the
    zero-length "15:00-15:00"."""
    a = dt.datetime.fromtimestamp(win["start"], UTC).astimezone(TZ)
    b = dt.datetime.fromtimestamp(win["end"] + step_min * 60, UTC).astimezone(TZ)
    kite = f"{win['size']:g} m {win['kite']}" if win["kite"] else "no kite fits"
    hours = (win["end"] - win["start"]) / 3600 + step_min / 60
    same = a.date() == b.date()
    when = (f"{a:%a %d %b %H:%M}–{b:%H:%M}" if same
            else f"{a:%a %d %b %H:%M}–{b:%a %d %b %H:%M}")
    return (f"{when} ({hours:.0f} h) · {kite} · {win['mean_kn']:.0f} kn gusting "
            f"{win['gust_kn']:.0f}, {compass(win['dir_deg'])}")


# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Zandmotor lagoon</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Semi+Condensed:wght@600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<style>
:root{--ink:#10293a;--muted:#5b7182;--line:#d5dee4;--paper:#fbfcfd;--go:#1f8a5b;--maybe:#d98b1c;--no:#b8423a;--buoy:#f07f1e}
*{box-sizing:border-box}html,body{margin:0;height:100%;font-family:Barlow,system-ui,sans-serif;color:var(--ink);background:var(--paper)}
.app{display:grid;grid-template-columns:minmax(300px,380px) 1fr;height:100%}
#map{height:100%}
aside{overflow-y:auto;padding:20px 22px;border-right:1px solid var(--line)}
h1{font-family:"Barlow Semi Condensed",Barlow,sans-serif;font-size:26px;margin:0 0 2px;letter-spacing:.2px}
.sub{color:var(--muted);font-size:14px;margin:0 0 18px}
.time{font-family:"Barlow Semi Condensed",Barlow,sans-serif;font-size:48px;font-weight:700;line-height:1}
.day{color:var(--muted);font-size:15px;margin-top:2px}
input[type=range]{width:100%;margin:16px 0 4px;accent-color:var(--buoy)}
.ticks{display:flex;justify-content:space-between;color:var(--muted);font-size:12px}
.verdict{margin:18px 0 6px;padding:12px 14px;border-radius:10px;color:#fff;font-weight:600;font-size:18px}
.verdict.go{background:var(--go)}.verdict.maybe{background:var(--maybe)}.verdict.no{background:var(--no)}
.notes{font-size:14px;color:var(--muted);min-height:18px}
dl{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;margin:16px 0;font-size:15px}
dt{color:var(--muted)}dd{margin:0;font-weight:500;text-align:right}
svg.chart{width:100%;height:120px;display:block;margin-top:6px}
.legend{margin-top:16px;font-size:14px}
.legend div{display:flex;align-items:center;gap:8px;margin:4px 0}
.north{background:rgba(255,255,255,.9);border-radius:6px;padding:4px 6px 1px;box-shadow:0 1px 4px rgba(0,0,0,.25)}
.sw{width:18px;height:12px;border-radius:3px;border:1px solid rgba(0,0,0,.12)}
.fine{font-size:12px;color:var(--muted);margin-top:18px;line-height:1.45}
.warn{background:#fff4e5;border:1px solid #f3d3a8;border-radius:8px;padding:8px 10px;font-size:13px;margin-bottom:14px}
.quiver{margin-top:24px;padding-top:18px;border-top:1px solid var(--line)}
.quiver h2{font-family:"Barlow Semi Condensed",Barlow,sans-serif;font-size:19px;margin:0 0 12px}
.q-item{display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid var(--line)}
.q-item:last-of-type{border-bottom:none}
.q-item img{width:56px;height:56px;object-fit:cover;border-radius:8px;background:var(--line);flex-shrink:0}
.q-info{flex:1;min-width:0}
.q-name{font-weight:600;font-size:15px}
.q-range{color:var(--muted);font-size:13px}
.q-badge{font-size:12px;font-weight:600;padding:4px 9px;border-radius:999px;color:#fff;white-space:nowrap}
.q-badge.ideal{background:var(--go)}.q-badge.marginal{background:var(--maybe)}.q-badge.off{background:var(--muted)}
.q-conclusion{margin-top:12px;padding:10px 14px;border-radius:10px;background:var(--paper);border:1px solid var(--line);font-weight:500;font-size:14px}
.q-est{font-size:10px;font-weight:700;color:var(--muted);border:1px solid var(--line);border-radius:3px;padding:0 3px;margin-left:4px}
.windows{margin:10px 0 4px;font-size:14px}
.windows h3{font-family:"Barlow Semi Condensed",Barlow,sans-serif;font-size:15px;margin:0 0 6px;color:var(--muted);font-weight:600;letter-spacing:.3px;text-transform:uppercase}
.win{padding:7px 10px;border-radius:8px;border:1px solid var(--line);margin-bottom:5px;line-height:1.35}
.win.go{border-left:4px solid var(--go)}.win.maybe{border-left:4px solid var(--maybe)}
.win-when{font-weight:600}
.win-detail{color:var(--muted);font-size:13px}
.win-none{color:var(--muted);font-style:italic}
table.tides{width:100%;border-collapse:collapse;font-size:13px;margin:6px 0 2px}
table.tides td{padding:2px 0;border-bottom:1px solid var(--line)}
table.tides td:last-child{text-align:right;font-weight:500}
table.tides td.kind{color:var(--muted);width:42%}
@media (max-width:760px){html,body{height:auto}.app{display:flex;flex-direction:column;height:auto}#map{order:-1;height:55vh;position:sticky;top:0;z-index:1}aside{overflow:visible;border-right:0;border-top:1px solid var(--line);background:var(--paper);position:relative;z-index:2}}
</style></head><body>
<div class="app">
<aside>
  <h1>Zandmotor lagoon</h1>
  <p class="sub">Generated __GENERATED__ from __LOCATION__ water levels</p>
  __WARNINGS__
  <div class="time" id="time"></div><div class="day" id="day"></div>
  <input type="range" id="slider" min="0" max="__MAXI__" value="0" aria-label="Time">
  <div class="ticks"><span id="t0"></span><span id="t1"></span></div>
  <div class="verdict" id="verdict"></div><div class="notes" id="notes"></div>
  <div class="windows" id="windows"></div>
  <dl>
    <dt>Sea level</dt><dd id="sea"></dd>
    <dt>Lagoon water</dt><dd id="lag"></dd>
    <dt>Beach shallows</dt><dd id="shal"></dd>
    <dt>Rideable water</dt><dd id="ride"></dd>
    <dt>Biggest patch</dt><dd id="patch"></dd>
    <dt>Wind</dt><dd id="wind"></dd>
    <dt>Water temp</dt><dd id="wtemp"></dd>
    <dt>Wetsuit</dt><dd id="suit"></dd>
  </dl>
  <div class="notes" id="suitnote"></div>
  <div class="sub" style="margin:0">Sea level (tide)</div>
  <svg class="chart" id="tidechart" viewBox="0 0 300 60" preserveAspectRatio="none"></svg>
  <table class="tides" id="tides"></table>
  <div class="sub" style="margin:8px 0 0">Rideable water: lagoon + intertidal beach (bars)</div>
  <svg class="chart" id="chart" viewBox="0 0 300 120" preserveAspectRatio="none"></svg>
  <div class="sub" id="lagoonObsLabel" style="margin:8px 0 0"></div>
  <svg class="chart" id="obschart" viewBox="0 0 300 90" preserveAspectRatio="none"></svg>
  <div class="legend">
    <div><span class="sw" style="background:rgb(56,126,168)"></span>Rideable water: lagoon + intertidal beach</div>
    <div><span class="sw" style="background:rgba(56,126,168,.28)"></span>Open sea and isolated dune ponds — drawn, not counted</div>
    <div><span class="sw" style="background:none;border:2px dashed var(--buoy)"></span>Lagoon outline used for the numbers</div>
  </div>
  <div class="quiver">
    <h2>My kite quiver</h2>
    <div id="quiverList"></div>
    <div class="q-conclusion" id="quiverConclusion"></div>
    <p class="fine" style="margin-top:8px">Ranges for <span id="riderInfo"></span>: manufacturer 75&nbsp;kg charts
    shifted by weight (width kept), capped at your <span id="ceilingInfo"></span>&nbsp;kn ceiling. Upper limits are
    tested against the gust, not the mean. <b>e</b> = estimated, no chart found. A starting point, not gospel.</p>
  </div>
  <p class="fine">Water level source per hour: <span id="src"></span>. Terrain: __TERRAIN__.
  Areas cover the lagoon and the intertidal beach only &mdash; the open sea is drawn but never counted.
  Model estimate, not a safety tool. Check conditions and local kite zones on the spot.</p>
</aside>
<div id="map"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script>
const D = __DATA__;
let overlay = null;
try {
  const map = L.map('map', {zoomControl:true});
  const aerial = L.tileLayer('https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg',
    {maxZoom:19, attribution:'Luchtfoto &copy; PDOK'});
  const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    {maxZoom:19, attribution:'&copy; OpenStreetMap'});
  map.createPane('terrainPane').style.zIndex = 350;
  const terrain = L.imageOverlay(D.terrain, D.bounds, {pane:'terrainPane'});
  const bases = {'Aerial photo':aerial, 'Map':osm, 'Terrain model (sea, beach, dunes)':terrain};
  const pick = new URLSearchParams(location.search).get('base');
  (pick === 'terrain' ? terrain : pick === 'map' ? osm : aerial).addTo(map);
  L.control.layers(bases, null, {position:'topright'}).addTo(map);
  // lagoon plus the sea and dunes around it (whole modelled area on the terrain view)
  map.fitBounds(pick === 'terrain' ? D.bounds : L.latLngBounds(D.lagoon_bounds).pad(1.2));
  const North = L.Control.extend({options:{position:'topright'}, onAdd(){
    const d = L.DomUtil.create('div','north'); d.title = 'North is up';
    d.innerHTML = '<svg viewBox="0 0 24 38" width="24" height="38" aria-label="North"><path d="M12 2 L20 25 L12 19 L4 25 Z" fill="#10293a" stroke="#fff" stroke-width="1.5"/><text x="12" y="36" text-anchor="middle" font-family="Barlow,sans-serif" font-size="11" font-weight="700" fill="#10293a">N</text></svg>';
    return d; }});
  new North().addTo(map);
  L.control.scale({imperial:false, position:'bottomleft'}).addTo(map);

  overlay = L.imageOverlay(D.frames[0].png, D.bounds, {opacity:0.85}).addTo(map);
  L.polygon(D.lagoon, {color:'#f07f1e', weight:2, dashArray:'6 5', fill:false}).addTo(map);
} catch (e) {
  document.getElementById('map').innerHTML = '<p style="padding:20px">The map could not load (no internet?). The numbers on the left still work.</p>';
}
const $ = id => document.getElementById(id);
const words = {go:'Looks rideable', maybe:'Rideable, with caveats', no:'Not rideable'};
const qWords = {ideal:'Ideal', marginal:'Marginal', off:'Not this one'};
function initQuiver(){
  $('riderInfo').textContent = `${D.rider.weight_kg}kg, ${D.rider.height_cm}cm on a ${D.rider.board}`;
  $('ceilingInfo').textContent = D.rider.skill_ceiling_kn;
  $('quiverList').innerHTML = D.quiver_meta.map((k,i)=>`
    <div class="q-item">
      <img src="${k.image}" alt="${k.brand} ${k.name} ${k.size_m2}m²">
      <div class="q-info">
        <div class="q-name">${k.size_m2}m² ${k.brand} ${k.name}</div>
        <div class="q-range">${k.color}, ${k.year} &middot; <span id="q-range-${i}"></span> kn<span id="q-flag-${i}"></span></div>
      </div>
      <span class="q-badge" id="q-badge-${i}"></span>
    </div>`).join('');
}
function drawQuiver(f){
  f.quiver.kites.forEach((k,i)=>{
    const b = $('q-badge-'+i);
    b.className = 'q-badge '+k.status; b.textContent = qWords[k.status];
    $('q-range-'+i).textContent = k.range[0]+'–'+k.range[1];
    const flag = $('q-flag-'+i); flag.innerHTML = '';
    if (k.estimated) flag.innerHTML += '<span class="q-est" title="No manufacturer chart found; range estimated from the size formula">e</span>';
    if (k.capped) flag.innerHTML += `<span class="q-est" title="Top of the range capped at your ${D.rider.skill_ceiling_kn} kn ceiling">cap</span>`;
  });
  $('quiverConclusion').textContent = f.quiver.conclusion;
}
// Rideable windows across the whole wind forecast, not just the slider range.
// Most hours are "no", so this is usually the only part worth reading.
function initWindows(){
  const el = $('windows');
  if (!D.windows.length){
    el.innerHTML = '<h3>Next rideable window</h3><div class="win-none">'+D.windows_none+'</div>';
    return;
  }
  el.innerHTML = '<h3>Next rideable window'+(D.windows.length>1?'s':'')+'</h3>' +
    D.windows.map(w=>`<div class="win ${w.verdict}"><div class="win-when">${w.when}</div>
      <div class="win-detail">${w.detail}</div></div>`).join('') +
    (D.windows_caveat ? `<div class="win-detail">${D.windows_caveat}</div>` : '');
}
function initTides(){
  if (!D.tides.length){ $('tides').innerHTML = ''; return; }
  $('tides').innerHTML = D.tides.map(t=>
    `<tr><td class="kind">${t.kind}</td><td>${t.when}</td><td>${t.level}</td></tr>`).join('');
}
function draw(i){
  const f = D.frames[i];
  if (overlay) overlay.setUrl(f.png);
  $('time').textContent = f.time; $('day').textContent = f.day + (f.dark ? ' · dark' : '');
  $('sea').textContent = f.sea.toFixed(2)+' m NAP';
  // lake mode reports one measured value with its spread for the whole run;
  // the tidal/wind modes vary per hour, so fall back to this frame's value
  $('lag').textContent = D.lagoon_label || (f.lagoon_ha.toFixed(1)+' ha');
  $('shal').textContent = f.shallows_ha.toFixed(1)+' ha';
  $('ride').textContent = f.rideable_ha.toFixed(1)+' ha';
  // the largest connected patch is what the verdict actually tests against
  $('patch').textContent = f.usable_ha.toFixed(1)+' ha';
  $('wind').textContent = f.wind;
  $('wtemp').textContent = f.water_c == null ? '—' : f.water_c.toFixed(1)+' °C';
  $('suit').textContent = f.wetsuit || '—';
  $('suitnote').textContent = f.wetsuit_note || '';
  const v = $('verdict'); v.className = 'verdict '+f.verdict; v.textContent = words[f.verdict];
  $('notes').textContent = f.notes.length ? f.notes.join(', ') : '';
  $('src').textContent = f.src;
  drawQuiver(f);
  tidechart(i);
  chart(i);
}
function tidechart(sel){
  const n = D.frames.length, W=300, H=60, pad=6;
  const sv = D.frames.map(f=>f.sea);
  const smin = Math.min(...sv,0)-0.1, smax = Math.max(...sv,0)+0.1;
  const x = i => pad + i*(W-2*pad)/Math.max(1,n-1);
  const y = v => H-pad-(H-2*pad)*(v-smin)/(smax-smin);
  const pts = sv.map((v,i)=>`${x(i)},${y(v)}`).join(' ');
  let s = `<line x1="${pad}" y1="${y(0)}" x2="${W-pad}" y2="${y(0)}" stroke="#d5dee4" stroke-width="1" stroke-dasharray="3 2" vector-effect="non-scaling-stroke"/>`;
  s += `<polyline points="${pts}" fill="none" stroke="#2f7fb8" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
  s += `<line x1="${x(sel)}" y1="${pad}" x2="${x(sel)}" y2="${H-pad}" stroke="#f07f1e" stroke-width="1" stroke-dasharray="2 2" opacity="0.6" vector-effect="non-scaling-stroke"/>`;
  s += `<circle cx="${x(sel)}" cy="${y(sv[sel])}" r="4" fill="#f07f1e" stroke="#fff" stroke-width="1.5"/>`;
  $('tidechart').innerHTML = s;
}
function chart(sel){
  const n = D.frames.length, W=300, H=120, pad=6;
  const rd = D.frames.map(f=>f.rideable_ha), rmax = Math.max(1,...rd);
  const x = i => pad + i*(W-2*pad)/Math.max(1,n-1);
  const bw = Math.max(2,(W-2*pad)/n*0.6);
  let s = '';
  rd.forEach((r,i)=>{
    const h=(H-2*pad)*r/rmax;
    const fill = i==sel ? '#f07f1e' : (D.frames[i].dark ? '#e3eaee' : '#b9d8e4');
    s+=`<rect x="${x(i)-bw/2}" y="${H-pad-h}" width="${bw}" height="${h}" fill="${fill}" rx="2"/>`;
  });
  // the threshold that decides "enough water to ride"
  const ty = H-pad-(H-2*pad)*D.min_rideable_ha/rmax;
  if (D.min_rideable_ha < rmax) s += `<line x1="${pad}" y1="${ty}" x2="${W-pad}" y2="${ty}" stroke="#b8423a" stroke-width="1" stroke-dasharray="4 3" vector-effect="non-scaling-stroke"/>`;
  $('chart').innerHTML = s;
}
// The lagoon's area is a MEASUREMENT, not a forecast: in lake mode nothing in
// the record predicts it, so this plots the satellite observations it is drawn
// from plus their spread, rather than a flat line pretending to be a forecast.
function obschart(){
  const o = D.lagoon_obs;
  if (!o || !o.times.length){ $('obschart').innerHTML = ''; $('lagoonObsLabel').textContent=''; return; }
  $('lagoonObsLabel').textContent = o.label;
  const W=300, H=90, pad=8;
  const t0 = Math.min(...o.times), t1 = Math.max(...o.times);
  const amax = Math.max(...o.areas, o.hi)*1.1;
  const x = t => pad + (t-t0)*(W-2*pad)/Math.max(1,t1-t0);
  const y = a => H-pad-(H-2*pad)*a/amax;
  let s = `<rect x="${pad}" y="${y(o.hi)}" width="${W-2*pad}" height="${Math.max(1,y(o.lo)-y(o.hi))}" fill="#b9d8e4" opacity="0.5"/>`;
  s += `<line x1="${pad}" y1="${y(o.median)}" x2="${W-pad}" y2="${y(o.median)}" stroke="#10293a" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`;
  o.times.forEach((t,i)=>{
    const used = o.used[i];
    s += `<circle cx="${x(t)}" cy="${y(o.areas[i])}" r="${used?2.6:2}" fill="${used?'#2f7fb8':'none'}" stroke="${used?'#fff':'#9db3c0'}" stroke-width="1"/>`;
  });
  $('obschart').innerHTML = s;
}
$('slider').addEventListener('input', e => draw(+e.target.value));
$('t0').textContent = D.frames[0].time; $('t1').textContent = D.frames[D.frames.length-1].time;
initQuiver();
initWindows();
initTides();
obschart();
draw(0);
</script></body></html>"""


def write_html(path, frames, grid, lagoon_ring, location, terrain_note, warnings, terrain_uri,
               windows, tides, lagoon_obs, lagoon_label):
    lats = [p[1] for p in lagoon_ring]
    lons = [p[0] for p in lagoon_ring]
    quiver_meta = [{"name": k["name"], "size_m2": k["size_m2"], "brand": k["brand"],
                    "color": k["color"], "year": k["year"],
                    "image": quiver_image_uri(k["image"], warnings)} for k in CFG["quiver"]]
    data = {"bounds": grid.latlon_bounds, "frames": frames, "terrain": terrain_uri,
            "lagoon": [[lat, lon] for lon, lat in lagoon_ring],
            "lagoon_bounds": [[min(lats), min(lons)], [max(lats), max(lons)]],
            "quiver_meta": quiver_meta, "rider": CFG["rider"],
            "min_rideable_ha": CFG["min_rideable_ha"],
            "windows": windows["list"], "windows_none": windows["none"],
            "windows_caveat": windows["caveat"],
            "tides": tides, "lagoon_obs": lagoon_obs, "lagoon_label": lagoon_label}
    warn_html = "".join(f'<div class="warn">{w}</div>' for w in warnings)
    html = (HTML_TEMPLATE
            .replace("__DATA__", json.dumps(data))
            .replace("__MAXI__", str(len(frames) - 1))
            .replace("__GENERATED__", dt.datetime.now(TZ).strftime("%a %d %b %H:%M"))
            .replace("__LOCATION__", location)
            .replace("__TERRAIN__", terrain_note)
            .replace("__WARNINGS__", warn_html))
    path.write_text(html, encoding="utf-8")


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
