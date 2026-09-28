"""Lagoon outline tracing: from Sentinel-2 imagery or OpenStreetMap, and saving it."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import requests
from pyproj import Transformer
from scipy import ndimage

from zandmotor.config import CFG, HERE, TZ, log
from zandmotor.geometry import mask_to_ring, polygon_mask
from zandmotor.rendering import outline_preview, terrain_rgb
from zandmotor.satellite.client import cdse_token
from zandmotor.satellite.optical import cdse_image, cdse_scenes
from zandmotor.tides import WaterLevel


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
