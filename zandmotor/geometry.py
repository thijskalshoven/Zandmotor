"""Grid, lagoon polygon I/O, and mask/ring conversions.

Web Mercator grid so the map overlay lines up exactly in Leaflet.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine, from_bounds
from scipy import ndimage

from zandmotor.config import CFG, DATA


@dataclass
class Grid:
    transform: object
    width: int
    height: int
    bounds_3857: tuple
    latlon_bounds: list      # [[lat_min, lon_min], [lat_max, lon_max]]
    cell_area_m2: float


def make_grid(bbox, res_m) -> Grid:
    to_merc = Transformer.from_crs(4326, 3857, always_xy=True)
    to_wgs = Transformer.from_crs(3857, 4326, always_xy=True)
    x0, y0 = to_merc.transform(bbox[0], bbox[1])
    x1, y1 = to_merc.transform(bbox[2], bbox[3])
    lat_c = (bbox[1] + bbox[3]) / 2
    res = res_m / math.cos(math.radians(lat_c))     # mercator units per cell
    w, h = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    x1, y1 = x0 + w * res, y0 + h * res
    lon0, lat0 = to_wgs.transform(x0, y0)
    lon1, lat1 = to_wgs.transform(x1, y1)
    return Grid(from_bounds(x0, y0, x1, y1, w, h), w, h, (x0, y0, x1, y1),
                [[lat0, lon0], [lat1, lon1]], res_m ** 2)


def crop_grid(grid, mask, pad_px):
    """A sub-grid around `mask`, padded by pad_px cells, aligned cell for cell
    with `grid`, plus the (row, col) slices that cut it out of full-grid arrays.

    The calibration only needs the lagoon, a few percent of the map, so its
    satellite requests ask for this window instead of the whole bbox: the same
    pixels for a fraction of the processing units and download time. Because
    the window is on the full grid's own cell lattice, results paste straight
    back into full-size arrays."""
    rows, cols = np.nonzero(mask)
    r0, r1 = max(int(rows.min()) - pad_px, 0), min(int(rows.max()) + pad_px + 1, grid.height)
    c0, c1 = max(int(cols.min()) - pad_px, 0), min(int(cols.max()) + pad_px + 1, grid.width)
    tr = grid.transform * Affine.translation(c0, r0)
    w, h = c1 - c0, r1 - r0
    x0, y1 = tr.c, tr.f                       # top-left corner
    x1, y0 = x0 + w * tr.a, y1 + h * tr.e     # tr.e is negative
    to_wgs = Transformer.from_crs(3857, 4326, always_xy=True)
    lon0, lat0 = to_wgs.transform(x0, y0)
    lon1, lat1 = to_wgs.transform(x1, y1)
    sub = Grid(tr, w, h, (x0, y0, x1, y1), [[lat0, lon0], [lat1, lon1]], grid.cell_area_m2)
    return sub, (slice(r0, r1), slice(c0, c1))


def load_lagoon_polygon():
    """Returns (ring, properties) from lagoon.geojson, or (None, None)."""
    return _load_ring(DATA / CFG["lagoon_geojson_file"])


def load_basin_polygon():
    """Returns (ring, properties) from lagoon_basin.geojson, or (None, None)."""
    return _load_ring(DATA / CFG["lagoon_basin_file"])


def basin_mask(grid, lagoon, z_raw):
    """The lagoon basin: the largest extent the lagoon has been traced at,
    widened by basin_margin_m for tracing error, with dune (AHN above
    lagoon_trim_land_m, the same rule as trim_lagoon_to_water) cut off. Always
    contains the lagoon outline.

    The outline is one trace, and a trace from a quiet day is small (4.5 ha on
    2026-11-16 against 13.4 ha at the 2026-09-01 fill). The basin keeps the
    rest, so the probable water edge can show the sea flowing back into it.

    AHN no-data patches that touch the basin are added to it: the laser gets
    no return off water, so they were lagoon when surveyed. Two strips west
    and north-east of the 13.4 ha trace (2.3 ha) were left out otherwise, and
    on 2026-11-12 the water went past the basin without them. Patches joined
    to the map edge are the sea, not lagoon.
    Returns (mask, found) - without a basin file it is the outline plus the
    margin, and found is False."""
    ring, _ = load_basin_polygon()
    basin = lagoon.copy() if ring is None else polygon_mask(grid, ring) | lagoon
    px = int(round(CFG["basin_margin_m"] / CFG["grid_res_m"]))
    if px > 0:
        basin = ndimage.binary_dilation(basin, iterations=px)
    gaps, _ = ndimage.label(~np.isfinite(z_raw))
    edge = np.unique(np.concatenate([gaps[0], gaps[-1], gaps[:, 0], gaps[:, -1]]))
    touching = np.setdiff1d(np.unique(gaps[ndimage.binary_dilation(basin)]), edge)
    basin |= np.isin(gaps, touching[touching > 0])
    land = np.isfinite(z_raw) & (z_raw > CFG["lagoon_trim_land_m"])
    return (basin & ~land) | lagoon, ring is not None


def _load_ring(f):
    if not f.exists():
        return None, None
    gj = json.loads(f.read_text())
    feat = gj["features"][0] if "features" in gj else gj
    geom = feat.get("geometry", feat)
    ring = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
    return [tuple(p[:2]) for p in ring], feat.get("properties") or {}


def polygon_mask(grid, ring_lonlat):
    to_merc = Transformer.from_crs(4326, 3857, always_xy=True)
    ring = [to_merc.transform(lon, lat) for lon, lat in ring_lonlat]
    geom = {"type": "Polygon", "coordinates": [ring]}
    return rasterize([(geom, 1)], out_shape=(grid.height, grid.width),
                     transform=grid.transform, fill=0, dtype="uint8").astype(bool)


def trim_lagoon_to_water(ring, lagoon, z_raw, grid):
    """Cut off parts of the traced outline that AHN elevation shows are
    persistently dry land. OSM/satellite traces can overshoot onto the
    beach or dunes, especially here since Zandmotor's coastline is built to
    keep reshaping. Returns (ring, lagoon, trimmed_ha)."""
    land = np.isfinite(z_raw) & (z_raw > CFG["lagoon_trim_land_m"])
    trimmed = lagoon & ~land
    if not trimmed.any() or trimmed.sum() == lagoon.sum():
        return ring, lagoon, 0.0
    new_ring = mask_to_ring(trimmed, grid)
    new_lagoon = polygon_mask(grid, new_ring)
    cut_ha = (lagoon.sum() - new_lagoon.sum()) * grid.cell_area_m2 / 1e4
    return new_ring, new_lagoon, cut_ha


def _simplify(pts, tol):
    """Douglas-Peucker line simplification."""
    pts = np.asarray(pts)
    if len(pts) < 4:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = pts[a], pts[b]
        seg = q - p
        L = np.hypot(*seg) or 1e-9
        rel = pts[a + 1:b] - p
        d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / L
        i = int(np.argmax(d))
        if d[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return pts[keep]


def mask_to_ring(mask, grid):
    from rasterio.features import shapes
    best, best_area = None, 0
    for geom, val in shapes(mask.astype("uint8"), mask=mask, transform=grid.transform):
        ext = np.array(geom["coordinates"][0])
        area = 0.5 * abs(np.dot(ext[:-1, 0], ext[1:, 1]) - np.dot(ext[1:, 0], ext[:-1, 1]))
        if area > best_area:
            best, best_area = ext, area
    res_merc = grid.transform.a
    # closed ring: split at the point farthest from the start, simplify both halves
    far = int(np.argmax(np.hypot(*(best - best[0]).T)))
    tol = 1.2 * res_merc
    ext = np.vstack([_simplify(best[:far + 1], tol), _simplify(best[far:], tol)[1:]])
    to_wgs = Transformer.from_crs(3857, 4326, always_xy=True)
    ring = [tuple(round(v, 6) for v in to_wgs.transform(x, y)) for x, y in ext]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring
