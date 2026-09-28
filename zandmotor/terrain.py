"""AHN elevation fetching/caching and terrain-derived masks."""

from __future__ import annotations

import sys
import time

import numpy as np
import requests
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from skimage.morphology import reconstruction

from zandmotor.config import CACHE, CFG, log


def _is_tiff(content: bytes) -> bool:
    return content[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def prune_cache():
    """Drop the per-day dry-override masks once they're stale. One is written
    per run-day at ~640 KB and nothing ever removed them."""
    cutoff = time.time() - CFG["flats_dry_cache_keep_days"] * 86400
    for f in CACHE.glob("flats_dry_*.npy"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            log.debug("Pruned stale cache file %s", f.name)


def dem_age_days():
    """Age of the cached AHN tile in days, or None if it isn't cached.
    Cached indefinitely, on a sand engine built to reshape - so the age is
    worth surfacing rather than assuming the terrain is current."""
    f = CACHE / "ahn_dtm.tif"
    if not f.exists():
        return None
    return (time.time() - f.stat().st_mtime) / 86400


def fetch_ahn(grid) -> np.ndarray:
    """Download AHN DTM for the grid area (cached) and put it on the grid."""
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / "ahn_dtm.tif"
    if not cache_file.exists():
        to_rd = Transformer.from_crs(3857, 28992, always_xy=True)
        x0, y0, x1, y1 = grid.bounds_3857
        xs, ys = zip(*[to_rd.transform(x, y) for x, y in
                       [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]])
        minx, miny, maxx, maxy = min(xs) - 50, min(ys) - 50, max(xs) + 50, max(ys) + 50
        r = CFG["ahn_request_res_m"]
        width, height = int((maxx - minx) / r), int((maxy - miny) / r)
        content = None
        for fmt in ("GEOTIFF_FLOAT32", "GEOTIFF", "image/tiff"):
            params = {"SERVICE": "WCS", "VERSION": "1.0.0", "REQUEST": "GetCoverage",
                      "COVERAGE": CFG["ahn_coverage"], "CRS": "EPSG:28992",
                      "RESPONSE_CRS": "EPSG:28992",
                      "BBOX": f"{minx},{miny},{maxx},{maxy}",
                      "WIDTH": width, "HEIGHT": height, "FORMAT": fmt}
            log.info("Downloading AHN terrain (format %s)...", fmt)
            try:
                resp = requests.get(CFG["ahn_wcs"], params=params, timeout=180)
                if resp.ok and _is_tiff(resp.content):
                    content = resp.content
                    break
                log.debug("AHN response not a GeoTIFF: %s", resp.text[:300])
            except requests.RequestException as e:
                log.debug("AHN request failed: %s", e)
        if content is None:
            sys.exit("Could not download AHN terrain from PDOK. Check the coverage name "
                     f"'{CFG['ahn_coverage']}' via {CFG['ahn_wcs']}?request=GetCapabilities"
                     "&service=WCS, or set CFG['local_dem'] to a GeoTIFF you downloaded.")
        cache_file.write_bytes(content)
    # max, not bilinear: keep a low sand ridge's full height rather than
    # averaging it away, so the flood model doesn't leak through it
    return read_to_grid(cache_file, grid, resampling=Resampling.max)


def read_to_grid(path, grid, resampling=Resampling.bilinear) -> np.ndarray:
    with open(path, "rb") as f, MemoryFile(f.read()) as mem, mem.open() as src:
        data = src.read(1).astype("float32")
        nod = src.nodata
        if nod is not None:
            data[data == nod] = np.nan
        data[(data > 1000) | (data < -100)] = np.nan
        out = np.full((grid.height, grid.width), np.nan, dtype="float32")
        reproject(data, out, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=np.nan, dst_transform=grid.transform, dst_crs="EPSG:3857",
                  dst_nodata=np.nan, resampling=resampling)
    return out


def prepare_terrain(z_raw, lagoon):
    """Fill terrain gaps for rendering. Missing data (offshore, or under the
    lagoon - AHN's laser can't see through water) gets a nominal
    "underwater" fill purely so the terrain layer has no holes; it's
    cosmetic only. The lagoon's bed elevation isn't modelled at all: only
    whether a spot is actually wet matters for kiting, not how deep, so
    wetness comes straight from satellite observations (see
    wetness_frequency) instead of inferred bathymetry."""
    z = z_raw.copy()
    nod = ~np.isfinite(z)
    lab, _ = ndimage.label(nod)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge = edge[edge > 0]
    sea = np.isin(lab, edge) & ~lagoon
    z[sea] = CFG["sea_fill_level"]
    z[~np.isfinite(z) & lagoon] = CFG["sea_fill_level"]
    z[~np.isfinite(z)] = 20.0              # other gaps: treat as land
    return z


def spill_elevation(z):
    """Lowest water level at which each cell connects to the open sea
    (priority-flood via morphological reconstruction). Uses a 4-connected
    (straight, not diagonal) footprint, so water can't hop through a gap a
    real flow couldn't fit through."""
    seed = np.full_like(z, z.max())
    border = np.zeros(z.shape, bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    sea = border & (z < CFG["sea_seed_level"])
    if not sea.any():
        sys.exit("No open sea found on the map edge; enlarge bbox_wgs84 seaward.")
    seed[sea] = z[sea]
    footprint = ndimage.generate_binary_structure(2, 1)
    return reconstruction(seed, z, method="erosion", footprint=footprint)


def open_sea_mask(z, lagoon, level):
    """The open sea: every cell below `level` that is connected to the map
    edge, minus the lagoon. Connectivity to the edge is what makes it the
    sea rather than some enclosed hollow that merely sits low."""
    border = np.zeros(z.shape, bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    below = z < level
    lab, _ = ndimage.label(below)
    sea_labels = set(np.unique(lab[border & below])) - {0}
    return np.isin(lab, list(sea_labels)) & ~lagoon


def coastal_zone_mask(z, lagoon):
    """Cells within coastal_zone_m of the open sea, plus the lagoon itself.
    Flooding is only ever rendered inside this zone. Elsewhere - a nearby
    town, say - low elevation doesn't mean tidal water reaches it; real
    dikes and pumped drainage aren't in the DEM, so an unrestricted
    flood-fill would otherwise "flood" any low ground anywhere in the grid.

    Note this zone CONTAINS the sea (sea cells are zero distance from
    themselves), so it is not on its own a mask of rideable ground - see
    permanent_sea_mask and shallows_mask."""
    sea = open_sea_mask(z, lagoon, CFG["sea_seed_level"])
    dist_px = ndimage.distance_transform_edt(~sea)
    return ((dist_px * CFG["grid_res_m"]) <= CFG["coastal_zone_m"]) | lagoon


def permanent_sea_mask(z, lagoon, min_level):
    """The sea that never dries out over the run: open sea below the lowest
    water level any frame will see. Everything wet outside this and outside
    the lagoon is therefore intertidal - beach that genuinely comes and goes
    with the tide.

    This exists because the open sea fills most of the map (~620 ha against a
    ~2.8 ha lagoon). Folded into one "total water" figure it swamped
    everything, and the "is there enough water to ride" test could never
    fail. The sea is excluded from every area figure and drawn separately;
    what is left wet outside it and outside the lagoon is the intertidal
    beach strip, which is counted as rideable."""
    return open_sea_mask(z, lagoon, min_level)
