"""CDSE (Copernicus Data Space Ecosystem) authentication and the request
helpers shared by the optical (Sentinel-2) and SAR (Sentinel-1) modules."""

from __future__ import annotations

import datetime as dt
import os

import requests
from rasterio.io import MemoryFile

from zandmotor.config import CFG


def cdse_token():
    cid, secret = os.environ.get("CDSE_CLIENT_ID"), os.environ.get("CDSE_CLIENT_SECRET")
    if not cid or not secret:
        raise RuntimeError("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET for --calibrate "
                           "(see top of file).")
    r = requests.post(CFG["cdse_token_url"], data={"grant_type": "client_credentials",
                                                  "client_id": cid, "client_secret": secret},
                      timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def dedupe_scenes_one_per_day(times):
    """From scene datetimes, keep only the newest one per calendar day,
    newest day first. Shared by cdse_scenes and cdse_sar_scenes."""
    seen, out = set(), []
    for t in sorted(times, reverse=True):
        if t.date() not in seen:
            seen.add(t.date())
            out.append(t)
    return out


def process_image(token, grid, t, *, collection, evalscript, extra_data_filter=None):
    """Sentinel Hub Process API request for one day's mosaic of `collection`
    on `grid`, decoded to a (bands, H, W) float32 array. Shared by cdse_image
    (Sentinel-2) and cdse_sar_image (Sentinel-1) - they differ only in which
    collection/evalscript/extra dataFilter fields they need."""
    day0 = t.replace(hour=0, minute=0, second=0, microsecond=0)
    data_filter = {"timeRange": {"from": day0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "to": (day0 + dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")}}
    if extra_data_filter:
        data_filter.update(extra_data_filter)
    body = {"input": {"bounds": {"bbox": list(grid.bounds_3857),
                                 "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"}},
                      "data": [{"type": collection, "dataFilter": data_filter}]},
            "output": {"width": grid.width, "height": grid.height,
                       "responses": [{"identifier": "default",
                                      "format": {"type": "image/tiff"}}]},
            "evalscript": evalscript}
    r = requests.post(CFG["cdse_process_url"], json=body, timeout=120,
                      headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    with MemoryFile(r.content) as mem, mem.open() as src:
        return src.read().astype("float32")
