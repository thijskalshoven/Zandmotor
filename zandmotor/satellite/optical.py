"""Sentinel-2 optical scene listing and NDWI/SCL/RGB extraction."""

from __future__ import annotations

import datetime as dt

import numpy as np
import requests
from rasterio.io import MemoryFile

from zandmotor.config import CFG
from zandmotor.time_utils import format_rws_datetime, parse_iso_datetime

EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["B02", "B03", "B04", "B08", "SCL"]}],
          output: {bands: 5, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) {
  return [(s.B03 - s.B08) / (s.B03 + s.B08 + 1e-6), s.SCL, s.B04, s.B03, s.B02];
}"""


def cdse_scenes(token, now, days_back=None, max_scenes=None):
    body = {"bbox": list(CFG["bbox_wgs84"]),
            "datetime": f"{format_rws_datetime(now - dt.timedelta(days=days_back or CFG['sat_days_back']))[:19]}Z/"
                        f"{format_rws_datetime(now)[:19]}Z",
            "collections": ["sentinel-2-l2a"], "limit": 100,
            "filter": {"op": "<", "args": [{"property": "eo:cloud_cover"},
                                           CFG["sat_max_cloud"]]},
            "filter-lang": "cql2-json"}
    r = requests.post(CFG["cdse_catalog_url"], json=body, timeout=60,
                      headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    times = sorted({parse_iso_datetime(f["properties"]["datetime"]) for f in r.json()["features"]},
                   reverse=True)
    # one per day, newest first
    seen, out = set(), []
    for t in times:
        if t.date() not in seen:
            seen.add(t.date())
            out.append(t)
    return out[:max_scenes or CFG["sat_max_scenes"]]


def cdse_image(token, grid, t):
    """Returns ndwi, scl, rgb (H, W, 3 reflectance) on the grid."""
    day0 = t.replace(hour=0, minute=0, second=0, microsecond=0)
    body = {"input": {"bounds": {"bbox": list(grid.bounds_3857),
                                 "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"}},
                      "data": [{"type": "sentinel-2-l2a",
                                "dataFilter": {"timeRange": {
                                    "from": day0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    "to": (day0 + dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")},
                                    "mosaickingOrder": "leastCC"}}]},
            "output": {"width": grid.width, "height": grid.height,
                       "responses": [{"identifier": "default",
                                      "format": {"type": "image/tiff"}}]},
            "evalscript": EVALSCRIPT}
    r = requests.post(CFG["cdse_process_url"], json=body, timeout=120,
                      headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    with MemoryFile(r.content) as mem, mem.open() as src:
        a = src.read().astype("float32")
    return a[0], a[1], np.moveaxis(a[2:5], 0, -1)
