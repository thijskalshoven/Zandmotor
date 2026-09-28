"""Sentinel-2 optical scene listing and NDWI/SCL/RGB extraction."""

from __future__ import annotations

import datetime as dt

import numpy as np
import requests

from zandmotor.config import CFG
from zandmotor.satellite.client import dedupe_scenes_one_per_day, process_image
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
    times = {parse_iso_datetime(f["properties"]["datetime"]) for f in r.json()["features"]}
    return dedupe_scenes_one_per_day(times)[:max_scenes or CFG["sat_max_scenes"]]


def cdse_image(token, grid, t):
    """Returns ndwi, scl, rgb (H, W, 3 reflectance) on the grid."""
    a = process_image(token, grid, t, collection="sentinel-2-l2a", evalscript=EVALSCRIPT,
                      extra_data_filter={"mosaickingOrder": "leastCC"})
    return a[0], a[1], np.moveaxis(a[2:5], 0, -1)
