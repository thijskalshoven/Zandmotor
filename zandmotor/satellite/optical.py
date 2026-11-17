"""Sentinel-2 optical scene listing and NDWI/SCL/RGB extraction."""

from __future__ import annotations

import datetime as dt

import numpy as np

from zandmotor.config import CFG
from zandmotor.satellite.client import catalog_search, dedupe_scenes_one_per_day, process_image
from zandmotor.time_utils import parse_iso_datetime

EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["B02", "B03", "B04", "B08", "SCL"]}],
          output: {bands: 5, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) {
  return [(s.B03 - s.B08) / (s.B03 + s.B08 + 1e-6), s.SCL, s.B04, s.B03, s.B02];
}"""

# Sentinel-2 scene classes that mean "the ground is visible": vegetation, bare
# soil, water, unclassified, snow. Everything else (cloud, shadow, cirrus,
# saturated, no data) hides the surface.
CLEAR_SCL = (4, 5, 6, 7, 11)


def clear_mask(scl):
    return np.isin(np.round(scl), CLEAR_SCL)


def cdse_scenes(token, now, days_back=None, max_scenes=None, max_cloud=None):
    """Sentinel-2 L2A scene datetimes, one per day, newest first.

    max_cloud is the tile-level eo:cloud_cover cut-off. That number describes a
    ~100 km tile, not the lagoon, so callers that check the lagoon pixel by
    pixel pass a loose value and decide per image (see sat_max_cloud)."""
    start = now - dt.timedelta(days=days_back or CFG["sat_days_back"])
    cql = {"op": "<", "args": [{"property": "eo:cloud_cover"},
                               CFG["sat_max_cloud"] if max_cloud is None else max_cloud]}
    feats = catalog_search(token, "sentinel-2-l2a", start, now, cql)
    times = {parse_iso_datetime(f["properties"]["datetime"]) for f in feats}
    return dedupe_scenes_one_per_day(times)[:max_scenes or CFG["sat_max_scenes"]]


def cdse_image(token, grid, t):
    """Returns ndwi, scl, rgb (H, W, 3 reflectance) on the grid."""
    a = process_image(token, grid, t, collection="sentinel-2-l2a", evalscript=EVALSCRIPT,
                      extra_data_filter={"mosaickingOrder": "leastCC"})
    return a[0], a[1], np.moveaxis(a[2:5], 0, -1)
