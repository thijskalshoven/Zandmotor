"""Landsat 8-9 (OLI) scene listing and NDWI/cloud extraction: a second optical
sensor for the lagoon record, alongside Sentinel-2.

Landsat 8 and 9 together pass every 8 days at ~10:35 UTC, on dates
Sentinel-2 often misses (in the 2025-26 record, 5 of the 9 Landsat scenes
that saw the lagoon had no Sentinel-2 image within 3 days). At 30 m it is
coarser, so its NDWI threshold is fitted against Sentinel-2 images of the
same days (lagoon_model.calibration), the way radar tracks were.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from zandmotor.config import CFG
from zandmotor.satellite.client import catalog_search, dedupe_scenes_one_per_day, process_image
from zandmotor.time_utils import parse_iso_datetime

EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["B03", "B05", "BQA"], units: ["REFLECTANCE", "REFLECTANCE", "DN"]}],
          output: {bands: 2, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) {
  return [(s.B03 - s.B05) / (s.B03 + s.B05 + 1e-6), s.BQA];
}"""

# QA_PIXEL bits that hide the ground: fill, dilated cloud, cirrus, cloud.
# Cloud SHADOW (bit 4) is left out on purpose: the cloud mask flags the dark
# lagoon itself as shadow (2026-09-01: two thirds of a clear, full lagoon).
HIDDEN_BITS = 0b1111


def landsat_clear(qa):
    """Pixels where the ground is visible, from the QA band."""
    return (np.asarray(qa, np.int64) & HIDDEN_BITS) == 0


def landsat_scenes(token, now, days_back=None, max_cloud=None):
    """Landsat 8-9 scene datetimes, one per day, newest first."""
    start = now - dt.timedelta(days=days_back or CFG["response_days_back"])
    cql = {"op": "<", "args": [{"property": "eo:cloud_cover"},
                               CFG["sat_max_cloud"] if max_cloud is None else max_cloud]}
    feats = catalog_search(token, CFG["landsat_collection"], start, now, cql)
    return dedupe_scenes_one_per_day({parse_iso_datetime(f["properties"]["datetime"])
                                      for f in feats})


def landsat_image(token, grid, t):
    """Returns (ndwi, clear) on the grid for the single pass at t."""
    a = process_image(token, grid, t, collection=CFG["landsat_collection"],
                      evalscript=EVALSCRIPT,
                      time_window=dt.timedelta(minutes=CFG["landsat_scene_window_min"]))
    return a[0], landsat_clear(a[1])
