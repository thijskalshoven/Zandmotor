"""Sentinel-1 SAR scene listing, VV/VH extraction, and wet/dry classification."""

from __future__ import annotations

import datetime as dt

import numpy as np
import requests
from rasterio.io import MemoryFile
from scipy import ndimage

from zandmotor.config import CFG
from zandmotor.time_utils import format_rws_datetime, parse_iso_datetime

SAR_EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["VV", "VH"]}],
          output: {bands: 2, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) {
  return [10*Math.log10(s.VV), 10*Math.log10(s.VH)];
}"""


def cdse_sar_scenes(token, now, days_back=None, max_scenes=None):
    """Sentinel-1 GRD scenes, one per day, fixed orbit direction for
    consistent backscatter geometry (ascending/descending look very
    different over the same ground)."""
    body = {"bbox": list(CFG["bbox_wgs84"]),
            "datetime": f"{format_rws_datetime(now - dt.timedelta(days=days_back or CFG['sar_days_back']))[:19]}Z/"
                        f"{format_rws_datetime(now)[:19]}Z",
            "collections": ["sentinel-1-grd"], "limit": 100,
            "filter": {"op": "=", "args": [{"property": "sat:orbit_state"}, CFG["sar_orbit"]]},
            "filter-lang": "cql2-json"}
    times = []
    cursor_end = now
    span = dt.timedelta(days=days_back or CFG["sar_days_back"])
    start_bound = now - span
    while cursor_end > start_bound:
        chunk_start = max(cursor_end - dt.timedelta(days=60), start_bound)
        body["datetime"] = f"{format_rws_datetime(chunk_start)[:19]}Z/{format_rws_datetime(cursor_end)[:19]}Z"
        r = requests.post(CFG["cdse_catalog_url"], json=body, timeout=60,
                          headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        times += [parse_iso_datetime(f["properties"]["datetime"]) for f in r.json()["features"]]
        cursor_end = chunk_start
    seen, out = set(), []
    for t in sorted(times, reverse=True):
        if t.date() not in seen:
            seen.add(t.date())
            out.append(t)
    return out[:max_scenes or CFG["sar_max_scenes"]]


def cdse_sar_image(token, grid, t):
    """Returns vv, vh (dB backscatter) on the grid."""
    day0 = t.replace(hour=0, minute=0, second=0, microsecond=0)
    body = {"input": {"bounds": {"bbox": list(grid.bounds_3857),
                                 "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"}},
                      "data": [{"type": "sentinel-1-grd",
                                "dataFilter": {"timeRange": {
                                    "from": day0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    "to": (day0 + dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")},
                                    "acquisitionMode": "IW", "polarization": "DV",
                                    "orbitDirection": CFG["sar_orbit"].upper()}}]},
            "output": {"width": grid.width, "height": grid.height,
                       "responses": [{"identifier": "default",
                                      "format": {"type": "image/tiff"}}]},
            "evalscript": SAR_EVALSCRIPT}
    r = requests.post(CFG["cdse_process_url"], json=body, timeout=120,
                      headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    with MemoryFile(r.content) as mem, mem.open() as src:
        a = src.read().astype("float32")
    return a[0], a[1]


def sar_wet_dry(vv):
    """Classify SAR backscatter into wet/dry/uncertain after despeckling.
    Deliberately conservative: only pixels clearly past sar_water_db on
    either side count, since wind-roughened water sits in the ambiguous
    middle and misclassifying it would bias against the wind effect we're
    trying to observe, not just add noise."""
    finite = np.nan_to_num(vv, nan=-40.0, posinf=-40.0, neginf=-40.0)
    smooth = ndimage.median_filter(finite, size=3)
    wet = smooth < CFG["sar_water_db"] - CFG["sar_margin_db"]
    dry = smooth > CFG["sar_water_db"] + CFG["sar_margin_db"]
    return wet, dry
