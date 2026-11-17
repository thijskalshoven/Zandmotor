"""Sentinel-1 SAR scene listing, VV/VH extraction, and wet/dry classification."""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
from scipy import ndimage

from zandmotor.config import CFG
from zandmotor.satellite.client import catalog_search, dedupe_scenes_one_per_day, process_image
from zandmotor.time_utils import parse_iso_datetime

SAR_EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["VV", "VH"]}],
          output: {bands: 2, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) {
  return [10*Math.log10(s.VV), 10*Math.log10(s.VH)];
}"""


@dataclasses.dataclass(frozen=True)
class SarScene:
    """One Sentinel-1 pass. The relative orbit is kept because it fixes the
    viewing angle, and the two descending tracks here read the same lagoon
    ~3 dB apart - so every SAR number has to know which track it came from."""
    t: dt.datetime
    orbit: int | None
    platform: str | None = None


def cdse_sar_scenes(token, now, days_back=None, max_scenes=None):
    """Sentinel-1 GRD passes, one per day, newest first, in a fixed orbit
    direction (ascending and descending look very different over the same
    ground). Returns SarScene records."""
    start = now - dt.timedelta(days=days_back or CFG["sar_days_back"])
    cql = {"op": "=", "args": [{"property": "sat:orbit_state"}, CFG["sar_orbit"]]}
    feats = catalog_search(token, "sentinel-1-grd", start, now, cql, chunk_days=60)
    scenes = [SarScene(parse_iso_datetime(f["properties"]["datetime"]),
                       f["properties"].get("sat:relative_orbit"),
                       f["properties"].get("platform")) for f in feats]
    return dedupe_scenes_one_per_day(scenes, key=lambda s: s.t)[
        :max_scenes or CFG["sar_max_scenes"]]


def cdse_sar_image(token, grid, t):
    """Returns vv, vh (dB backscatter) on the grid, for the single pass at t."""
    a = process_image(token, grid, t, collection="sentinel-1-grd", evalscript=SAR_EVALSCRIPT,
                      extra_data_filter={"acquisitionMode": "IW", "polarization": "DV",
                                         "orbitDirection": CFG["sar_orbit"].upper()},
                      time_window=dt.timedelta(minutes=CFG["sar_scene_window_min"]))
    return a[0], a[1]


def sar_smooth(vv):
    """Despeckled VV (3x3 median). Missing data becomes -40 dB here, so
    callers that care must track np.isfinite(vv) themselves."""
    finite = np.nan_to_num(vv, nan=-40.0, posinf=-40.0, neginf=-40.0)
    return ndimage.median_filter(finite, size=3)


def sar_wet_dry(vv):
    """Classify SAR backscatter into wet/dry/uncertain after despeckling,
    with one fixed threshold and an ambiguity band.

    Used only by the flats dry-override now. It is NOT used for the lagoon:
    there, half the lagoon fell in the ambiguous band on every scene and was
    counted as dry, and the fixed threshold read the two tracks ~2 ha apart
    (see fit_track_threshold)."""
    smooth = sar_smooth(vv)
    wet = smooth < CFG["sar_water_db"] - CFG["sar_margin_db"]
    dry = smooth > CFG["sar_water_db"] + CFG["sar_margin_db"]
    return wet, dry


def threshold_grid():
    a, b, step = CFG["sar_threshold_grid_db"]
    return np.arange(a, b + step / 2, step)


def fit_track_threshold(pairs, thresholds):
    """The VV threshold under which one track's SAR best agrees with optical.

    `pairs` is a list of (vv, sar_seen, opt_wet, opt_seen), all 1-D arrays over
    the same lagoon pixels: a SAR scene and an optical scene of the same few
    days. For each candidate threshold, each pair scores the share of jointly
    seen pixels the two disagree on; the threshold with the lowest MEDIAN score
    wins. The median rather than the sum: a pair can straddle a fill or a
    drain (the lagoon went from 12.4 to 2.4 ha in ten days in May 2026), and
    one such pair shouldn't drag the whole track's threshold.

    Pixel agreement rather than matching areas: equal areas can hide two
    different shapes, and the pixels are what the wetness map is built from.

    Returns (threshold_db, area_errors_px) with one SAR-minus-optical wet-pixel
    difference per pair at that threshold, or (None, []) without pairs."""
    if not pairs:
        return None, []
    thresholds = np.asarray(thresholds, float)
    scores = np.empty((len(thresholds), len(pairs)))
    for j, (vv, sar_seen, opt_wet, opt_seen) in enumerate(pairs):
        both = sar_seen & opt_seen
        n = max(int(both.sum()), 1)
        v, o = vv[both], opt_wet[both]
        for i, thr in enumerate(thresholds):
            scores[i, j] = np.count_nonzero((v < thr) != o) / n
    med = np.median(scores, axis=1)
    best = np.flatnonzero(med == med.min())
    thr = float(thresholds[best[len(best) // 2]])     # middle of a tied run
    errors = []
    for vv, sar_seen, opt_wet, opt_seen in pairs:
        both = sar_seen & opt_seen
        errors.append(int(np.count_nonzero(vv[both] < thr)) - int(np.count_nonzero(opt_wet[both])))
    return thr, errors
