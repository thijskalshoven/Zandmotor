"""Consumes the fitted lagoon response model - runs on every invocation."""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import numpy as np

from zandmotor.config import CFG, UTC, log
from zandmotor.wind import max_gust_window


def lake_recent_observations(model, now):
    """The recent satellite observations that lake mode should average over.
    Drops implausibly small readings (detection failures - the record holds a
    0.02 ha reading among 64, and the lagoon did not drain that day) and then
    widens the time window until at least lake_min_obs remain.
    Returns (times, areas) with the most recent last, or (None, None)."""
    times, areas = np.array(model["times"]), np.array(model["areas"])
    keep = areas >= CFG["lake_min_plausible_ha"]
    dropped = int((~keep).sum())
    times, areas = times[keep], areas[keep]
    if not len(times):
        return None, None
    if dropped:
        log.debug("Lake mode: ignored %d observation(s) below %.2f ha as failed detections",
                  dropped, CFG["lake_min_plausible_ha"])
    span = CFG["lake_recent_days"]
    while span < 400:
        sel = times >= now.timestamp() - span * 86400
        if sel.sum() >= CFG["lake_min_obs"]:
            return times[sel], areas[sel]
        span *= 2
    return times, areas


def lake_area_summary(model, now):
    """Lake-mode area as a measurement with its spread, not a forecast:
    (median_ha, p25_ha, p75_ha, as_of_datetime, n_used).

    The area does not vary over the forecast - nothing in the record predicts
    it (sea r=-0.05, wind r=0.11 at n=64) - so this is one value for the whole
    run and the UI says so. The median rather than the newest scene: a single
    noisy image should not be able to claim the lagoon is gone."""
    times, areas = lake_recent_observations(model, now)
    if times is None:
        return None, None, None, None, 0
    return (float(np.median(areas)), float(np.percentile(areas, 25)),
            float(np.percentile(areas, 75)),
            dt.datetime.fromtimestamp(float(times[-1]), UTC), int(len(areas)))


def lagoon_area_from_model(model, sea_level_fn, te, wind_hist=None, lake_area=None):
    """How many hectares of the lagoon are wet at time te. sea_level_fn(epoch)
    -> (level, src), same signature as WaterLevel.level. wind_hist, if given,
    is (t_arr, gust_arr) covering at least the trailing window the wind
    model needs (only used in "wind" mode). lake_area is the precomputed
    median from lake_area_summary, used in "lake" mode."""
    if model["mode"] == "tidal":
        sea_lag, _ = sea_level_fn(te - model["delay_min"] * 60)
        area = model["damping"] * sea_lag + model["offset"]
        return max(area, model["floor_ha"])
    if model["mode"] == "wind" and wind_hist is not None:
        gust = max_gust_window(wind_hist[0], wind_hist[1], te, model["window_h"])
        if np.isfinite(gust):
            area = model["slope"] * gust + model["offset"]
            return float(np.clip(area, model["area_min_ha"], model["area_max_ha"]))
    if lake_area is not None:
        return lake_area
    times, areas = np.array(model["times"]), np.array(model["areas"])
    return float(np.interp(te, times, areas))


def wetness_fingerprint(lagoon, ring):
    """Identifies the exact outline a wetness frequency map was fitted for.

    The frequency map is only meaningful for the lagoon footprint it was built
    from: re-trace the outline and its per-pixel values refer to ground that
    is no longer in the mask, while newly-included cells have never been
    observed at all. Comparing array shapes does not catch this - the shape is
    fixed by bbox and resolution, so it matches no matter which outline was
    used."""
    h = hashlib.sha1()
    h.update(json.dumps({"bbox": list(CFG["bbox_wgs84"]), "res": CFG["grid_res_m"],
                         "ring": [[round(c, 6) for c in p] for p in ring]},
                        sort_keys=True).encode())
    h.update(np.packbits(lagoon).tobytes())
    return h.hexdigest()


def wet_mask_from_area(freq, lagoon, target_ha, cell_area_m2):
    """Which lagoon cells are wet for a given target area: the ones most
    often observed wet in the satellite record fill in first - a direct
    measurement of where the water actually is, not a bathymetry guess."""
    target_px = int(round(target_ha * 1e4 / cell_area_m2))
    ys, xs = np.nonzero(lagoon)
    order = np.argsort(-freq[ys, xs])
    mask = np.zeros(lagoon.shape, bool)
    chosen = order[:max(target_px, 0)]
    mask[ys[chosen], xs[chosen]] = True
    return mask
