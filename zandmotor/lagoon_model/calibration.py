"""Fits the lagoon's wet-area response to sea level or wind (--calibrate only).

Rare, expensive (walks up to ~180 satellite scenes) - see lagoon_model.runtime
for the code that actually consumes the fitted model on every invocation.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import requests
from scipy import ndimage

from zandmotor.config import CFG, UTC, log
from zandmotor.satellite.client import cdse_token
from zandmotor.satellite.optical import cdse_image, cdse_scenes
from zandmotor.satellite.sar import cdse_sar_image, cdse_sar_scenes, sar_wet_dry
from zandmotor.tides import rws_series
from zandmotor.wind import fetch_wind_history, max_gust_window


def fit_lagoon_response(grid, lagoon, now):
    """For each usable satellite image over the past year, measure the
    lagoon's wet area and compare it with the sea level at that moment
    (searching over a range of lags). If area tracks the sea closely, fit a
    delay + damping + floor. If not, check wind gust intensity instead
    (experimental - see wind_windows_h above). If neither holds up, treat
    the lagoon as a slowly-varying lake, its area read from each image
    directly.

    Also builds the wetness frequency map: how often each lagoon pixel was
    actually observed wet. Depth doesn't matter here, only whether a spot
    holds water at all, so there's no need to infer bed elevation from any
    of this the way the old waterline-calibration approach did - the
    observations answer the only question that matters directly.

    Returns (model, freq, valid_count) or (None, None, None)."""
    zone = ndimage.binary_dilation(lagoon, iterations=int(CFG["calib_zone_m"] /
                                                          CFG["grid_res_m"]))
    wet_count = np.zeros(lagoon.shape, "int32")
    valid_count = np.zeros(lagoon.shape, "int32")

    token = cdse_token()
    scenes = cdse_scenes(token, now, days_back=CFG["response_days_back"],
                         max_scenes=CFG["response_max_scenes"])
    loc = CFG["rws_locations"][0]
    sea_t, sea_v = rws_series(loc, now - dt.timedelta(days=CFG["response_days_back"] + 2),
                              now, "meting")
    if len(sea_t) < 100:
        log.warning("Not enough historical sea level data to fit a lagoon response model.")
        return None, None, None

    obs_t, obs_area, obs_src = [], [], []
    for t in scenes:
        te = t.timestamp()
        if not (sea_t[0] <= te <= sea_t[-1]):
            continue
        try:
            ndwi, scl, _ = cdse_image(token, grid, t)
        except requests.RequestException as e:
            log.warning("Skipping image %s: %s", t.date(), e)
            continue
        clear = np.isin(scl.round(), [4, 5, 6, 7, 11])
        valid = zone & clear
        if valid.sum() < 0.5 * zone.sum():
            continue                       # too cloudy over the lagoon itself
        wet = valid & (ndwi > CFG["ndwi_wet"])
        wet_count[valid & lagoon] += wet[valid & lagoon]
        valid_count[valid & lagoon] += 1
        obs_t.append(te)
        obs_area.append((wet & lagoon).sum() * grid.cell_area_m2 / 1e4)
        obs_src.append("optical")
        log.info("  [optical] %s  wet %.2f ha", t.strftime("%Y-%m-%d %H:%M"), obs_area[-1])

    # Sentinel-1 SAR: not cloud-limited, so far more usable days/year here (~120 vs
    # ~20) - crucially including the calm/low-wind conditions optical alone keeps
    # missing. But SAR water detection needs calm water to look dark. On an early
    # ungated sample, every near-zero-area SAR reading coincided with a 19-47 kn gust
    # while every genuinely-wet one stayed under ~29 kn, i.e. rough water read as dry
    # - exactly backwards from any real effect, so those scenes are skipped rather
    # than trusted. See sar_max_wind_kn: the gate now removes those scenes, so this
    # can't be re-confirmed from the cached record without raising it and refitting.
    sar_wind = None
    try:
        sar_scenes = cdse_sar_scenes(token, now, days_back=CFG["sar_days_back"],
                                     max_scenes=CFG["sar_max_scenes"])
        if sar_scenes:
            lat_c = (CFG["bbox_wgs84"][1] + CFG["bbox_wgs84"][3]) / 2
            lon_c = (CFG["bbox_wgs84"][0] + CFG["bbox_wgs84"][2]) / 2
            start = min(t.timestamp() for t in sar_scenes)
            start = dt.datetime.fromtimestamp(start, UTC) - dt.timedelta(hours=6)
            sw_t, _, sw_g = fetch_wind_history(lat_c, lon_c, start, now)
            sar_wind = (sw_t, sw_g)
    except requests.RequestException as e:
        log.warning("Could not list Sentinel-1 scenes or their wind context (%s)", e)
        sar_scenes = []
    for t in sar_scenes:
        te = t.timestamp()
        if not (sea_t[0] <= te <= sea_t[-1]):
            continue
        if sar_wind is not None:
            recent_gust = max_gust_window(sar_wind[0], sar_wind[1], te, CFG["sar_wind_check_h"])
            if np.isfinite(recent_gust) and recent_gust > CFG["sar_max_wind_kn"]:
                continue                   # too windy to trust SAR's water/land call
        try:
            vv, _ = cdse_sar_image(token, grid, t)
        except requests.RequestException as e:
            log.warning("Skipping SAR image %s: %s", t.date(), e)
            continue
        wet_px, dry_px = sar_wet_dry(vv)
        valid = zone & (wet_px | dry_px)
        if valid.sum() < 0.5 * zone.sum():
            continue                       # too much ambiguous/rough water to trust
        wet_count[valid & lagoon] += wet_px[valid & lagoon]
        valid_count[valid & lagoon] += 1
        obs_t.append(te)
        obs_area.append((wet_px & lagoon & valid).sum() * grid.cell_area_m2 / 1e4)
        obs_src.append("sar")
        log.info("  [sar]     %s  wet %.2f ha", t.strftime("%Y-%m-%d %H:%M"), obs_area[-1])

    n = len(obs_t)
    freq = wet_count / np.maximum(valid_count, 1)
    if n < CFG["response_min_scenes"]:
        log.warning("Only %d usable images over the lagoon; not enough to fit a "
                    "response model.", n)
        return None, freq, valid_count
    obs_t, obs_area, obs_src = np.array(obs_t), np.array(obs_area), np.array(obs_src)
    n_opt, n_sar = int((obs_src == "optical").sum()), int((obs_src == "sar").sum())
    log.info("Combined observation set: %d optical + %d SAR = %d images", n_opt, n_sar, n)

    # cross-check SAR against optical on nearby dates - if SAR systematically reads
    # low (wind-roughened water misread as dry), that would bias any wind-driven
    # fit against itself, so this needs to be visible, not just trusted silently
    opt_i, sar_i = obs_src == "optical", obs_src == "sar"
    if opt_i.any() and sar_i.any():
        diffs = [obs_area[sar_i][np.argmin(np.abs(obs_t[sar_i] - ot))] - oa
                for ot, oa in zip(obs_t[opt_i], obs_area[opt_i])
                if (np.abs(obs_t[sar_i] - ot) < 2 * 86400).any()]
        if diffs:
            log.info("SAR vs optical area, %d nearby-date pairs (<=2d apart): "
                    "mean diff %+.2f ha (SAR - optical), std %.2f ha",
                    len(diffs), float(np.mean(diffs)), float(np.std(diffs)))

    best_r, best_delay = 0.0, 0
    for dmin in CFG["response_delays_min"]:
        sea_lagged = np.interp(obs_t - dmin * 60, sea_t, sea_v)
        if sea_lagged.std() < 1e-6 or obs_area.std() < 1e-6:
            continue
        r = float(np.corrcoef(sea_lagged, obs_area)[0, 1])
        if abs(r) > abs(best_r):
            best_r, best_delay = r, int(dmin)

    order = np.argsort(obs_t)
    fallback = {"times": obs_t[order].tolist(), "areas": obs_area[order].tolist(),
               "n_optical": n_opt, "n_sar": n_sar}

    if abs(best_r) >= CFG["tidal_r_min"]:
        sea_best = np.interp(obs_t - best_delay * 60, sea_t, sea_v)
        m, c = [float(v) for v in np.polyfit(sea_best, obs_area, 1)]
        floor = float(np.percentile(obs_area, 5))
        log.info("Lagoon response: TIDAL - delay %d min, damping %.2f ha/m, floor %.2f ha "
                "(r=%.2f, n=%d images)", best_delay, m, floor, best_r, n)
        return {"mode": "tidal", "delay_min": best_delay, "damping": m, "offset": c,
               "floor_ha": floor, "r": best_r, "n": n, "rws_location": loc,
               "fitted": now.isoformat(), **fallback}, freq, valid_count

    # not tidal (sea level explains ~nothing): check whether wind does instead
    lat_c = (CFG["bbox_wgs84"][1] + CFG["bbox_wgs84"][3]) / 2
    lon_c = (CFG["bbox_wgs84"][0] + CFG["bbox_wgs84"][2]) / 2
    best_wr, best_wh = 0.0, None
    try:
        wind_start = dt.datetime.fromtimestamp(obs_t.min(), UTC) - dt.timedelta(days=3)
        wind_t, _, wind_gst = fetch_wind_history(lat_c, lon_c, wind_start, now)
        for hrs in CFG["wind_windows_h"]:
            feat = np.array([max_gust_window(wind_t, wind_gst, t, hrs) for t in obs_t])
            ok = np.isfinite(feat)
            if ok.sum() < CFG["response_min_scenes"] or feat[ok].std() < 1e-6:
                continue
            r = float(np.corrcoef(feat[ok], obs_area[ok])[0, 1])
            if abs(r) > abs(best_wr):
                best_wr, best_wh = r, hrs
    except requests.RequestException as e:
        log.warning("Could not fetch wind history for the response fit (%s)", e)

    if best_wh is not None and abs(best_wr) >= CFG["wind_r_min"]:
        feat = np.array([max_gust_window(wind_t, wind_gst, t, best_wh) for t in obs_t])
        ok = np.isfinite(feat)
        b, a = [float(v) for v in np.polyfit(feat[ok], obs_area[ok], 1)]
        log.info("Lagoon response: WIND (experimental, unconfirmed at n=%d) - %dh peak "
                "gust, r=%.2f; area = %.3f*gust_kn + %.2f ha, clipped to the observed "
                "range", n, best_wh, best_wr, b, a)
        return {"mode": "wind", "window_h": best_wh, "slope": b, "offset": a,
               "area_min_ha": float(obs_area[ok].min()), "area_max_ha": float(obs_area[ok].max()),
               "r": best_wr, "sea_r": best_r, "n": n, "fitted": now.isoformat(), **fallback}, \
              freq, valid_count

    log.info("Lagoon response: LAKE - neither sea level (r=%.2f) nor wind (r=%.2f) clear "
            "the threshold at n=%d images; area read from each image directly instead.",
            best_r, best_wr, n)
    return {"mode": "lake", "r": best_r, "wind_r": best_wr, "n": n, "fitted": now.isoformat(),
           **fallback}, freq, valid_count
