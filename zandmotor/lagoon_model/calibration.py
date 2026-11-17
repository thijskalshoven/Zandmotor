"""Fits the lagoon's wet-area response to sea level or wind (--calibrate only).

Rare, expensive (walks a year of optical and radar scenes) - see
lagoon_model.runtime for the code that consumes the fitted model on every
invocation.

How an observation is measured, and why (all from the 2026-10 review of the
2025-26 record - see "What the calibration actually found" in
zandmotor_lagoon.py):
  * Only pixels the satellite actually sees count. An image is used only if
    at least sat_min_clear_lagoon of the LAGOON is visible; cloud-covered or
    missing pixels used to count as dry, which shrank the lagoon on every
    partly cloudy day.
  * Optical images are screened by their per-pixel scene classification over
    the lagoon, not by the tile-level cloud figure, which threw away clear
    images (including one of the two full-lagoon events).
  * Radar gets one threshold per relative orbit, fitted against optical
    images of the same few days, and then classifies every lagoon pixel. One
    fixed threshold read the two tracks ~2 ha apart, and its ambiguity band
    dropped half the lagoon from every scene.
  * Water is measured over the whole basin (geometry.basin_mask), counting
    what joins the water inside the outline. Measured over the outline only,
    every area was capped at the outline: after a re-trace on a quiet day
    (4.5 ha) the 2026-09-01 fill read 4.52 ha instead of 13.3, and the
    2026-11-15 image 4.33 ha instead of at least 5.7. The "seen" rule above
    still applies to the outline, so a cloud over the far basin doesn't
    discard an image; such an area is a lower bound.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import requests
from scipy import ndimage

from zandmotor.config import CFG, log
from zandmotor.errors import CalibrationIncompleteError
from zandmotor.geometry import crop_grid
from zandmotor.lagoon_model.types import LakeResponse, TidalResponse, WindResponse
from zandmotor.satellite.client import cdse_token
from zandmotor.satellite.landsat import landsat_image, landsat_scenes
from zandmotor.satellite.optical import cdse_image, cdse_scenes, clear_mask
from zandmotor.satellite.sar import (
    cdse_sar_image, cdse_sar_scenes, fit_track_threshold, sar_smooth, threshold_grid,
)
from zandmotor.tides import rws_series_long
from zandmotor.wind import fetch_wind_history, max_gust_window, sum_window


def _num(x, nd=2):
    """Round for the JSON cache; NaN (no data) becomes null."""
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _context(te, sea, weather):
    """What was going on around one observation, stored with it so a later
    re-test needs no network: sea level then and over the previous 3 days,
    gusts, and rain. `weather` is (t, gust, rain) or None."""
    sea_t, sea_v = sea
    win3d = (sea_t > te - 3 * 86400) & (sea_t <= te)
    ctx = {"sea_m": _num(np.interp(te, sea_t, sea_v)),
           "sea_max_3d_m": _num(sea_v[win3d].max() if win3d.any() else np.nan)}
    if weather is not None:
        wt, wg, wr = weather
        ctx.update({"gust_max_3h_kn": _num(max_gust_window(wt, wg, te, 3), 1),
                    "gust_max_12h_kn": _num(max_gust_window(wt, wg, te, 12), 1),
                    "rain_3d_mm": _num(sum_window(wt, wr, te - 3 * 86400, te), 1),
                    "rain_7d_mm": _num(sum_window(wt, wr, te - 7 * 86400, te), 1)})
    return ctx


class _Failures:
    """Counts failed image requests per pass. A few are normal (a timeout, a
    missing tile); many mean the record would be a fragment of the year, and a
    model fitted on a fragment must not replace the cached one."""

    def __init__(self, what):
        self.what, self.tried, self.failed = what, 0, 0

    def check(self):
        if self.tried and self.failed > max(3, CFG["calib_max_failed_frac"] * self.tried):
            raise CalibrationIncompleteError(
                f"{self.failed} of {self.tried} {self.what} image requests failed, so the "
                "calibration was stopped and the cached lagoon model left as it was. "
                "Check the CDSE credentials and connection, then run --calibrate again.")


def _joined(wet, lag):
    """Wet pixels connected (4-way) to the water inside the outline: the
    lagoon, not a separate puddle elsewhere in the basin."""
    lab, _ = ndimage.label(wet, structure=ndimage.generate_binary_structure(2, 1))
    ids = np.unique(lab[wet & lag])
    return np.isin(lab, ids[ids > 0])


def _optical_pass(token, sub, lag, reg, sea_range, cell_ha, now):
    """Every usable Sentinel-2 image of the past year over the lagoon window.
    `reg` is the basin, which is what gets measured; `lag` the outline, which
    has to be seen. Returns (obs, pixels) where pixels maps epoch -> (wet,
    seen) over the basin's pixels, for pairing with radar."""
    n_lag = int(lag.sum())
    scenes = cdse_scenes(token, now, days_back=CFG["response_days_back"],
                         max_scenes=CFG["response_max_scenes"])
    log.info("Optical: %d scene-days listed (tile cloud < %d%%)", len(scenes),
             CFG["sat_max_cloud"])
    obs, pixels = [], {}
    fails = _Failures("optical")
    for t in scenes:
        te = t.timestamp()
        if not (sea_range[0] <= te <= sea_range[1]):
            continue
        fails.tried += 1
        try:
            ndwi, scl, _ = cdse_image(token, sub, t)
        except requests.RequestException as e:
            log.warning("Skipping image %s: %s", t.date(), e)
            fails.failed += 1
            fails.check()
            continue
        seen = clear_mask(scl) & reg
        frac = (seen & lag).sum() / n_lag
        if frac < CFG["sat_min_clear_lagoon"]:
            continue                       # too much of the lagoon hidden
        wet = _joined(seen & (ndwi > CFG["ndwi_wet"]), lag)
        pixels[te] = (wet[reg], seen[reg])
        obs.append({"t": te, "sensor": "optical", "track": "S2",
                    "area_ha": float(wet.sum() * cell_ha), "seen": float(frac),
                    "seen_basin": float(seen.sum() / reg.sum())})
        log.info("  [optical] %s  wet %.2f ha  (%.0f%% of lagoon, %.0f%% of basin seen)",
                 t.strftime("%Y-%m-%d %H:%M"), obs[-1]["area_ha"], 100 * frac,
                 100 * obs[-1]["seen_basin"])
    return obs, pixels


def _sar_pass(token, sub, lag, reg, sea_range, weather, now):
    """Every usable Sentinel-1 pass, despeckled but not yet classified: the
    threshold depends on the track and is fitted afterwards.
    Returns a list of (SarScene, vv, seen) over the basin's pixels. (Radar
    counts every basin pixel below the threshold, not only what joins the
    lagoon: a flat pixel list has no neighbours. Dry sand also reads dark to
    radar, so a track that can't separate the two disagrees with optical and
    is left out by _calibrate_tracks.)"""
    n_lag = int(lag.sum())
    try:
        scenes = cdse_sar_scenes(token, now, days_back=CFG["sar_days_back"],
                                 max_scenes=CFG["sar_max_scenes"])
    except requests.RequestException as e:
        raise CalibrationIncompleteError(
            f"Could not list Sentinel-1 scenes ({e}); the cached lagoon model was left as "
            "it was. Run --calibrate again.") from e
    log.info("Radar: %d scene-days listed", len(scenes))
    out, gated = [], 0
    fails = _Failures("radar")
    for sc in scenes:
        te = sc.t.timestamp()
        if not (sea_range[0] <= te <= sea_range[1]):
            continue
        # Wind gate: rough water backscatters like land. On an early ungated
        # sample every near-zero SAR area coincided with 19-47 kn gusts. The
        # gate removes exactly those scenes, so it can't be re-checked from
        # the cached record; to re-test it, raise sar_max_wind_kn and refit.
        if weather is not None:
            gust = max_gust_window(weather[0], weather[1], te, CFG["sar_wind_check_h"])
            if np.isfinite(gust) and gust > CFG["sar_max_wind_kn"]:
                gated += 1
                continue
        fails.tried += 1
        try:
            vv, _ = cdse_sar_image(token, sub, sc.t)
        except requests.RequestException as e:
            log.warning("Skipping SAR image %s: %s", sc.t.date(), e)
            fails.failed += 1
            fails.check()
            continue
        seen = np.isfinite(vv) & reg
        if (seen & lag).sum() / n_lag < CFG["sat_min_clear_lagoon"]:
            continue
        out.append((sc, sar_smooth(vv)[reg], seen[reg]))
    log.info("Radar: %d passes kept, %d skipped for wind above %d kn", len(out), gated,
             CFG["sar_max_wind_kn"])
    return out


def _landsat_pass(token, sub, lag, reg, sea_range, now):
    """Every Landsat 8-9 scene of the past year that sees the lagoon
    (sat_min_clear_lagoon of the outline). Returns [(t, ndwi, clear)] over
    the window, unclassified: the NDWI threshold is fitted against Sentinel-2
    afterwards (_calibrate_landsat)."""
    try:
        scenes = landsat_scenes(token, now)
    except requests.RequestException as e:
        log.warning("Could not list Landsat scenes (%s); the record is Sentinel-2 only.", e)
        return []
    out = []
    fails = _Failures("Landsat")
    for t in sorted(scenes):
        if not (sea_range[0] <= t.timestamp() <= sea_range[1]):
            continue
        fails.tried += 1
        try:
            ndwi, clear = landsat_image(token, sub, t)
        except requests.RequestException as e:
            log.warning("Skipping Landsat image %s: %s", t.date(), e)
            fails.failed += 1
            fails.check()
            continue
        if (clear & lag).sum() / lag.sum() >= CFG["sat_min_clear_lagoon"]:
            out.append((t, ndwi, clear & reg))
    log.info("Landsat: %d of %d scenes see the lagoon", len(out), fails.tried)
    return out


def _calibrate_landsat(raw, optical_pixels, lag, reg, cell_ha):
    """Fit Landsat's NDWI threshold against Sentinel-2 images of the same few
    days (as for a radar track: fit_track_threshold, on -NDWI so that "wet"
    is below the threshold), then measure every scene with it - water joined
    to the outline's, over the basin. Left out entirely if it still misses
    Sentinel-2 by more than sar_max_track_mae_ha. 2025-26: threshold 0.15,
    0.40 ha on 4 pairs (1 Sep fill: 14.4 ha against Sentinel-2's 13.7).
    Returns (obs, wet_masks, info)."""
    opt_t = np.array(sorted(optical_pixels))
    pairs = []
    for t, ndwi, clear in raw:
        if not len(opt_t):
            break
        k = int(np.argmin(np.abs(opt_t - t.timestamp())))
        if abs(opt_t[k] - t.timestamp()) <= CFG["sar_pair_days"] * 86400:
            ow, osn = optical_pixels[opt_t[k]]
            pairs.append((-ndwi[reg], clear[reg], ow, osn))
    info = {"scenes": len(raw), "pairs": len(pairs), "ndwi_threshold": None, "mae_ha": None,
            "bias_ha": None, "used": False}
    if len(pairs) < CFG["sar_min_pairs"]:
        log.info("Landsat: %d pairs with Sentinel-2, too few to calibrate; not used", len(pairs))
        return [], [], info
    lo, hi, step = CFG["landsat_ndwi_grid"]
    neg_thr, err_px = fit_track_threshold(pairs, -np.arange(lo, hi + step / 2, step))
    thr = round(-neg_thr, 3)
    err = np.asarray(err_px, float) * cell_ha
    info.update(ndwi_threshold=thr, mae_ha=round(float(np.median(np.abs(err))), 2),
                bias_ha=round(float(np.median(err)), 2))
    info["used"] = info["mae_ha"] <= CFG["sar_max_track_mae_ha"]
    log.info("Landsat: %d scenes, %d Sentinel-2 pairs -> NDWI threshold %.3f, median error "
             "%.2f ha, bias %+.2f ha%s", len(raw), len(pairs), thr, info["mae_ha"],
             info["bias_ha"], "" if info["used"] else "  (NOT USED)")
    if not info["used"]:
        return [], [], info
    obs, masks = [], []
    for t, ndwi, clear in raw:
        wet = _joined(clear & (ndwi > thr), lag)
        masks.append((wet[reg], clear[reg]))
        obs.append({"t": t.timestamp(), "sensor": "landsat", "track": "Landsat 8-9",
                    "area_ha": float(wet.sum() * cell_ha),
                    "seen": float((clear & lag).sum() / lag.sum()),
                    "seen_basin": float(clear.sum() / reg.sum()), "track_mae_ha": info["mae_ha"]})
    return obs, masks, info


def _calibrate_tracks(sar_raw, optical_pixels, cell_ha):
    """Fit one threshold per relative orbit against nearby optical images and
    classify that track's passes with it. Tracks with too few pairs, or that
    still disagree with optical by more than sar_max_track_mae_ha, are left
    out rather than trusted. Returns (obs, wet_masks, tracks_info) with
    wet_masks a list of (wet, seen) over the basin's pixels."""
    opt_t = np.array(sorted(optical_pixels))
    by_track = {}
    for item in sar_raw:
        by_track.setdefault(item[0].orbit, []).append(item)
    obs, masks, info = [], [], {}
    for orbit in sorted(by_track, key=str):
        items = by_track[orbit]
        pairs = []
        for sc, vv, seen in items:
            if not len(opt_t):
                break
            k = int(np.argmin(np.abs(opt_t - sc.t.timestamp())))
            if abs(opt_t[k] - sc.t.timestamp()) <= CFG["sar_pair_days"] * 86400:
                ow, osn = optical_pixels[opt_t[k]]
                pairs.append((vv, seen, ow, osn))
        rec = {"passes": len(items), "pairs": len(pairs), "threshold_db": None,
               "mae_ha": None, "bias_ha": None, "used": False}
        if len(pairs) >= CFG["sar_min_pairs"]:
            thr, err_px = fit_track_threshold(pairs, threshold_grid())
            err = np.asarray(err_px, float) * cell_ha
            rec.update(threshold_db=thr, mae_ha=round(float(np.median(np.abs(err))), 2),
                       bias_ha=round(float(np.median(err)), 2))
            rec["used"] = rec["mae_ha"] <= CFG["sar_max_track_mae_ha"]
        info[str(orbit)] = rec
        log.info("Radar orbit %s: %d passes, %d optical pairs within %d days -> threshold %s "
                 "dB, median error %s ha, bias %s ha%s", orbit, rec["passes"], rec["pairs"],
                 CFG["sar_pair_days"], rec["threshold_db"], rec["mae_ha"], rec["bias_ha"],
                 "" if rec["used"] else "  (NOT USED)")
        if not rec["used"]:
            continue
        for sc, vv, seen in items:
            wet = seen & (vv < rec["threshold_db"])
            masks.append((wet, seen))
            obs.append({"t": sc.t.timestamp(), "sensor": "sar", "track": f"S1 orbit {orbit}",
                        "area_ha": float(wet.sum() * cell_ha), "seen": float(seen.mean()),
                        "track_mae_ha": rec["mae_ha"]})
    return obs, masks, info


def fetch_sea_record(now):
    """A year of measured sea level (t, level) for the fit and the fill-and-drain
    replay, fetched once per --calibrate."""
    loc = CFG["rws_locations"][0]
    span_start = now - dt.timedelta(days=CFG["response_days_back"] + 2)
    try:
        return rws_series_long(loc, span_start, now, "meting")
    except (requests.RequestException, RuntimeError) as e:
        raise CalibrationIncompleteError(
            f"Could not fetch a year of measured sea level from Rijkswaterstaat ({e}); "
            "the cached lagoon model was left as it was. Run --calibrate again.") from e


def fit_lagoon_response(grid, lagoon, now, basin=None, sea=None):
    """For each usable satellite image over the past year, measure the
    lagoon's wet area and compare it with the sea level at that moment
    (searching over a range of lags). If area tracks the sea closely, fit a
    delay + damping + floor. If not, check wind gust intensity instead. If
    neither holds up, treat the lagoon as a slowly-varying lake, its area
    read from each image directly.

    Also builds the wetness frequency map: how often each lagoon pixel was
    actually observed wet, out of the times it was actually seen. Depth
    doesn't matter here, only whether a spot holds water at all, so the
    observations answer the only question that matters directly.

    Measures over `basin` (the outline when not given), counting the water
    that joins the outline's; see the module docstring. `sea` is the
    fetch_sea_record result, fetched here when not given.
    Returns (model, freq, valid_count) or (None, None, None)."""
    basin = lagoon if basin is None else basin | lagoon
    pad = int(CFG["calib_zone_m"] / CFG["grid_res_m"])
    sub, (rs, cs) = crop_grid(grid, basin, pad)
    lag, reg = lagoon[rs, cs], basin[rs, cs]
    cell_ha = grid.cell_area_m2 / 1e4

    token = cdse_token()
    loc = CFG["rws_locations"][0]
    span_start = now - dt.timedelta(days=CFG["response_days_back"] + 2)
    sea_t, sea_v = fetch_sea_record(now) if sea is None else sea
    if len(sea_t) < 100:
        log.warning("Not enough historical sea level data to fit a lagoon response model.")
        return None, None, None
    sea_range = (sea_t[0], sea_t[-1])

    lat_c = (CFG["bbox_wgs84"][1] + CFG["bbox_wgs84"][3]) / 2
    lon_c = (CFG["bbox_wgs84"][0] + CFG["bbox_wgs84"][2]) / 2
    weather = None
    try:
        wt, _, wg, wr = fetch_wind_history(lat_c, lon_c, span_start - dt.timedelta(days=8), now)
        weather = (wt, wg, wr)
    except requests.RequestException as e:
        log.warning("Could not fetch weather history; SAR scenes go ungated and the wind "
                    "test is skipped (%s)", e)

    opt_obs, opt_pixels = _optical_pass(token, sub, lag, reg, sea_range, cell_ha, now)
    ls_obs, ls_masks, ls_info = _calibrate_landsat(
        _landsat_pass(token, sub, lag, reg, sea_range, now), opt_pixels, lag, reg, cell_ha)
    if CFG["sar_in_calibration"]:
        sar_raw = _sar_pass(token, sub, lag, reg, sea_range, weather, now)
        sar_obs, sar_masks, sar_tracks = _calibrate_tracks(sar_raw, opt_pixels, cell_ha)
    else:
        sar_obs, sar_masks, sar_tracks = [], [], {}
    # per-sensor calibration records; Landsat's sits with the radar tracks'
    sar_tracks["landsat"] = ls_info

    # Wetness frequency over the basin window, pasted back into full-grid arrays.
    wet_c = np.zeros(reg.shape, "int32")
    valid_c = np.zeros(reg.shape, "int32")
    for wet, seen in list(opt_pixels.values()) + ls_masks + sar_masks:
        wet_c[reg] += wet
        valid_c[reg] += seen
    wet_count = np.zeros(lagoon.shape, "int32")
    valid_count = np.zeros(lagoon.shape, "int32")
    wet_count[rs, cs] = wet_c
    valid_count[rs, cs] = valid_c
    freq = wet_count / np.maximum(valid_count, 1)

    obs = sorted(opt_obs + ls_obs + sar_obs, key=lambda o: o["t"])
    for o in obs:
        o.update(_context(o["t"], (sea_t, sea_v), weather))
        o["area_ha"] = round(o["area_ha"], 3)
        o["seen"] = round(o["seen"], 3)
    n = len(obs)
    if n < CFG["response_min_scenes"]:
        log.warning("Only %d usable images over the lagoon; not enough to fit a "
                    "response model.", n)
        return None, freq, valid_count
    obs_t = np.array([o["t"] for o in obs])
    obs_area = np.array([o["area_ha"] for o in obs])
    n_opt, n_sar = len(opt_obs) + len(ls_obs), len(sar_obs)
    log.info("Combined observation set: %d Sentinel-2 + %d Landsat + %d SAR = %d images",
             len(opt_obs), len(ls_obs), n_sar, n)

    best_r, best_delay = 0.0, 0
    for dmin in CFG["response_delays_min"]:
        sea_lagged = np.interp(obs_t - dmin * 60, sea_t, sea_v)
        if sea_lagged.std() < 1e-6 or obs_area.std() < 1e-6:
            continue
        r = float(np.corrcoef(sea_lagged, obs_area)[0, 1])
        if abs(r) > abs(best_r):
            best_r, best_delay = r, int(dmin)

    record = {"times": obs_t.tolist(), "areas": obs_area.tolist(), "n_optical": n_opt,
              "n_sar": n_sar, "obs": obs, "sar_tracks": sar_tracks}

    if abs(best_r) >= CFG["tidal_r_min"]:
        sea_best = np.interp(obs_t - best_delay * 60, sea_t, sea_v)
        m, c = [float(v) for v in np.polyfit(sea_best, obs_area, 1)]
        floor = float(np.percentile(obs_area, 5))
        log.info("Lagoon response: TIDAL - delay %d min, damping %.2f ha/m, floor %.2f ha "
                 "(r=%.2f, n=%d images)", best_delay, m, floor, best_r, n)
        return TidalResponse(delay_min=best_delay, damping=m, offset=c, floor_ha=floor,
                             r=best_r, n=n, rws_location=loc, fitted=now.isoformat(),
                             **record), freq, valid_count

    # not tidal (sea level explains ~nothing): check whether wind does instead
    best_wr, best_wh = 0.0, None
    if weather is not None:
        for hrs in CFG["wind_windows_h"]:
            feat = np.array([max_gust_window(weather[0], weather[1], t, hrs) for t in obs_t])
            ok = np.isfinite(feat)
            if ok.sum() < CFG["response_min_scenes"] or feat[ok].std() < 1e-6:
                continue
            r = float(np.corrcoef(feat[ok], obs_area[ok])[0, 1])
            if abs(r) > abs(best_wr):
                best_wr, best_wh = r, hrs

    if best_wh is not None and abs(best_wr) >= CFG["wind_r_min"] and n >= CFG["wind_min_n"]:
        feat = np.array([max_gust_window(weather[0], weather[1], t, best_wh) for t in obs_t])
        ok = np.isfinite(feat)
        b, a = [float(v) for v in np.polyfit(feat[ok], obs_area[ok], 1)]
        log.info("Lagoon response: WIND (experimental, unconfirmed at n=%d) - %dh peak "
                 "gust, r=%.2f; area = %.3f*gust_kn + %.2f ha, clipped to the observed "
                 "range", n, best_wh, best_wr, b, a)
        return WindResponse(window_h=best_wh, slope=b, offset=a,
                            area_min_ha=float(obs_area[ok].min()),
                            area_max_ha=float(obs_area[ok].max()),
                            r=best_wr, sea_r=best_r, n=n, fitted=now.isoformat(),
                            **record), freq, valid_count

    log.info("Lagoon response: LAKE - neither sea level (r=%.2f) nor wind (r=%.2f) clear "
             "the threshold at n=%d images; area read from each image directly instead.",
             best_r, best_wr, n)
    return LakeResponse(r=best_r, wind_r=best_wr, n=n, fitted=now.isoformat(),
                        **record), freq, valid_count
