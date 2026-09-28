"""Per-day-cached, satellite-derived dry override for the intertidal flats."""

from __future__ import annotations

import datetime as dt

import numpy as np
import requests

from zandmotor.config import CACHE, CFG, UTC, log
from zandmotor.satellite.client import cdse_token
from zandmotor.satellite.optical import cdse_image, cdse_scenes
from zandmotor.satellite.sar import cdse_sar_image, cdse_sar_scenes, sar_wet_dry
from zandmotor.wind import fetch_wind_history, max_gust_window


def flats_dry_override(grid, lagoon, now):
    """A recent satellite record beats old elevation data on the flats:
    wherever it's consistently dry, treat it as dry regardless of what the
    DEM-based flood model predicts there - important since the elevation+
    connectivity bathtub model has no idea about real dikes/pumped drainage
    (a nearby town can look "flooded" even though it demonstrably isn't) or
    isolated dune depressions that aren't actually tidally connected
    despite looking low enough in the DEM.

    Uses several recent scenes (optical + SAR, with the same wind-gate used
    elsewhere), not just one: single-scene SAR backscatter is noisy enough
    that a known-dry patch and a known-wet one can each read either way on
    a given day (validated here - over 10 dates, a random dry dune patch
    swung -11 to -22 dB, overlapping the water range). A cell only counts
    as dry-confirmed once it has enough independent looks and most of them
    say dry - a plain majority vote, not zero-tolerance for a wet reading
    (that turned out to never fire on exactly the noisy cells this exists
    to fix). Cells that are wet most of the time (the real lagoon, real
    canals) still won't qualify, since their wet count stays high.
    Cached per calendar day so a normal run doesn't refetch imagery."""
    cache_file = CACHE / f"flats_dry_{now.date()}.npy"
    if cache_file.exists():
        cached = np.load(cache_file)
        if cached.shape == lagoon.shape:
            return cached

    dry_count = np.zeros(lagoon.shape, "int32")
    wet_count = np.zeros(lagoon.shape, "int32")
    n_used = 0
    try:
        token = cdse_token()
        scenes = cdse_scenes(token, now, days_back=CFG["flats_dry_days_back"],
                             max_scenes=CFG["flats_dry_max_scenes"])
        for t in scenes:
            try:
                ndwi, scl, _ = cdse_image(token, grid, t)
            except requests.RequestException as e:
                log.warning("Skipping optical image %s: %s", t.date(), e)
                continue
            clear = np.isin(scl.round(), [4, 5, 6, 7, 11])
            if clear.mean() < 0.8:
                continue
            dry_count[clear & (ndwi < CFG["ndwi_dry"])] += 1
            wet_count[clear & (ndwi > CFG["ndwi_wet"])] += 1
            n_used += 1

        sar_scenes = cdse_sar_scenes(token, now, days_back=CFG["flats_dry_days_back"],
                                     max_scenes=CFG["flats_dry_max_scenes"])
        wind_hist = None
        if sar_scenes:
            lat_c = (CFG["bbox_wgs84"][1] + CFG["bbox_wgs84"][3]) / 2
            lon_c = (CFG["bbox_wgs84"][0] + CFG["bbox_wgs84"][2]) / 2
            start = dt.datetime.fromtimestamp(min(t.timestamp() for t in sar_scenes),
                                              UTC) - dt.timedelta(hours=6)
            try:
                wt, _, wg = fetch_wind_history(lat_c, lon_c, start, now)
                wind_hist = (wt, wg)
            except requests.RequestException:
                pass
        for t in sar_scenes:
            if wind_hist is not None:
                gust = max_gust_window(wind_hist[0], wind_hist[1], t.timestamp(),
                                       CFG["sar_wind_check_h"])
                if np.isfinite(gust) and gust > CFG["sar_max_wind_kn"]:
                    continue
            try:
                vv, _ = cdse_sar_image(token, grid, t)
            except requests.RequestException as e:
                log.warning("Skipping SAR image %s: %s", t.date(), e)
                continue
            wet_px, dry_px = sar_wet_dry(vv)
            if (wet_px | dry_px).mean() < 0.5:
                continue
            dry_count[dry_px] += 1
            wet_count[wet_px] += 1
            n_used += 1
    except (requests.RequestException, RuntimeError) as e:
        log.warning("Could not fetch recent images for the flats dry-override (%s)", e)

    if n_used < CFG["flats_dry_min_looks"]:
        log.warning("Only %d usable recent image(s) for the flats dry-override (need "
                    "%d independent looks); skipping it this run.",
                    n_used, CFG["flats_dry_min_looks"])
        return None

    # Majority vote, not a zero-tolerance veto: single-scene noise means even a
    # genuinely dry cell will occasionally read "wet" (validated above), so
    # requiring zero wet looks ends up never overriding the noisy cells this
    # exists to fix. Clearly wet-most-of-the-time cells (the real lagoon, real
    # canals) still won't qualify since their wet_count stays high.
    dry = (dry_count >= CFG["flats_dry_min_looks"]) & (dry_count > wet_count) & ~lagoon
    np.save(cache_file, dry)
    log.info("Flats dry-override from %d recent images (optical+SAR): %.2f ha "
            "confirmed dry", n_used, dry.sum() * grid.cell_area_m2 / 1e4)
    return dry
