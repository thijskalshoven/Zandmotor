"""Configuration: everything you might want to tweak is here."""

from __future__ import annotations

import datetime as dt
import logging
import pathlib
from zoneinfo import ZoneInfo

CFG = {
    # Map area (lon_min, lat_min, lon_max, lat_max), covers the whole Zandmotor
    "bbox_wgs84": (4.165, 52.035, 4.225, 52.070),
    "grid_res_m": 5.0,                      # ground resolution of the model

    # Optional (lon, lat) point inside the lagoon, to help --outline pick it
    "lagoon_hint": (4.1938, 52.0565),
    "outline_days_back": 60,               # search this far back for an image
    "outline_min_ha": 2.0,                 # ignore water bodies smaller than this
    # Last-resort placeholder outline (lon, lat), only used if nothing else works
    "lagoon_polygon": [(4.186, 52.055), (4.204, 52.055), (4.206, 52.064),
                       (4.190, 52.066), (4.186, 52.055)],

    # Terrain
    "ahn_wcs": "https://service.pdok.nl/rws/ahn/wcs/v1_0",
    "ahn_coverage": "dtm_05m",
    "ahn_request_res_m": 2.0,
    "local_dem": None,                      # path to your own GeoTIFF (any CRS, m NAP)
    "sea_fill_level": -6.0,                 # m NAP for missing data offshore, and a
                                             # cosmetic fill for the lagoon interior (its
                                             # bed elevation isn't modelled - see the
                                             # module docstring on why depth doesn't matter)
    "sea_seed_level": -1.0,                 # border cells below this = open sea
    "lagoon_trim_land_m": 0.5,              # AHN elevation above this inside the traced
                                             # outline is real land, not lagoon (the source
                                             # trace can overshoot onto beach/dunes since
                                             # Zandmotor's coastline keeps reshaping)
    "coastal_zone_m": 500,                  # only render flooding this far inland of the
                                             # open sea (plus the lagoon itself). Real towns
                                             # nearby sit below sea level behind dikes and
                                             # pumped drainage an elevation-only flood model
                                             # has no way to know about, and would otherwise
                                             # get rendered as "flooded"

    # Water level (Rijkswaterstaat)
    "rws_url": "https://ddapi20-waterwebservices.rijkswaterstaat.nl/"
               "ONLINEWAARNEMINGENSERVICES/OphalenWaarnemingen",
    "rws_locations": ["scheveningen", "hoekvanholland"],
    "surge_decay_hours": 12.0,              # for astro + surge fallback

    # Lagoon behaviour
    "lagoon_lag_min": 30,                   # lagoon level lags the sea by this
    "pond_min_depth": 0.05,                 # ignore ponds shallower than this
    "pond_drawdown": 0.05,                  # retained water sits a bit below sill
    "flood_margin_m": 0.10,                 # a connecting route must be at least this
                                             # far below the water level to flood (a thin,
                                             # averaged-down sand ridge shouldn't leak)
    "min_visible_depth_m": 0.10,            # hide water films thinner than this
    "min_patch_ha": 0.02,                   # drop isolated wet patches smaller than this

    # Kitesurfing. Depth doesn't matter here, only whether there's water at
    # all: just need enough contiguous surface to ride on. Counted over the
    # lagoon plus the intertidal beach strip, never the open sea.
    "min_rideable_ha": 3.0,                 # need at least this much water to ride
    "gusty_delta_kn": 10,                   # gust - mean above this = gusty
    "shore_normal_deg": 300,                # direction the coast faces (approx.)

    # Personal kite quiver: which of your own kites fits the current wind.
    # Wind ranges are manufacturer charts at a 75 kg reference rider, shifted
    # to "rider"/weight_kg below. The shift moves the range's CENTRE by
    # weight/75 (per the standard size formula size_m2 = weight_kg*2.2/wind_kn,
    # so for a fixed size the usable wind scales with rider weight) but keeps
    # its WIDTH: how wide a window a kite has is a depower/skill property, not
    # something that stretches because the rider is heavier. Scaling both ends
    # instead used to hand out a 7 m good to 37 kn.
    # No manufacturer chart could be found for the 2014 Drifter 7m, so its
    # range is estimated from that same formula instead of a real chart -
    # treat it as the roughest of the three and adjust once you've flown it.
    # Every range top is then capped at skill_ceiling_kn: what a kite can
    # technically hold and what you want to be out in are different numbers.
    "rider": {"weight_kg": 88, "height_cm": 192, "board": "North Prime 141x41",
              "skill_ceiling_kn": 28},
    "quiver": [
        {"name": "Drifter", "brand": "Cabrinha", "size_m2": 7, "color": "Orange",
         "year": 2014, "image": "Drifter 7.png", "wind_range_75kg": None},
        {"name": "Switchblade", "brand": "Cabrinha", "size_m2": 10, "color": "Green",
         "year": 2014, "image": "Switchblade 10.jpg", "wind_range_75kg": (11, 25)},
        {"name": "Bandit", "brand": "F-One", "size_m2": 14, "color": "Blue",
         "year": 2019, "image": "Bandit 14.jpg", "wind_range_75kg": (8, 20)},
    ],
    "kite_images_dir": "Kites",              # relative to this script
    "kite_thumb_px": 112,                    # photos are shown at 56 px; embed at 2x for
                                             # retina and no more. The full-size originals
                                             # were 3.4 MB of a 4.6 MB output file.
    "kite_est_range_frac": 0.25,             # +/- this around the formula centre when no
                                             # manufacturer chart exists (the Drifter)
    "kite_marginal_frac": 0.15,              # how far outside its range a kite is still
                                             # "marginal" rather than "off"

    # Satellite calibration
    "cdse_token_url": "https://identity.dataspace.copernicus.eu/auth/realms/"
                      "CDSE/protocol/openid-connect/token",
    "cdse_catalog_url": "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search",
    "cdse_process_url": "https://sh.dataspace.copernicus.eu/api/v1/process",
    "sat_days_back": 120,                   # only recent images (bed changes)
    "sat_max_cloud": 30,
    "sat_max_scenes": 12,
    "ndwi_wet": 0.10,                       # NDWI above = water
    "ndwi_dry": -0.05,                      # NDWI below = dry
    "calib_zone_m": 60,                     # also calibrate this far around lagoon

    # Sentinel-1 SAR (radar): not blocked by clouds, so far more usable images than
    # optical (~120 vs ~20 clear days/year here) - the current record is 44 SAR to
    # 20 optical. Calm water is dark (low backscatter): measured on this lagoon at
    # -20.6 dB median vs -13.1 dB over dry dune land. Those are medians, not clean
    # classes - a single dry dune patch was seen swinging -11 to -22 dB across dates,
    # overlapping the water range entirely (see flats_dry_override), which is why
    # nothing here trusts one scene on its own.
    # Caveat: wind-roughened water gets brighter and can be missed - so classification
    # is deliberately conservative (a buffer zone counts as neither wet nor dry) and
    # cross-checked against optical observations rather than trusted blindly.
    "sar_days_back": 365,
    "sar_max_scenes": 120,
    "sar_orbit": "descending",              # fixed orbit direction for consistent geometry
    "sar_water_db": -17.0,                  # VV backscatter midpoint between land/lagoon
    "sar_margin_db": 2.5,                   # buffer: only count pixels clearly past this
    "sar_max_wind_kn": 20,                  # skip a SAR scene if gust exceeded this
                                             # shortly before acquisition. Rationale: on an
                                             # early ungated sample, near-zero-area SAR
                                             # readings all coincided with 19-47 kn gusts
                                             # while genuinely-wet ones stayed under ~29 kn
                                             # - rough water read as dry rather than
                                             # actually disappearing. Note this can no
                                             # longer be re-checked from the cached record,
                                             # because the gate now removes exactly those
                                             # scenes; to re-test it, raise this and refit.
    "sar_wind_check_h": 3,                  # window before acquisition to check gust in

    # Lagoon water-level response, fitted from a year of satellite images
    "response_days_back": 365,              # history span to search for tide-phase spread
    "response_max_scenes": 60,              # cap on how many images to process
    "response_min_scenes": 6,               # need at least this many usable images to fit
    "response_delays_min": list(range(0, 181, 10)),  # candidate lag search grid
    "tidal_r_min": 0.40,                    # |correlation| above this = treat as tidal
    # If it isn't tidal, check whether wind explains it instead - the idea being wave
    # overtopping across the low, actively-eroding sand berm. Only gust magnitude is
    # tested, not direction (onshore-component correlated worse than plain magnitude).
    # Status: this does NOT currently hold. At n=20 it looked strong (12h trailing max
    # gust, r=0.60, p=0.005) and was described here as validated; at n=64 the same
    # correlation is r=0.11. The test is kept so a future record can settle it, but
    # nothing downstream should present it as a known mechanism.
    "wind_windows_h": [12, 24, 48],         # candidate trailing windows to test
    "wind_r_min": 0.35,                     # |correlation| above this = use the wind fit

    # Lake mode (the current mode): area is the median of recent observations, not the
    # single newest one. The record contains a 0.02 ha reading among 64 - a detection
    # failure, not a drained lagoon - and holding whichever scene happens to be last
    # would let one bad image claim the lagoon is gone.
    "lake_recent_days": 45,                 # prefer observations this recent
    "lake_min_obs": 3,                      # widen the window until at least this many
    "lake_min_plausible_ha": 0.3,           # below this, treat as failed detection
    "dem_max_age_days": 180,                # warn when the cached AHN tile is older
    "flats_dry_cache_keep_days": 7,         # prune older per-day dry-override caches
    "flats_dry_days_back": 30,              # how recent an image must be for the
                                             # satellite-priority dry override on the flats
    "flats_dry_max_scenes": 8,              # how many recent images to check
    "flats_dry_min_looks": 3,               # need this many independent dry looks
                                             # (with zero wet looks) before overriding -
                                             # single-scene SAR is too noisy to trust alone

    # Wetsuit call from sea-surface temperature (Open-Meteo marine API, free, no key).
    # Highest matching threshold wins, so keep this sorted coldest-first. Tune the
    # numbers from experience - only you know when your hands stop working.
    "wetsuit": [
        (-99, 12.0, "4/3 + boots + gloves"),
        (12.0, 19.0, "4/3, bare hands and feet"),
        (19.0, 99.0, "shorty"),
    ],
    # A cold, hard wind makes the water feel colder than its temperature between
    # runs: bump one step warmer-dressed when air is this far below water and it's
    # blowing at least this hard.
    "wetsuit_chill_air_delta_c": 4.0,
    "wetsuit_chill_wind_kn": 18.0,

    "timezone": "Europe/Amsterdam",
    "cache_dir": "cache",
}

# Project root (where cache/, Kites/, lagoon.geojson, and the output HTML live) -
# two levels up from this file: zandmotor/config.py -> zandmotor/ -> project root.
HERE = pathlib.Path(__file__).resolve().parent.parent
CACHE = HERE / CFG["cache_dir"]
TZ = ZoneInfo(CFG["timezone"])
UTC = dt.timezone.utc
log = logging.getLogger("zandmotor")
