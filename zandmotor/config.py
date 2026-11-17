"""Configuration: everything you might want to tweak is here.

The rider, the quiver and the wetsuit table below are only defaults: each
user's own values come from settings.toml (see zandmotor.settings), so other
people can install this without editing code."""

from __future__ import annotations

import datetime as dt
import logging
import os
import pathlib
from zoneinfo import ZoneInfo

import platformdirs

CFG = {
    # Map area (lon_min, lat_min, lon_max, lat_max), covers the whole Zandmotor
    "bbox_wgs84": (4.165, 52.035, 4.225, 52.070),
    "grid_res_m": 5.0,                      # ground resolution of the model

    # Optional (lon, lat) point inside the lagoon, to help --outline pick it
    "lagoon_hint": (4.1938, 52.0565),
    "outline_days_back": 60,               # search this far back for an image
    "outline_min_ha": 2.0,                 # ignore water bodies smaller than this
    "outline_osm_url": "https://overpass-api.de/api/interpreter",
    # Last-resort placeholder outline (lon, lat), only used if nothing else works
    "lagoon_polygon": [(4.186, 52.055), (4.204, 52.055), (4.206, 52.064),
                       (4.190, 52.066), (4.186, 52.055)],

    # Filenames (relative to DATA unless marked; see the layouts at the bottom)
    "lagoon_geojson_file": "lagoon.geojson",
    "lagoon_basin_file": "lagoon_basin.geojson",   # the lagoon's largest extent seen
    "outline_check_image_file": "lagoon_outline_check.png",
    "ahn_dtm_file": "ahn_dtm.tif",           # relative to CACHE
    "flats_dry_cache_glob": "flats_dry_*.npy",     # relative to CACHE
    "flats_dry_cache_pattern": "flats_dry_{date}.npy",  # relative to CACHE
    "flats_dry_shared_file": "shared_dry_override.npy",  # relative to CACHE: the newest
                                             # mask, carried by the shared model for copies
                                             # without a Copernicus account
    "lagoon_response_file": "lagoon_response.json",     # relative to CACHE: the lagoon
    "wetness_frequency_file": "wetness_frequency.npy",  # model --calibrate fits, which
    "wetness_confidence_file": "wetness_confidence.npy",  # the shared model snapshot
    "wetness_meta_file": "wetness_meta.json",           # carries (see model_bundle)

    # Terrain
    "ahn_wcs": "https://service.pdok.nl/rws/ahn/wcs/v1_0",
    "ahn_coverage": "dtm_05m",
    "ahn_request_res_m": 2.0,
    "local_dem": None,                      # path to your own GeoTIFF (any CRS, m NAP)
    # A newer survey laid over AHN wherever it has data (relative to the project root;
    # None to switch off). The Rijkswaterstaat 2026 coastal DTM (2 m) measures 80% of
    # the basin cells AHN has no return for - the basin floor at 0.8-1.8 m NAP, where
    # the AHN-only model had to assume one level - and agrees with AHN to ~0.1 m in
    # the basin where both measured. Heights only: the outline trim and the basin's
    # "no return = water when surveyed" evidence stay on AHN (see cli).
    "coastal_dem": "cache/rws_kust_2026_dtm_2m.tif",
    "coastal_dem_label": "Rijkswaterstaat coastal DTM 2026",
    "sea_fill_level": -6.0,                 # m NAP for missing data offshore, and a
                                             # cosmetic fill for the lagoon interior (its
                                             # bed elevation isn't modelled - see the
                                             # module docstring on why depth doesn't matter)
    "sea_seed_level": -1.0,                 # border cells below this = open sea
    "lagoon_trim_land_m": 3.0,              # AHN elevation above this inside the traced
                                             # outline is dune, not lagoon (a trace can
                                             # overshoot onto the dunes). Dune height, NOT
                                             # beach height: this used to be 0.5 m NAP, and
                                             # it cut 5.9 of a 13.4 ha outline that was open
                                             # water on the 2026-09-01 image. The lagoon is
                                             # perched - AHN puts its margins at 1.1-1.5 m
                                             # NAP - so a sea-level-ish cut removes real
                                             # lagoon and capped every measured area at the
                                             # 7.45 ha that was left.
    "coastal_zone_m": 500,                  # only render flooding this far inland of the
                                             # open sea (plus the lagoon itself). Real towns
                                             # nearby sit below sea level behind dikes and
                                             # pumped drainage an elevation-only flood model
                                             # has no way to know about, and would otherwise
                                             # get rendered as "flooded"
    # The lagoon basin: the largest extent the lagoon has been traced at. Every
    # --outline merges into it, so it only grows (a trace from a quiet day shrank
    # the outline from 13.4 to 4.5 ha on 2026-11-16, and the basin keeps the rest).
    # The probable water edge reaches a bit beyond it (probable_reach_m); the surge
    # fill and the satellite measurements are bounded by it - see flood.py.
    "basin_margin_m": 10.0,                 # widen the basin by this for tracing error
    "basin_gap_fill_m": 1.3,                # m NAP for AHN gaps in the basin outside the
                                             # outline. AHN gets no return off water, so the
                                             # gaps are where the lagoon stood when surveyed;
                                             # left as "land" (20 m) they walled the tide out
                                             # of the basin. Where AHN did measure, the basin
                                             # floor sits at 1.15-1.53 m NAP (5-95 %, median
                                             # 1.29). 1.3 also fitted the record best (see
                                             # flood.FillDrain); 1.2 scored 0.97 ha, 1.4 1.11.
    "probable_smooth_px": 2,                # closing radius that joins the probable water
                                             # edge into one body across thin dry slivers
    "probable_min_extra_ha": 0.1,           # only draw the probable edge when it adds this
                                             # much to the lagoon (else it traces the lagoon)
    "probable_reach_m": 300,                # the drawn edge follows flooded water joined to
                                             # the lagoon this far beyond the basin. On
                                             # 2026-11-12 (+1.81 m NAP) lagoon and sea were one
                                             # sheet, and the open sea is 150-280 m from the
                                             # basin; unlimited, the edge ran along the whole
                                             # 4 km beach (60-76 ha) from +1.6 m.
    "surge_fill_step_min": 15,              # time step of the fill-and-drain replay
    "lagoon_drain_days": 1.0,               # a fill's excess over the resting size decays
                                             # with this time constant (half-life ~17 h).
                                             # Default until --calibrate fits it (cache/
                                             # fill_drain.json). On the 2025-26 record:
                                             # 0.5-1.5 d all within 0.1 ha; 3 d +0.5 ha, 7 d
                                             # +1.3 ha. Much faster than the "one to three
                                             # weeks" once inferred from sparse images.
    "lagoon_rest_days": 60,                 # resting size = median image over this span
    # --calibrate refits drain time and basin floor by replaying the record image to
    # image (lagoon_model.fill_drain_fit); the two values above are the defaults until
    # then. Every combination of these is tried:
    # flood level = sea + wave_k x wave term + rain_k x rain - offset (flood.
    # effective_level); the defaults are still water at the DEM's level until
    # --calibrate fits them
    "lagoon_wave_k": 0.0,
    "lagoon_flood_offset_m": 0.0,
    "lagoon_wave_form": "height",           # wind.wave_forcing
    "lagoon_rain_k": 0.0,
    "fill_rain_window_h": 48,               # rain summed over this window before each step
    "fill_fit_drain_days": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
    "fill_fit_gap_fill_m": [1.2, 1.4],      # matters little since the 2026 coastal DTM
                                             # measures most of the basin floor
    "fill_fit_wave_forms": {                # candidate wave terms (form: weights)
        "height": [0.15, 0.3, 0.5],
        "height_dir": [0.2, 0.4, 0.6, 0.8],
        "runup": [0.02, 0.04, 0.06, 0.08],
        "runup_dir": [0.03, 0.05, 0.08, 0.11],
    },
    "fill_fit_rain_k": [0.0, 0.005, 0.01, 0.02],   # m of flood level per mm in the window
    "fill_fit_offset_m": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "fill_fit_tolerance_ha": 0.05,          # prefer the simpler within this of the best
    "fill_fit_levels_m": (0.6, 2.4, 0.02),  # effective levels (lo, hi, step) the flood is
                                             # tested at; the 2026 DTM floods the basin a
                                             # little from +0.7 m (the north-west channel)
    "fill_fit_full_view": 0.9,              # an image seeing less of the basin is a lower bound
    "fill_drain_file": "fill_drain.json",   # relative to CACHE
    "trend_days_back": 30,                  # the size-over-time chart starts this long ago
    "trend_days_ahead": 7,                  # and runs this far into the forecast

    # Water level (Rijkswaterstaat)
    "rws_url": "https://ddapi20-waterwebservices.rijkswaterstaat.nl/"
               "ONLINEWAARNEMINGENSERVICES/OphalenWaarnemingen",
    "rws_locations": ["scheveningen", "hoekvanholland"],
    "surge_decay_hours": 12.0,              # for astro + surge fallback
    "rws_request_deadline_s": 120,          # whole-response limit per RWS request (a
                                             # 60-day chunk normally takes ~3 s)

    # Wind and sea temperature (Open-Meteo, free, no key)
    "open_meteo_forecast_url": "https://api.open-meteo.com/v1/forecast",
    "open_meteo_marine_url": "https://marine-api.open-meteo.com/v1/marine",
    "open_meteo_archive_url": "https://archive-api.open-meteo.com/v1/archive",

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
    # lagoon ONLY. The wet beach and the open sea are drawn but never counted:
    # the beach term swung 0-21 ha with every tide and, merged with the lagoon
    # at high water, tripled the "biggest patch" the verdict tests.
    "min_rideable_ha": 3.0,                 # need at least this much lagoon water to ride
    "gusty_delta_kn": 14,                   # gust - mean above this = gusty
    "shore_normal_deg": 300,                # direction the coast faces (approx.)

    # Personal kite quiver: which of your own kites fits the current wind.
    # rider, quiver and the wetsuit table are DEFAULTS only - an example rider
    # with two example kites. Each user's own values come from settings.toml
    # (zandmotor.settings, written by `zandmotor --setup`), which also keeps the
    # notes on individual kites.
    # Wind ranges are manufacturer charts at a 75 kg reference rider, shifted
    # to "rider"/weight_kg below. The shift moves the range's CENTRE by
    # weight/75 (per the standard size formula size_m2 = weight_kg*2.2/wind_kn,
    # so for a fixed size the usable wind scales with rider weight) but keeps
    # its WIDTH: how wide a window a kite has is a depower/skill property, not
    # something that stretches because the rider is heavier. Scaling both ends
    # instead used to hand out a 7 m good to 37 kn.
    # A kite without a manufacturer chart (wind_range_75kg None) gets a range
    # estimated from that same formula - the roughest of the lot.
    # Every range top is then capped at skill_ceiling_kn: what a kite can
    # technically hold and what you want to be out in are different numbers.
    # exempt_from_ceiling opts a storm kite out of that cap. With a 28 kn ceiling
    # a 4.5 m storm kite that only makes sense above 28 kn was left a 28-28 kn
    # range that was never picked. Hours past the ceiling where only it fits come
    # out as "maybe", not "go"; below the ceiling a storm-kite hour is an ordinary
    # "go".
    # min_wind_kn is a hard floor on the MEAN wind, set for the rider directly:
    # the kite is only considered above it, and neither the weight shift nor the
    # "marginal" buffer moves it. Without it a 28-40 kn storm-kite chart shifted
    # to 33.9-45.9 kn for 88 kg, so it was ignored at 28-29 kn and only "marginal"
    # up to 33.9 kn.
    "rider": {"weight_kg": 75, "height_cm": None, "board": "", "skill_ceiling_kn": 35},
    "quiver": [
        {"name": "Example kite", "brand": "", "size_m2": 9, "color": "", "year": None,
         "image": None, "wind_range_75kg": (14, 26)},
        {"name": "Example kite", "brand": "", "size_m2": 12, "color": "", "year": None,
         "image": None, "wind_range_75kg": (10, 21)},
    ],
    "kite_images_dir": "Kites",              # relative to USER_DIR
    "kite_thumb_px": 112,                    # photos are shown at 56 px; embed at 2x for
                                             # retina and no more. The full-size originals
                                             # were 3.4 MB of a 4.6 MB output file.
    "kite_est_range_frac": 0.25,             # +/- this around the formula centre when no
                                             # manufacturer chart exists (the Drifter 7m)
    "kite_marginal_frac": 0.15,              # how far outside its range a kite is still
                                             # "marginal" rather than "off"

    # Satellite calibration
    "cdse_token_url": "https://identity.dataspace.copernicus.eu/auth/realms/"
                      "CDSE/protocol/openid-connect/token",
    "cdse_catalog_url": "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search",
    "cdse_process_url": "https://sh.dataspace.copernicus.eu/api/v1/process",
    "cdse_request_deadline_s": 60,          # whole-response limit per CDSE request (a
                                             # lagoon-window image normally takes ~1 s)
    "sat_days_back": 120,                   # only recent images (bed changes)
    # Cloud handling. eo:cloud_cover is scored over a whole ~100 km Sentinel-2 tile,
    # so it says little about a 13 ha lagoon: 7 of the 19 images that were clear
    # over the lagoon in 2025-26 sat on tiles with >=30% cloud, among them the
    # 2026-05-14 image that caught the lagoon full. So the catalogue filter is only
    # a coarse pre-filter for the calibration, and what decides is the per-pixel
    # scene classification (SCL) over the lagoon itself.
    "sat_max_cloud": 95,                    # catalogue pre-filter for lagoon calibration
    "sat_max_cloud_full_scene": 30,         # for uses that need the whole map clear
                                             # (outline tracing, the flats dry-override)
    "sat_min_clear_lagoon": 0.90,           # an image only counts if at least this share
                                             # of the lagoon is seen (cloud-free optical,
                                             # finite radar). Unseen pixels used to count
                                             # as dry: 2025-12-25 was recorded at 0.99 ha
                                             # with ~2.5 ha of water under thin cirrus.
                                             # Lowered from 0.95 to 0.90 on request (2026-10),
                                             # to admit more images; up to a tenth of the
                                             # lagoon can now be hidden, and only seen
                                             # pixels count, so an image can read up to
                                             # ~1.3 ha low when cloud sits over water.
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
    # Radar is no longer used for the lagoon record. Over this flat, wet, sandy basin
    # it can't tell water from sand: on 19 passes within a day of a Sentinel-2 image,
    # a wet pixel was darker than a dry one only 46-73% of the time in VV (orbit 37
    # ~0.7, orbit 110 ~0.5 - chance), and no better in VH or VV-VH. On the 2026-05-14
    # fill (13.5 ha) orbit 110 read the lagoon at -18.8 dB and the sand at -18.2.
    # Per-track thresholds, a per-pixel change detection against each pixel's own
    # median, and both combined all fitted to "nothing is wet". Switching it off
    # halves --calibrate's downloads. The flats dry-override still uses radar (a
    # majority vote on whether sand is dry, a different question).
    "sar_in_calibration": False,
    "landsat_collection": "landsat-ot-l1",
    "landsat_scene_window_min": 20,
    "landsat_ndwi_grid": (-0.1, 0.4, 0.025),   # candidate thresholds (start, stop, step)
    "sar_days_back": 365,
    "sar_max_scenes": 200,                  # two descending tracks x two satellites
                                             # (S1C, S1D) give ~120 scene-days a year
    "sar_orbit": "descending",              # fixed orbit direction for consistent geometry
    "sar_water_db": -17.0,                  # VV backscatter midpoint between land/lagoon
    "sar_margin_db": 2.5,                   # buffer: only count pixels clearly past this
                                             # (these two now serve only the flats
                                             # dry-override; the lagoon uses per-track
                                             # thresholds, below)
    # Per-track thresholds for the lagoon. The two descending tracks over the
    # Zandmotor look at it from different angles: relative orbit 37 (~05:49 UTC)
    # sees the lagoon ~3 dB darker than orbit 110 (~05:57 UTC). With the single
    # threshold above, orbit 37 reported a median 4.05 ha and orbit 110 1.79 ha,
    # orbit 110 reading lower in 6 of 6 pairs taken a day apart; and half the
    # lagoon fell in the ambiguous band and was counted as dry. So each track gets
    # its own threshold, chosen to agree pixel by pixel with optical images from
    # the same few days, and then classifies every lagoon pixel.
    "sar_pair_days": 3,                     # SAR and optical this close count as a pair
    "sar_min_pairs": 3,                     # a track needs this many pairs, or it's unused
    "sar_threshold_grid_db": (-26.0, -12.0, 0.25),   # candidate thresholds (start, stop, step)
    "sar_max_track_mae_ha": 1.0,            # drop a track whose median area error against
                                             # its optical pairs is worse than this
    "sar_scene_window_min": 15,             # fetch only this pass (+/- minutes), not the
                                             # whole day's mosaic
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
    "response_max_scenes": 150,             # cap on how many optical images to process
    "calib_max_failed_frac": 0.10,          # abort --calibrate (and keep the old cache) if
                                             # more than this share of image requests fail
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
    "wind_r_min": 0.50,                     # |correlation| above this = use the wind fit
    "wind_min_n": 40,                       # and at least this many images. Was 0.35 with
                                             # no n floor: on 2026-11-16 it picked wind mode
                                             # at r=0.36, n=28, on areas capped by a 4.5 ha
                                             # outline - the same small-sample trap as above.

    # Lake mode (the current mode): the area shown is the NEWEST reliable observation.
    # It used to be the median of the last 45 days, which (a) mixed two radar tracks
    # that disagree by ~2 ha, so it followed whichever track had passed most lately,
    # and (b) smeared the lagoon's real behaviour: it sits at ~2.5-4.5 ha and twice in
    # 2025-26 filled to ~13 ha for one to three weeks. A median spanning a fill shows
    # neither state. Single-image noise is now handled at the source (per-track
    # thresholds, seen-pixel rules) rather than by averaging.
    "lake_min_plausible_ha": 0.3,           # below this, treat as failed detection
    # "May have filled since" flag. Surges are now modelled (flood.FillDrain); what
    # is left is rain: the 2026-05-14 fill (13.5 ha) came with 19 mm and only +1.18 m.
    # Not a model either - one event - so it only warns that the image may be out of
    # date. (Surges that seemed to leave no fill, such as 2025-11-25 at +1.58 m, turned
    # out to have drained before the next image: fills drain in days, not weeks.)
    "fill_flag_sea_m": 1.4,                 # reported by possible_fill; no longer warns
    "fill_flag_rain_mm": 30.0,              # this much rain since the image warns
    "fill_flag_max_days": 21,               # look back at most this far
    "lagoon_obs_stale_days": 14,            # warn when the newest satellite image of the
                                             # lagoon is older than this (run --calibrate)
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
    "cache_dir": "cache",                   # relative to DATA

    # Installed copies (see the layouts below)
    "user_dir_name": "Zandmotor Kite",      # folder in the home folder for settings,
                                             # kite photos and the report: visible in
                                             # Finder, unlike ~/Library
    "settings_file": "settings.toml",       # relative to USER_DIR
    # The shared lagoon model (zandmotor.model_bundle). A new user has no Copernicus
    # account to --calibrate with, so the maintainer's calibrated files are published
    # as the rolling "model" pre-release and installed copies download them.
    "model_manifest_url": "https://github.com/thijskalshoven/Zandmotor/releases/"
                          "download/model/manifest.json",
    "model_bundle_url": "https://github.com/thijskalshoven/Zandmotor/releases/"
                        "download/model/zandmotor-model.zip",
    "model_check_hours": 6,                 # look for a newer snapshot at most this often
    "model_request_deadline_s": 300,        # whole-download limit (~30 MB)
    "model_marker_local": "LOCAL_MODEL",    # in DATA: this copy calibrated its own model,
                                             # so the shared one must not overwrite it
}

# Two layouts. In a git checkout (the maintainer, CI, the tests) everything stays
# where it always was, in the project root. In an installed copy the code sits in
# site-packages, which is no place for a user's files: settings, kite photos and
# the report go to a visible ~/Zandmotor Kite, and the lagoon model and download
# cache to the user cache folder (~/Library/Caches/zandmotor on a Mac).
# ZANDMOTOR_HOME moves USER_DIR in either layout.
def resolve_layout(code_root, env=os.environ, home=None):
    """(is_checkout, user_dir, data) for code living in code_root."""
    code_root = pathlib.Path(code_root)
    is_checkout = (code_root / ".git").exists() and (code_root / "pyproject.toml").exists()
    home = pathlib.Path(home) if home else pathlib.Path.home()
    user_dir = pathlib.Path(env.get("ZANDMOTOR_HOME") or
                            (code_root if is_checkout else home / CFG["user_dir_name"]))
    data = code_root if is_checkout else pathlib.Path(platformdirs.user_cache_dir("zandmotor"))
    return is_checkout, user_dir.expanduser(), data


CODE_ROOT = pathlib.Path(__file__).resolve().parent.parent
IS_CHECKOUT, USER_DIR, DATA = resolve_layout(CODE_ROOT)
CACHE = DATA / CFG["cache_dir"]
TZ = ZoneInfo(CFG["timezone"])
UTC = dt.timezone.utc
log = logging.getLogger("zandmotor")
