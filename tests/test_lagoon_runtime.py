import datetime as dt

import numpy as np
import pytest
from scipy import ndimage

from zandmotor.config import CFG, UTC
from zandmotor.flood import compute_wet_masks
from zandmotor.geometry import make_grid
from zandmotor.lagoon_model.runtime import (
    LagoonWater, describe_observation, latest_observation, possible_fill, wet_mask_from_area,
)
from zandmotor.lagoon_model.types import LakeResponse, TidalResponse

NOW = dt.datetime(2026, 10, 6, 20, tzinfo=UTC)
DAY = 86400.0


def _lake(obs=None, times=(), areas=()):
    return LakeResponse(r=0.1, wind_r=0.1, n=len(times) or len(obs or []),
                        fitted=NOW.isoformat(), times=list(times), areas=list(areas),
                        n_optical=0, n_sar=0, obs=obs or [])


def test_latest_observation_is_newest_plausible_and_not_in_the_future():
    te = NOW.timestamp()
    obs = [{"t": te - 3 * DAY, "area_ha": 3.4, "sensor": "optical", "track": "S2"},
           {"t": te - 1 * DAY, "area_ha": 0.02, "sensor": "sar", "track": "S1 orbit 110"},
           {"t": te - 2 * DAY, "area_ha": 4.1, "sensor": "sar", "track": "S1 orbit 37",
            "track_mae_ha": 0.4},
           {"t": te + 1 * DAY, "area_ha": 9.0, "sensor": "optical", "track": "S2"}]
    o = latest_observation(_lake(obs), NOW)
    # the 0.02 ha reading is a detection failure, the future one isn't seen yet
    assert o["area_ha"] == 4.1
    assert describe_observation(o) == "radar, orbit 37"


def test_latest_observation_falls_back_to_old_cache_columns():
    te = NOW.timestamp()
    o = latest_observation(_lake(times=[te - 2 * DAY, te - DAY], areas=[2.0, 3.0]), NOW)
    assert o["area_ha"] == 3.0 and o["sensor"] is None


def test_possible_fill_flags_high_water_or_heavy_rain_only_after_the_image():
    te = NOW.timestamp()
    t = np.arange(te - 10 * DAY, te + DAY, 3600.0)
    sea = np.full_like(t, 0.9)
    sea[t < te - 5 * DAY] = 2.0             # a surge BEFORE the image: already seen
    rain = np.zeros_like(t)
    since = te - 4 * DAY
    calm = possible_fill(since, te, (t, sea), (t, rain))
    assert not calm["flagged"] and calm["max_sea_m"] == pytest.approx(0.9)
    sea2 = sea.copy()
    sea2[(t > since) & (t < te)] = CFG["fill_flag_sea_m"] + 0.1
    assert possible_fill(since, te, (t, sea2), (t, rain))["flagged"]
    rain2 = rain.copy()
    rain2[(t > since) & (t <= te)] = 1.0    # ~96 mm over four days
    assert possible_fill(since, te, (t, sea), (t, rain2))["flagged"]
    # rain forecast for tomorrow doesn't count
    rain3 = np.where(t > te, 50.0, 0.0)
    assert not possible_fill(since, te, (t, sea), (t, rain3))["flagged"]
    assert not possible_fill(since, te, None, None)["flagged"]


@pytest.fixture(scope="module")
def grid():
    return make_grid((4.19, 52.05, 4.195, 52.053), res_m=5.0)


def test_lagoon_water_lake_mode_is_constant_and_counts_only_the_lagoon(grid):
    lagoon = np.zeros((grid.height, grid.width), bool)
    lagoon[10:40, 10:40] = True             # 900 cells = 2.25 ha
    freq = np.zeros(lagoon.shape)
    freq[10:40, 10:30] = 0.9                # the often-wet part fills first
    lw = LagoonWater(lagoon, grid, _lake(times=[0.0], areas=[1.5]), freq, lake_area=1.5)
    assert lw.constant
    wet, ha, patch = lw.at(0.0)
    assert ha == pytest.approx(1.5)
    assert patch == pytest.approx(1.5)
    assert not (wet & ~lagoon).any()
    assert wet[10:40, 10:30].all()          # the often-wet cells, not the rest


def test_lagoon_water_without_model_is_the_whole_outline(grid):
    lagoon = np.zeros((grid.height, grid.width), bool)
    lagoon[5:25, 5:25] = True
    lw = LagoonWater(lagoon, grid)
    assert lw.constant
    assert lw.at(123.0)[1] == pytest.approx(400 * 25 / 1e4)


def test_lagoon_water_tidal_mode_varies_per_instant(grid):
    lagoon = np.zeros((grid.height, grid.width), bool)
    lagoon[10:40, 10:40] = True
    freq = np.linspace(0, 1, lagoon.size).reshape(lagoon.shape)
    model = TidalResponse(delay_min=0, damping=1.0, offset=1.0, floor_ha=0.5, r=0.6, n=10,
                          rws_location="x", fitted=NOW.isoformat(), times=[], areas=[],
                          n_optical=0, n_sar=0)
    lw = LagoonWater(lagoon, grid, model, freq, sea_level_fn=lambda te: (te, "test"))
    assert not lw.constant
    assert lw.at(0.0)[1] == pytest.approx(1.0)
    assert lw.at(0.5)[1] == pytest.approx(1.5)


def test_wet_mask_from_area_fills_most_often_wet_first():
    lagoon = np.ones((4, 4), bool)
    freq = np.zeros((4, 4))
    freq[0, :] = 1.0
    m = wet_mask_from_area(freq, lagoon, target_ha=4 * 25 / 1e4, cell_area_m2=25)
    assert m[0].all() and m.sum() == 4


def test_wet_beach_is_drawn_but_never_part_of_the_lagoon(grid):
    h, w = grid.height, grid.width
    z = np.full((h, w), 5.0, "float32")
    z[:, :8] = -3.0                          # open sea on the west edge
    z[:, 8:14] = 0.2                         # a low beach the tide covers
    lagoon = np.zeros((h, w), bool)
    lagoon[20:40, 30:50] = True
    sea = np.zeros((h, w), bool)
    sea[:, :8] = True
    F = np.where(np.arange(w)[None, :] < 14, -3.0, 5.0) * np.ones((h, 1))
    coastal = np.ones((h, w), bool)
    beach, uncounted = compute_wet_masks(z, F, 1.0, coastal, lagoon, sea, None, grid)
    assert beach[:, 8:14].all()              # the beach floods at +1.0 m
    assert not (beach & lagoon).any() and not (beach & sea).any()
    assert uncounted[:, :8].all()            # the sea is drawn, not counted


def test_wet_mask_from_area_grows_one_body_from_the_core():
    # a far corner is wetter than the water joining it, but it may only fill
    # once the body has grown to it - never as a separate patch
    lagoon = np.ones((5, 9), bool)
    freq = np.full((5, 9), 0.5)
    freq[2, 1:4] = 1.0                        # the core
    freq[0, 8] = 0.9                          # isolated, often-wet corner
    m = wet_mask_from_area(freq, lagoon, target_ha=6 * 25 / 1e4, cell_area_m2=25)
    assert m[2, 1:4].all() and not m[0, 8]
    _, n = ndimage.label(m)
    assert n == 1 and m.sum() == 6


def test_wet_mask_from_area_never_exceeds_the_outline():
    lagoon = np.zeros((4, 4), bool)
    lagoon[1:3, 1:3] = True
    m = wet_mask_from_area(np.ones((4, 4)), lagoon, target_ha=1.0, cell_area_m2=25)
    assert (m == lagoon).all()
