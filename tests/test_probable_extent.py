import numpy as np
import pytest

from zandmotor.cli import _demote_weak_wind
from zandmotor.config import CFG
from zandmotor.flood import probable_extent
from zandmotor.lagoon_model.types import WindResponse
from zandmotor.terrain import prepare_terrain


def _masks(shape=(30, 30)):
    lagoon = np.zeros(shape, bool)
    lagoon[10:20, 10:20] = True
    basin = np.zeros(shape, bool)
    basin[5:25, 5:25] = True
    z = np.zeros(shape, bool)
    return lagoon, basin, z.copy(), z.copy(), z.copy()


def test_probable_extent_takes_basin_water_joined_to_the_lagoon():
    lagoon, basin, beach, uncounted, sea = _masks()
    beach[10:20, 20:24] = True                # flooded basin floor next to the lagoon
    beach[10:20, 26:29] = True                # flooded beach outside the basin
    uncounted[6:8, 6:8] = True                # a pond in the basin, not joined
    m = probable_extent(lagoon, beach, uncounted, sea, basin)
    assert m[lagoon].all() and m[10:20, 20:24].all()
    assert not m[:, 25:].any() and not m[6:8, 6:8].any()


def test_probable_extent_is_one_body_without_holes():
    lagoon, basin, beach, uncounted, sea = _masks()
    beach[8:22, 8:22] = True
    beach[14, 21] = False                     # a dry speck in the flooded margin
    beach[9, 8:22] = False                    # a one-cell dry sliver
    lagoon_wet = lagoon.copy()
    lagoon_wet[14, 14] = False                # the lagoon itself has a hole
    m = probable_extent(lagoon_wet, beach, uncounted, sea, basin)
    from scipy import ndimage
    _, n = ndimage.label(m)
    assert n == 1 and m[8:22, 8:22].all()


def test_probable_extent_never_includes_the_open_sea():
    lagoon, basin, beach, uncounted, sea = _masks()
    uncounted[10:20, 20:24] = True
    sea[10:20, 20:24] = True
    m = probable_extent(lagoon, beach, uncounted, sea, basin)
    assert not m[10:20, 21:24].any()


def test_prepare_terrain_fills_basin_gaps_at_basin_floor_level():
    z = np.full((20, 20), 2.0)
    z[8:12, 8:12] = np.nan                    # under the lagoon
    z[8:12, 13:15] = np.nan                   # basin, outside the outline
    z[2:4, 2:4] = np.nan                      # a gap away from the lagoon
    lagoon = np.zeros(z.shape, bool)
    lagoon[7:13, 7:13] = True
    basin = np.zeros(z.shape, bool)
    basin[6:14, 6:16] = True
    out = prepare_terrain(z, lagoon, basin)
    assert (out[8:12, 8:12] == CFG["sea_fill_level"]).all()
    assert (out[8:12, 13:15] == CFG["basin_gap_fill_m"]).all()
    assert (out[2:4, 2:4] == 20.0).all()


def _wind(r, n):
    return WindResponse(window_h=48, slope=0.04, offset=2.0, area_min_ha=2.1, area_max_ha=4.5,
                        r=r, sea_r=0.17, n=n, fitted="2026-11-16T09:00:00+00:00",
                        times=[1.0], areas=[4.3], n_optical=n, n_sar=0)


def test_weak_cached_wind_fit_is_demoted_to_lake():
    lake = _demote_weak_wind(_wind(0.36, 28))     # the 2026-11-16 fit
    assert lake.mode == "lake" and lake.wind_r == 0.36 and lake.r == 0.17
    assert lake.areas == [4.3]
    strong = _wind(0.7, 80)
    assert _demote_weak_wind(strong) is strong


DAY = 86400.0


def _fill_drain(level_at, basin_water, obs_ha=0.25, rest_ha=0.25, days=3):
    """A FillDrain from t=0 (an image of the 10 x 10 cell lagoon) whose flood
    floods `basin_water` above +1.5 m."""
    from zandmotor.flood import FillDrain
    lagoon, basin, _, uncounted, sea = _masks(basin_water.shape)

    def masks(lvl):
        return (basin_water if lvl >= 1.5 else np.zeros_like(lagoon)), uncounted
    return FillDrain(0.0, obs_ha, rest_ha, days * DAY, lambda t: (level_at(t), "measured"),
                     masks, sea, basin, lagoon, 25.0), lagoon, basin


def _surge(*hours):
    return lambda t: 1.8 if any(h * 3600 <= t < (h + 1) * 3600 for h in hours) else 0.5


def test_fill_drain_fills_at_the_surge_and_drains_back_to_rest(monkeypatch):
    monkeypatch.setitem(CFG, "lagoon_drain_days", 1.0)
    _, _, beach, _, _ = _masks()
    beach[10:20, 20:24] = True                # the surge floods 40 cells (0.1 ha) of basin
    fd, lagoon, basin = _fill_drain(_surge(3), beach)
    assert fd.at(2 * 3600)[0] == 0.25 and fd.at(2 * 3600)[2] is None
    area, region, event = fd.at(3 * 3600)
    assert area == pytest.approx(0.35) and event == (1.8, 3 * 3600, 3 * 3600)
    assert region[10:20, 20:24].all() and region[lagoon].all()
    # the surge keeps it full until its last sample (3:45); one drain time
    # constant after that, the excess is down to 1/e
    assert fd.at(3.75 * 3600 + DAY)[0] == pytest.approx(0.25 + 0.1 / np.e, abs=1e-3)
    assert fd.varies


def test_fill_drain_refills_on_the_next_surge(monkeypatch):
    monkeypatch.setitem(CFG, "lagoon_drain_days", 1.0)
    _, _, beach, _, _ = _masks()
    beach[10:20, 20:24] = True
    fd, *_ = _fill_drain(_surge(3, 3 + 24), beach)
    assert fd.at(26 * 3600)[0] < 0.30
    area, _, event = fd.at(27 * 3600)
    assert area == pytest.approx(0.35) and event[1] == 27 * 3600


def test_fill_drain_labels_the_surge_by_its_peak():
    _, _, beach, _, _ = _masks()
    beach[10:20, 20:24] = True
    rising = {3: 1.6, 4: 1.9, 5: 1.7}          # one surge, peaking at 4:00
    fd, *_ = _fill_drain(lambda t: rising.get(int(t // 3600), 0.5), beach)
    assert fd.at(5.75 * 3600)[2] == (1.9, 4 * 3600, 3 * 3600)   # peak, and when it began


def test_fill_drain_drains_a_fill_the_image_caught(monkeypatch):
    # no surge since, but the image itself is above the resting size
    monkeypatch.setitem(CFG, "lagoon_drain_days", 1.0)
    _, _, beach, _, _ = _masks()
    fd, *_ = _fill_drain(lambda t: 0.5, beach, obs_ha=0.35, rest_ha=0.25)
    assert fd.at(DAY)[0] == pytest.approx(0.25 + 0.1 / np.e, abs=1e-3)
    # and never drains below the image itself
    still, *_ = _fill_drain(lambda t: 0.5, beach, obs_ha=0.2, rest_ha=0.25)
    assert still.at(DAY)[0] == 0.2 and not still.varies


def test_ponds_in_the_basin_are_not_a_surge_fill():
    # a depression joined to the lagoon holds water at any tide; it must not
    # count as a fill, or the lagoon could never drain below it
    _, _, _, uncounted, _ = _masks()
    uncounted[10:20, 20:24] = True
    from zandmotor.flood import FillDrain
    lagoon, basin, beach, _, sea = _masks()
    fd = FillDrain(0.0, 0.35, 0.25, DAY, lambda t: (-0.5, "measured"),
                   lambda lvl: (beach, uncounted), sea, basin, lagoon, 25.0, drain_days=1.0)
    assert fd.at(DAY)[0] == pytest.approx(0.25 + 0.1 / np.e, abs=1e-3)
    assert fd.at(DAY)[2] is None


def test_fill_drain_counts_only_the_basin():
    _, _, beach, _, _ = _masks()
    beach[10:20, 20:29] = True                # floods past the basin edge (col 25)
    fd, *_ = _fill_drain(lambda t: 1.8, beach)
    region = fd.at(3600)[1]
    assert region[10:20, 20:25].all() and not region[:, 25:].any()


def test_lagoon_water_follows_fill_and_drain(monkeypatch):
    from zandmotor.geometry import make_grid
    from zandmotor.lagoon_model.runtime import LagoonWater
    from zandmotor.lagoon_model.types import LakeResponse
    monkeypatch.setitem(CFG, "lagoon_drain_days", 1.0)
    grid = make_grid((4.18, 52.05, 4.1822, 52.0514), 5.0)     # ~30 x 30 cells; the
    # masks below fit any grid of at least 25 x 25
    _, _, beach, _, _ = _masks((grid.height, grid.width))
    beach[10:20, 20:24] = True
    lagoon, basin, *_ = _masks(beach.shape)
    lake = LakeResponse(r=0.1, wind_r=0.1, n=1, fitted="2026-11-16T00:00:00+00:00",
                        times=[0.0], areas=[0.25], n_optical=1, n_sar=0)
    lw = LagoonWater(lagoon, grid, lake, lagoon.astype(float), lake_area=0.25, basin=basin,
                     fill_drain=lambda base: _fill_drain(_surge(3), beach)[0])
    assert not lw.constant
    assert lw.at(3600.0)[1] == 0.25 and lw.state(3600.0)[1] is None
    assert lw.at(3 * 3600.0)[1] == pytest.approx(0.35) and lw.state(3 * 3600.0)[1][0] == 1.8
    later = lw.at(3 * 3600.0 + DAY)
    assert 0.25 < later[1] < 0.35 and later[0][lagoon].all()   # drains from the edges


def test_basin_takes_in_ahn_gaps_that_touch_it(monkeypatch):
    from zandmotor import geometry
    monkeypatch.setattr(geometry, "load_basin_polygon", lambda: (None, None))
    monkeypatch.setitem(CFG, "basin_margin_m", 0.0)
    z = np.full((20, 20), 1.3)
    z[5:8, 9:12] = np.nan                     # gap touching the lagoon: water when surveyed
    z[15:18, 15:18] = np.nan                  # gap away from it
    z[0:3, 0:3] = np.nan                      # gap on the map edge: the sea
    lagoon = np.zeros(z.shape, bool)
    lagoon[8:12, 8:12] = True
    basin, found = geometry.basin_mask(None, lagoon, z)
    assert not found and basin[5:8, 9:12].all()
    assert not basin[15:18, 15:18].any() and not basin[0:3, 0:3].any()


def test_calibration_measures_basin_water_joined_to_the_lagoon():
    from zandmotor.lagoon_model.calibration import _joined
    lag = np.zeros((10, 10), bool)
    lag[3:6, 3:6] = True
    wet = np.zeros((10, 10), bool)
    wet[3:6, 3:8] = True                      # lagoon plus an arm beyond the outline
    wet[8:10, 8:10] = True                    # a separate puddle
    m = _joined(wet, lag)
    assert m[3:6, 3:8].all() and not m[8:10, 8:10].any()


def test_fingerprint_changes_with_the_basin():
    from zandmotor.lagoon_model.runtime import wetness_fingerprint
    lagoon, basin, *_ = _masks()
    ring = [(4.19, 52.05), (4.2, 52.05), (4.2, 52.06), (4.19, 52.05)]
    bigger = basin.copy()
    bigger[2, 2] = True
    assert wetness_fingerprint(lagoon, ring, basin) != wetness_fingerprint(lagoon, ring, bigger)
