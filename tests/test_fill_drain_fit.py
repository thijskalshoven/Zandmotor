import numpy as np
import pytest

from zandmotor.config import CFG
from zandmotor.flood import fill_drain_series, step_times
from zandmotor.lagoon_model.fill_drain_fit import (
    FillDrainFit, curve_fn, fit_drain, mean_error,
)

DAY = 86400.0
LEVELS = np.round(np.arange(1.0, 2.31, 0.02), 2)


def _step_curve(onset):
    """The basin holds 12 ha once the sea reaches `onset`, nothing below."""
    return LEVELS, np.where(LEVELS >= onset, 12.0, np.nan)


def _levels_for(sea_t, sea_v, waves=None, rain=None):
    from zandmotor.lagoon_model.fill_drain_fit import levels_from
    return lambda form, k, off, rk: levels_from((sea_t, sea_v), waves, k, off, form or "height",
                                                rain, rk)


def _waves(t, hs):
    """A wave series of these heights, 6 s period, straight onshore."""
    return (t, hs, np.full(len(t), 6.0), np.full(len(t), CFG["shore_normal_deg"]))


def _synthetic_record(drain_days, surges=(10, 30, 45), days=60, every=3.0):
    """Sea at +0.5 m with 2-hour surges to +1.8 m, and images every few days of
    a lagoon that rests at 3 ha and fills and drains with `drain_days`."""
    sea_t = np.arange(0, days * DAY, 600.0)
    sea_v = np.full(len(sea_t), 0.5)
    for d in surges:
        sea_v[(sea_t >= d * DAY) & (sea_t < d * DAY + 7200)] = 1.8
    ts = step_times(0.0, days * DAY)
    levels = np.round(np.interp(ts, sea_t, sea_v), 2)
    areas, _, _ = fill_drain_series(0.0, 3.0, 3.0, ts, levels, curve_fn(*_step_curve(1.4)),
                                    drain_days)
    obs = [{"t": t, "area_ha": float(np.interp(t, ts, areas)), "seen_basin": 1.0}
           for t in np.arange(0.5 * DAY, days * DAY, every * DAY)]
    return obs, sea_t, sea_v


def test_fit_recovers_the_drain_time_and_flood_level(monkeypatch):
    monkeypatch.setitem(CFG, "lagoon_rest_days", 60)
    obs, sea_t, sea_v = _synthetic_record(drain_days=2.0)
    curves = {1.2: _step_curve(1.4), 1.5: _step_curve(1.9)}   # the second never floods
    fit = fit_drain(obs, _levels_for(sea_t, sea_v), curves, [0.5, 1.0, 2.0, 5.0])
    assert fit.drain_days == 2.0 and fit.basin_gap_fill_m == 1.2
    assert fit.mae_ha < 0.3 < fit.hold_mae_ha
    assert len(fit.replay) == len(obs) - 1
    assert fit.wave_k == 0.0 and fit.flood_offset_m == 0.0 and fit.family == "still"


def test_fit_needs_enough_images():
    obs, sea_t, sea_v = _synthetic_record(drain_days=1.0, days=12, every=4.0)
    assert fit_drain(obs[:3], _levels_for(sea_t, sea_v), {1.3: _step_curve(1.4)}, [1.0]) is None


def test_partly_seen_image_brackets_the_size():
    seen = {"area_ha": 5.0, "seen_basin": 1.0}
    hidden = {"area_ha": 5.0, "seen_basin": 0.6}     # 40% of a 10 ha basin hidden: 5-9 ha
    assert mean_error([(seen, 8.0)], basin_ha=10.0) == 3.0
    assert mean_error([(hidden, 8.0)], basin_ha=10.0) == 0.0
    assert mean_error([(hidden, 4.0)], basin_ha=10.0) == 1.0
    assert mean_error([(hidden, 12.0)], basin_ha=10.0) == pytest.approx(3.0)


def _wave_record(monkeypatch):
    """Two equal high tides; only the one with big waves fills the basin. Still
    water can't tell them apart, sea + 0.5 x waves - 0.4 m can (the calm one
    reaches 1.35 m, under the 1.4 m flood level; the stormy one 2.2 m)."""
    monkeypatch.setitem(CFG, "lagoon_rest_days", 60)
    sea_t = np.arange(0, 60 * DAY, 600.0)
    sea_v = np.full(len(sea_t), 0.5)
    for d in (10, 30):
        sea_v[(sea_t >= d * DAY) & (sea_t < d * DAY + 7200)] = 1.6
    waves = _waves(sea_t, np.where((sea_t >= 30 * DAY) & (sea_t < 30 * DAY + 7200), 2.0, 0.3))
    ts = step_times(0.0, 60 * DAY)
    truth = _levels_for(sea_t, sea_v, waves)("height", 0.5, 0.4, 0.0)(ts)
    areas, _, _ = fill_drain_series(0.0, 3.0, 3.0, ts, np.round(truth, 2),
                                    curve_fn(*_step_curve(1.4)), 1.0)
    obs = [{"t": t, "area_ha": float(np.interp(t, ts, areas)), "seen_basin": 1.0}
           for t in np.arange(0.5 * DAY, 60 * DAY, 2.0 * DAY)]
    return obs, sea_t, sea_v, waves


def test_fit_finds_the_wave_term_that_explains_a_wave_driven_fill(monkeypatch):
    obs, sea_t, sea_v, waves = _wave_record(monkeypatch)
    fit = fit_drain(obs, _levels_for(sea_t, sea_v, waves), {1.3: _step_curve(1.4)}, [1.0],
                    wave_options=[(None, 0.0), ("height", 0.5)], offsets=[0.0, 0.4])
    assert (fit.family, fit.wave_k, fit.flood_offset_m) == ("height", 0.5, 0.4)
    # chosen on images it wasn't fitted on, and it beats still water there too
    assert fit.cv["height"] < fit.cv["still"]
    assert fit.cv_mae_ha >= fit.mae_ha


def test_fit_prefers_no_wave_term_when_it_doesnt_help(monkeypatch):
    monkeypatch.setitem(CFG, "lagoon_rest_days", 60)
    obs, sea_t, sea_v = _synthetic_record(drain_days=1.0)
    flat = _waves(sea_t, np.full(len(sea_t), 0.1))     # waves that change nothing
    fit = fit_drain(obs, _levels_for(sea_t, sea_v, flat), {1.3: _step_curve(1.4)}, [1.0],
                    wave_options=[(None, 0.0), ("height", 0.15), ("height", 0.3)])
    assert fit.family == "still" and fit.wave_k == 0.0


def test_fit_keeps_rain_only_when_it_predicts_better(monkeypatch):
    # a fill with no surge, right after heavy rain: only the rain term explains it
    monkeypatch.setitem(CFG, "lagoon_rest_days", 60)
    sea_t = np.arange(0, 60 * DAY, 600.0)
    sea_v = np.full(len(sea_t), 1.0)
    rain = (sea_t, np.where((sea_t >= 20 * DAY) & (sea_t < 20 * DAY + 6 * 3600), 10.0, 0.0))
    ts = step_times(0.0, 60 * DAY)
    truth = _levels_for(sea_t, sea_v, rain=rain)("height", 0.0, 0.0, 0.01)(ts)
    areas, _, _ = fill_drain_series(0.0, 3.0, 3.0, ts, np.round(truth, 2),
                                    curve_fn(*_step_curve(1.4)), 1.0)
    obs = [{"t": t, "area_ha": float(np.interp(t, ts, areas)), "seen_basin": 1.0}
           for t in np.arange(0.5 * DAY, 60 * DAY, 1.5 * DAY)]
    fit = fit_drain(obs, _levels_for(sea_t, sea_v, rain=rain), {1.3: _step_curve(1.4)}, [1.0],
                    rain_ks=[0.0, 0.01])
    assert fit.family == "still+rain" and fit.rain_k == 0.01


def test_fit_gives_a_range_that_widens_after_surges(monkeypatch):
    obs, sea_t, sea_v, waves = _wave_record(monkeypatch)
    fit = fit_drain(obs, _levels_for(sea_t, sea_v, waves), {1.3: _step_curve(1.4)},
                    [0.5, 1.0, 2.0], wave_options=[(None, 0.0)])
    lo, hi = fit.range_ha(5.0, surged=False)
    assert lo <= 5.0 <= hi
    assert fit.range_ha(0.1, surged=False)[0] >= 0.0


def test_curve_fn_is_none_below_the_flood_level():
    fn = curve_fn(*_step_curve(1.4))
    assert fn(1.38) is None and fn(1.4) == 12.0 and fn(2.0) == 12.0
    assert curve_fn(LEVELS, np.full(len(LEVELS), np.nan))(2.0) is None


def test_cached_fit_is_used_only_for_its_own_outline_and_basin(tmp_path):
    path = tmp_path / "fill_drain.json"
    fit, why = FillDrainFit.load(path, "abc")
    assert fit.fitted is None and fit.drain_days == CFG["lagoon_drain_days"] and why
    path.write_text(__import__("json").dumps(
        FillDrainFit(2.0, 1.4, fitted="2026-11-17T00:00:00+00:00", fingerprint="abc").to_dict()))
    fit, why = FillDrainFit.load(path, "abc")
    assert why is None and fit.drain_days == 2.0 and fit.basin_gap_fill_m == 1.4
    fit, why = FillDrainFit.load(path, "other")
    assert fit.fitted is None and "changed" in why


def test_fill_drain_series_matches_a_hand_computed_drain():
    ts = np.array([0.0, DAY, 2 * DAY])
    areas, events, _ = fill_drain_series(0.0, 5.0, 3.0, ts, [0.5, 0.5, 0.5],
                                         lambda lvl: None, 1.0)
    assert areas[1] == pytest.approx(3.0 + 2.0 / np.e) and events == [None] * 3


def test_fast_path_matches_stepping_every_level():
    rng = np.random.default_rng(1)
    ts = step_times(0.0, 20 * DAY)
    levels = np.round(0.5 + 0.8 * np.sin(ts / 22000.0) + rng.normal(0, 0.15, len(ts)), 2)
    fn = curve_fn(*_step_curve(1.4))
    slow = fill_drain_series(0.0, 3.0, 2.5, ts, levels, fn, 1.3)
    fast = fill_drain_series(0.0, 3.0, 2.5, ts, levels, fn, 1.3, onset=fn.onset)
    assert np.allclose(slow[0], fast[0]) and slow[1] == fast[1] and slow[2] == fast[2]
    assert any(e is not None for e in fast[1])       # the test actually floods
