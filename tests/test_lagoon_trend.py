import datetime as dt
from types import SimpleNamespace

import numpy as np

from zandmotor.cli import _lagoon_trend
from zandmotor.config import UTC
from zandmotor.lagoon_model.fill_drain_fit import FillDrainFit
from zandmotor.lagoon_model.types import LakeResponse

NOW = dt.datetime(2026, 11, 17, 12, tzinfo=UTC)
DAY = 86400.0


def _model(days_ago_and_ha):
    te = NOW.timestamp()
    obs = [{"t": te - d * DAY, "area_ha": a, "sensor": "optical", "track": "S2",
            "seen_basin": 1.0 if d != 2 else 0.6} for d, a in days_ago_and_ha]
    return LakeResponse(r=0.1, wind_r=0.1, n=len(obs), fitted=NOW.isoformat(),
                        times=[o["t"] for o in obs], areas=[o["area_ha"] for o in obs],
                        n_optical=len(obs), n_sar=0, obs=obs)


def _fit():
    levels = np.round(np.arange(1.0, 2.31, 0.02), 2)
    return FillDrainFit(1.0, 1.3, fitted=NOW.isoformat(), curve=[
        [float(lvl), 12.0 if lvl >= 1.5 else None] for lvl in levels])


def test_trend_replays_between_images_and_marks_one_surge_per_episode():
    model = _model([(9, 4.0), (5, 4.2), (2, 5.0)])
    latest = model.obs[-1]
    surge_day = NOW.timestamp() - 4 * DAY       # a 3-hour surge peaking at +1.8 m

    def flood_level(te):
        dt_h = (te - surge_day) / 3600
        return (1.2 + 0.6 * (1 - abs(dt_h - 1.5) / 1.5) if 0 <= dt_h <= 3 else 0.5), "measured"

    lw = SimpleNamespace(fill_drain=None, range_at=lambda t, area_ha=None: None)
    T = _lagoon_trend(model, lw, _fit(), flood_level, NOW, latest, lambda t: "x")
    assert [m["ha"] for m in T["images"]] == [4.0, 4.2, 5.0]
    assert T["images"][2]["partial"] and not T["images"][0]["partial"]
    assert T["t0"] == model.obs[0]["t"] - 2 * DAY          # starts just before the first image
    kinds = [g["kind"] for g in T["segments"]]
    assert kinds == ["replay", "replay", "estimate"]
    # each replay starts at its image, and the one spanning the surge fills to the curve
    assert T["segments"][1]["pts"][0] == [model.obs[1]["t"], 4.2]
    assert max(p[1] for p in T["segments"][1]["pts"]) == 12.0
    # the surge rose over several steps but is one marker, at its peak
    assert len(T["surges"]) == 1 and T["surges"][0]["t"] == surge_day + 1.5 * 3600


def test_verdict_tests_the_low_end_of_the_range():
    from zandmotor.flood import rideable_at
    t = np.arange(4) * 3600.0
    wind = (t, np.full(4, 18.0), np.full(4, 22.0), np.full(4, 270.0), np.full(4, 15.0), [],
            np.zeros(4))
    mask = np.ones((2, 2), bool)
    sure = SimpleNamespace(at=lambda te: (mask, 5.0, 5.0), range_at=lambda te, area_ha=None: None)
    unsure = SimpleNamespace(at=lambda te: (mask, 5.0, 5.0),
                             range_at=lambda te, area_ha=None: (2.0, 8.0))
    assert rideable_at(3600.0, sure, wind)[4] != "no"
    assert rideable_at(3600.0, unsure, wind)[4] == "no"     # 2 ha is below enough to ride


def test_trend_is_only_for_lake_mode():
    model = _model([(5, 4.2)])
    tidal = SimpleNamespace(mode="tidal")
    assert _lagoon_trend(tidal, None, _fit(), None, NOW, model.obs[0], str) is None
    assert _lagoon_trend(model, None, _fit(), None, NOW, None, str) is None
