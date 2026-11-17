import pytest

from zandmotor.config import CFG
from zandmotor.kiting import kite_advice, kite_wind_range, quiver_advice

# The storm-kite tests below were written against this rider and quiver, which
# used to be CFG's own; CFG now ships an example rider (settings.toml holds the
# real one), so they set it explicitly.
RIDER = {"weight_kg": 88, "height_cm": 192, "board": "North Prime 141x41",
         "skill_ceiling_kn": 46}
QUIVER = [
    {"name": "Drifter", "size_m2": 4.5, "wind_range_75kg": (28, 40),
     "exempt_from_ceiling": True, "min_wind_kn": 28},
    {"name": "Drifter", "size_m2": 7, "wind_range_75kg": None},
    {"name": "Switchblade", "size_m2": 10, "wind_range_75kg": (11, 25)},
    {"name": "Bandit", "size_m2": 14, "wind_range_75kg": (8, 20)},
]


@pytest.fixture
def storm_quiver(monkeypatch):
    monkeypatch.setitem(CFG, "rider", RIDER)
    monkeypatch.setitem(CFG, "quiver", QUIVER)


def test_kite_wind_range_uses_manufacturer_chart_when_given():
    kite = {"wind_range_75kg": (11, 25), "size_m2": 10}
    rider = {"weight_kg": 75, "skill_ceiling_kn": 40}
    lo, hi, estimated, capped = kite_wind_range(kite, rider)
    assert not estimated
    assert not capped
    assert lo == 11 and hi == 25


def test_kite_wind_range_caps_at_skill_ceiling():
    kite = {"wind_range_75kg": (11, 25), "size_m2": 10}
    rider = {"weight_kg": 75, "skill_ceiling_kn": 20}
    lo, hi, estimated, capped = kite_wind_range(kite, rider)
    assert capped
    assert hi == 20


def test_kite_wind_range_estimates_without_chart():
    kite = {"wind_range_75kg": None, "size_m2": 7}
    rider = {"weight_kg": 75, "skill_ceiling_kn": 40}
    lo, hi, estimated, capped = kite_wind_range(kite, rider)
    assert estimated
    assert lo < hi


def test_quiver_advice_no_wind_data():
    result = quiver_advice(None, {"weight_kg": 88, "skill_ceiling_kn": 46})
    assert result["best"] is None
    assert all(k["status"] == "off" for k in result["kites"])


def test_quiver_advice_picks_ideal_kite():
    rider = {"weight_kg": 75, "skill_ceiling_kn": 40}
    # wind squarely inside a manufacturer-chart kite's range
    w = (15.0, 18.0, 270.0, 15.0)
    result = quiver_advice(w, rider)
    assert result["best_status"] in ("ideal", "marginal")


def test_kite_advice_no_wind_data_is_no():
    quiver = {"best": None, "best_status": None, "over_ceiling": False,
              "conclusion": "No wind data"}
    wind_txt, verdict, notes = kite_advice(None, 5.0, quiver)
    assert verdict == "no"


def test_kite_advice_not_enough_water_is_no():
    quiver = {"best": "Bandit", "best_status": "ideal", "over_ceiling": False,
              "conclusion": "ok"}
    w = (12.0, 14.0, 270.0, 15.0)
    wind_txt, verdict, notes = kite_advice(w, 0.5, quiver)
    assert verdict == "no"
    assert any("largest patch" in n for n in notes)


def test_kite_wind_range_exempt_kite_ignores_ceiling():
    kite = {"wind_range_75kg": (40, 55), "size_m2": 3, "exempt_from_ceiling": True}
    rider = {"weight_kg": 75, "skill_ceiling_kn": 46}
    lo, hi, estimated, capped = kite_wind_range(kite, rider)
    assert not capped
    assert lo == 40 and hi == 55


def test_quiver_advice_exempt_kite_rides_past_ceiling(monkeypatch):
    monkeypatch.setitem(CFG, "quiver", [
        {"name": "Drifter", "size_m2": 3, "wind_range_75kg": (40, 55), "exempt_from_ceiling": True},
        {"name": "Switchblade", "size_m2": 10, "wind_range_75kg": (11, 25)},
    ])
    rider = {"weight_kg": 75, "skill_ceiling_kn": 46}
    result = quiver_advice((44.0, 50.0, 270.0, 15.0), rider)
    assert result["best"] == "Drifter"
    assert result["past_ceiling"] and not result["over_ceiling"]
    # rideable, but past the ceiling it is never a plain "go"
    wind_txt, verdict, notes = kite_advice((44.0, 50.0, 270.0, 15.0), 5.0, result)
    assert verdict == "maybe"
    assert any("storm kite only" in n for n in notes)


def test_quiver_advice_past_ceiling_without_exempt_kite_is_no(monkeypatch):
    monkeypatch.setitem(CFG, "quiver", [
        {"name": "Switchblade", "size_m2": 10, "wind_range_75kg": (11, 25)},
    ])
    rider = {"weight_kg": 75, "skill_ceiling_kn": 46}
    result = quiver_advice((40.0, 48.0, 270.0, 15.0), rider)
    assert result["over_ceiling"]
    wind_txt, verdict, notes = kite_advice((40.0, 48.0, 270.0, 15.0), 5.0, result)
    assert verdict == "no"


def _storm(result):
    return next(k for k in result["kites"] if k["size_m2"] == 4.5)


def test_storm_kite_range_starts_at_its_floor_for_this_rider(storm_quiver):
    storm = next(k for k in CFG["quiver"] if k.get("min_wind_kn") is not None)
    lo, hi, estimated, capped = kite_wind_range(storm, CFG["rider"])
    # the 28-40 kn chart would shift to 33.9-45.9 kn at 88 kg; the floor keeps 28
    assert lo == storm["min_wind_kn"] == 28
    assert hi > 40 and not capped


def test_storm_kite_is_picked_once_the_7m_is_overpowered(storm_quiver):
    # under the 46 kn ceiling a storm-kite hour is an ordinary "go"
    result = quiver_advice((36.0, 42.0, 270.0, 12.0), CFG["rider"])
    assert result["best_size_m2"] == 4.5 and _storm(result)["status"] == "ideal"
    assert kite_advice((36.0, 42.0, 270.0, 12.0), 5.0, result)[1] == "go"
    # just above 28 kn the 7 m still fits and is the better pick
    assert quiver_advice((29.0, 33.0, 270.0, 12.0), CFG["rider"])["best_size_m2"] == 7


def test_storm_kite_past_the_46_kn_ceiling_is_maybe(storm_quiver):
    result = quiver_advice((40.0, 47.0, 270.0, 12.0), CFG["rider"])
    assert result["best_size_m2"] == 4.5 and result["past_ceiling"]
    wind_txt, verdict, notes = kite_advice((40.0, 47.0, 270.0, 12.0), 5.0, result)
    assert verdict == "maybe"                # past the ceiling, never a plain "go"
    assert any("storm kite only" in n for n in notes)


def test_storm_kite_is_off_at_or_below_28_kn_with_no_marginal_band(storm_quiver):
    for w in [(28.0, 30.0, 270.0, 12.0),     # exactly 28: not "above 28"
              (26.0, 27.0, 270.0, 12.0),     # would be "marginal" without the floor
              (24.0, 31.0, 270.0, 12.0)]:    # gusty, but the mean is too light
        assert _storm(quiver_advice(w, CFG["rider"]))["status"] == "off"


def test_gusts_past_the_ceiling_below_28_kn_mean_is_no(storm_quiver):
    # the storm kite's 28 kn floor is on the mean, so it can't take these gusts
    result = quiver_advice((26.0, 48.0, 270.0, 12.0), CFG["rider"])
    assert result["over_ceiling"] and result["best"] is None
    assert kite_advice((26.0, 48.0, 270.0, 12.0), 5.0, result)[1] == "no"


def test_best_kite_carries_its_size_when_names_repeat(storm_quiver):
    # 26 kn gusting 27: the 7 m Drifter fits, not the 4.5 m storm Drifter
    result = quiver_advice((26.0, 27.0, 270.0, 12.0), CFG["rider"])
    assert result["best"] == "Drifter" and result["best_size_m2"] == 7
