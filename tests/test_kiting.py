from zandmotor.kiting import kite_advice, kite_wind_range, quiver_advice


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
    result = quiver_advice(None, {"weight_kg": 88, "skill_ceiling_kn": 28})
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
