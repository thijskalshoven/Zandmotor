import pytest

from zandmotor.wind import compass, is_daylight, shore_class, wetsuit_advice


@pytest.mark.parametrize("deg,expected", [
    (0, "N"), (90, "E"), (180, "S"), (270, "W"),
    (359, "N"), (22.5, "NNE"), (350, "N"),
])
def test_compass_boundaries(deg, expected):
    assert compass(deg) == expected


def test_shore_class_onshore_at_normal():
    from zandmotor.config import CFG
    assert shore_class(CFG["shore_normal_deg"]) == "onshore"


def test_shore_class_offshore_opposite():
    from zandmotor.config import CFG
    opposite = (CFG["shore_normal_deg"] + 180) % 360
    assert shore_class(opposite) == "offshore"


def test_is_daylight_no_data_defaults_true():
    assert is_daylight([], 12345) is True


def test_is_daylight_inside_and_outside_window():
    daylight = [(100, 200)]
    assert is_daylight(daylight, 150) is True
    assert is_daylight(daylight, 50) is False


def test_wetsuit_advice_no_temperature():
    call, note = wetsuit_advice(None, None)
    assert call is None and note is None


def test_wetsuit_advice_picks_coldest_bracket():
    call, note = wetsuit_advice(5.0, None)
    assert "4/3" in call and "boots" in call


def test_wetsuit_advice_picks_warmest_bracket():
    call, note = wetsuit_advice(25.0, None)
    assert call == "shorty"
