from pathlib import Path

import pytest
from PIL import Image

from zandmotor.config import CFG
from zandmotor.errors import SettingsError
from zandmotor.settings import apply_user_settings, dump_settings, load_settings, save_settings
from zandmotor.setup_wizard import parse_dropped_path, parse_range, run_setup

GOOD = """
[rider]
weight_kg = 80
skill_ceiling_kn = 30

[[kite]]
size_m2 = 9
name = "Rebel"
wind_range_75kg = [14, 26]

[[kite]]
size_m2 = 4.5
exempt_from_ceiling = true
min_wind_kn = 28

[[wetsuit]]
below_c = 14
suit = "5/4"

[[wetsuit]]
suit = "3/2"
"""


def _write(tmp_path, text):
    p = tmp_path / "settings.toml"
    p.write_text(text)
    return p


def test_valid_file_maps_to_cfg_keys(tmp_path):
    out = load_settings(_write(tmp_path, GOOD))
    assert out["rider"]["weight_kg"] == 80 and out["rider"]["skill_ceiling_kn"] == 30
    assert out["quiver"][0] == {"name": "Rebel", "brand": "", "size_m2": 9, "color": "",
                                "year": None, "image": None, "wind_range_75kg": (14, 26)}
    assert out["quiver"][1]["exempt_from_ceiling"] and out["quiver"][1]["min_wind_kn"] == 28
    assert out["quiver"][1]["wind_range_75kg"] is None
    assert out["wetsuit"] == [(-99.0, 14, "5/4"), (14, 99.0, "3/2")]


def test_apply_merges_over_cfg(tmp_path, monkeypatch):
    monkeypatch.setitem(CFG, "rider", dict(CFG["rider"]))
    monkeypatch.setitem(CFG, "quiver", list(CFG["quiver"]))
    monkeypatch.setitem(CFG, "wetsuit", list(CFG["wetsuit"]))
    assert apply_user_settings(_write(tmp_path, GOOD))
    assert CFG["rider"]["weight_kg"] == 80 and len(CFG["quiver"]) == 2


def test_no_file_keeps_the_defaults(tmp_path):
    assert not apply_user_settings(tmp_path / "missing.toml")


@pytest.mark.parametrize("bad, words", [
    (GOOD.replace("weight_kg = 80", "weight_kg = 8"), "weight_kg"),
    (GOOD.replace("[14, 26]", "[26, 14]"), "wind_range_75kg"),
    (GOOD.replace('name = "Rebel"', 'nmae = "Rebel"'), "unknown setting nmae"),
    (GOOD.replace("weight_kg = 80", "weight_kg = eighty"), "not valid"),
    ("[rider]\nweight_kg = 80\nskill_ceiling_kn = 30\n", "no kites"),
    (GOOD.replace("below_c = 14", "below_c = -10"), None),
])
def test_bad_values_raise_a_settings_error_naming_them(tmp_path, bad, words):
    if words is None:                       # -10 C is out of range
        words = "below_c"
    with pytest.raises(SettingsError, match=words):
        load_settings(_write(tmp_path, bad))


def test_dump_reads_back_to_the_same_values(tmp_path):
    values = load_settings(_write(tmp_path, GOOD))
    values["quiver"][0]["note"] = 'a "quoted" note\nover two lines'
    values["rider"]["board"] = "Twin tip 138"
    back = load_settings(_write(tmp_path, dump_settings(values)))
    for k in values:
        assert back[k] == values[k], k


def test_save_refuses_a_file_it_could_not_read_back(tmp_path):
    values = load_settings(_write(tmp_path, GOOD))
    values["rider"]["weight_kg"] = 500
    with pytest.raises(SettingsError):
        save_settings(values, tmp_path / "new.toml")
    assert not (tmp_path / "new.toml").exists()


def test_parse_range_and_dropped_paths():
    assert parse_range("11-25") == (11, 25) == parse_range("11 to 25")
    assert parse_range("8,5 - 20") == (8.5, 20)
    assert parse_range("25-11") is None and parse_range("12") is None
    assert parse_dropped_path("/Users/me/My\\ Kite\\ 7.jpg ") == Path("/Users/me/My Kite 7.jpg")
    assert parse_dropped_path("'/tmp/a b.png'") == Path("/tmp/a b.png")


def test_wizard_writes_a_file_from_answers(tmp_path):
    photo = tmp_path / "drop" / "My Kite.png"
    photo.parent.mkdir()
    Image.new("RGB", (8, 8)).save(photo)
    escaped = str(photo).replace(" ", "\\ ")
    answers = iter(["82",                   # weight
                    "",                     # ceiling: keep the default
                    "abc", "10",            # size, after a typo
                    "Cabrinha Switchblade",
                    "11-25",
                    escaped + " ",          # photo dragged in
                    "",                     # no more kites
                    ])
    out = []
    path = run_setup(lambda q: next(answers), out.append, path=tmp_path / "settings.toml")
    s = load_settings(path)
    assert s["rider"]["weight_kg"] == 82
    assert s["rider"]["skill_ceiling_kn"] == CFG["rider"]["skill_ceiling_kn"]
    assert s["quiver"] == [{"name": "Cabrinha Switchblade", "brand": "", "size_m2": 10,
                            "color": "", "year": None, "image": "My Kite.png",
                            "wind_range_75kg": (11, 25)}]
    assert (tmp_path / "Kites" / "My Kite.png").exists()
    assert any("number" in line for line in out)


def test_wizard_rerun_keeps_kites_and_advanced_keys(tmp_path):
    path = _write(tmp_path, GOOD)
    answers = iter(["", "", ""])            # keep weight, ceiling and kites
    run_setup(lambda q: next(answers), lambda *a: None, path=path)
    s = load_settings(path)
    assert s["rider"]["weight_kg"] == 80 and len(s["quiver"]) == 2
    assert s["quiver"][1]["exempt_from_ceiling"] and s["wetsuit"][0][2] == "5/4"
