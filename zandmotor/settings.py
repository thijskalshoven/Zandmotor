"""Each user's own settings: settings.toml in USER_DIR, merged over CFG.

CFG holds the site and model constants and an example rider. The rider, the
quiver, the wetsuit table and the rideable-water threshold are personal, so
they come from this file instead, written by `zandmotor --setup`
(setup_wizard) and editable by hand. Only those keys can be set here: a file
that could override any CFG constant would let a typo quietly change the
lagoon model for one user.

The file's layout is friendlier than CFG's (a [[kite]] block per kite, a
wetsuit table by "below this temperature"); load_settings and dump_settings
convert between the two, so the rest of the code only ever sees CFG."""

from __future__ import annotations

import json
import string
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:                                       # pragma: no cover
    import tomli as tomllib

from zandmotor.config import CFG, USER_DIR
from zandmotor.errors import SettingsError

_TEMPLATE_PATH = Path(__file__).resolve().parent / "settings_template.toml"

FIX_HINT = ("Double-click \"Change settings\" in your Zandmotor Kite folder (or run "
            "zandmotor --setup) to write it again, or fix it in a text editor.")

# What each table may hold; anything else is a typo or a key from a newer version
_TABLES = {
    "rider": {"weight_kg", "skill_ceiling_kn", "height_cm", "board"},
    "riding": {"min_rideable_ha"},
    "kite": {"size_m2", "name", "brand", "color", "year", "photo", "wind_range_75kg",
             "exempt_from_ceiling", "min_wind_kn", "note"},
    "wetsuit": {"below_c", "suit"},
    "wetsuit_chill": {"air_delta_c", "wind_kn"},
}
# wetsuit_advice reads the table as [lo, hi) bands; these stand in for "no limit"
_COLDEST, _WARMEST = -99.0, 99.0


def settings_path() -> Path:
    return USER_DIR / CFG["settings_file"]


def _fail(path, msg):
    raise SettingsError(f"{path}: {msg}. {FIX_HINT}")


def _check_keys(path, where, table, allowed):
    if not isinstance(table, dict):
        _fail(path, f"{where} should be a table of settings")
    unknown = sorted(set(table) - allowed)
    if unknown:
        _fail(path, f"unknown setting {', '.join(unknown)} in {where} "
                    f"(allowed: {', '.join(sorted(allowed))})")


def _number(path, where, table, key, lo, hi, required=True, integer=False):
    v = table.get(key)
    if v is None:
        if required:
            _fail(path, f"{where} needs {key}")
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and v != int(v)):
        _fail(path, f"{where} {key} should be a {'whole ' if integer else ''}number, not {v!r}")
    if not lo <= v <= hi:
        _fail(path, f"{where} {key} = {v} is outside {lo}-{hi}")
    return int(v) if integer else v


def _text(path, where, table, key):
    v = table.get(key, "")
    if not isinstance(v, str):
        _fail(path, f"{where} {key} should be text in quotes, not {v!r}")
    return v


def _kite(path, i, k):
    where = f"kite {i + 1}"
    _check_keys(path, where, k, _TABLES["kite"])
    size = _number(path, where, k, "size_m2", 1, 25)
    rng = k.get("wind_range_75kg")
    if rng is not None:
        if (not isinstance(rng, list) or len(rng) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in rng)
                or not 0 < rng[0] < rng[1] <= 80):
            _fail(path, f"{where} wind_range_75kg should be two numbers like [11, 25], "
                        f"low first, not {rng!r}")
        rng = (rng[0], rng[1])
    exempt = k.get("exempt_from_ceiling", False)
    if not isinstance(exempt, bool):
        _fail(path, f"{where} exempt_from_ceiling should be true or false")
    out = {"name": _text(path, where, k, "name") or "kite", "brand": _text(path, where, k, "brand"),
           "size_m2": size, "color": _text(path, where, k, "color"),
           "year": _number(path, where, k, "year", 1990, 2100, required=False, integer=True),
           "image": _text(path, where, k, "photo") or None, "wind_range_75kg": rng}
    if exempt:
        out["exempt_from_ceiling"] = True
    floor = _number(path, where, k, "min_wind_kn", 0, 80, required=False)
    if floor is not None:
        out["min_wind_kn"] = floor
    note = _text(path, where, k, "note")
    if note:
        out["note"] = note
    return out


def _wetsuit(path, rows):
    if not isinstance(rows, list) or not rows:
        _fail(path, "the wetsuit table needs at least one [[wetsuit]] block")
    table, lo = [], _COLDEST
    for i, row in enumerate(rows):
        where = f"wetsuit {i + 1}"
        _check_keys(path, where, row, _TABLES["wetsuit"])
        suit = _text(path, where, row, "suit")
        if not suit:
            _fail(path, f"{where} needs suit = \"...\"")
        last = i == len(rows) - 1
        hi = _number(path, where, row, "below_c", -5, 40, required=not last)
        hi = _WARMEST if hi is None else hi
        if hi <= lo:
            _fail(path, f"{where} below_c = {hi} should be warmer than the block before it")
        table.append((lo, hi, suit))
        lo = hi
    return table


def load_settings(path) -> dict:
    """settings.toml as CFG entries ({"rider": ..., "quiver": [...], ...}).
    Raises SettingsError, naming the setting, on anything it can't use."""
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        _fail(path, f"could not be read ({e})")
    except tomllib.TOMLDecodeError as e:
        _fail(path, f"is not valid ({e})")
    _check_keys(path, "the file", raw, set(_TABLES))

    rider = raw.get("rider")
    if rider is None:
        _fail(path, "the [rider] table is missing")
    _check_keys(path, "[rider]", rider, _TABLES["rider"])
    out = {"rider": {
        "weight_kg": _number(path, "[rider]", rider, "weight_kg", 30, 200),
        "height_cm": _number(path, "[rider]", rider, "height_cm", 100, 250, required=False),
        "board": _text(path, "[rider]", rider, "board"),
        "skill_ceiling_kn": _number(path, "[rider]", rider, "skill_ceiling_kn", 10, 80)}}

    kites = raw.get("kite")
    if not isinstance(kites, list) or not kites:
        _fail(path, "there are no kites: add at least one [[kite]] block")
    out["quiver"] = [_kite(path, i, k) for i, k in enumerate(kites)]

    riding = raw.get("riding", {})
    _check_keys(path, "[riding]", riding, _TABLES["riding"])
    ha = _number(path, "[riding]", riding, "min_rideable_ha", 0.1, 50, required=False)
    if ha is not None:
        out["min_rideable_ha"] = ha

    if "wetsuit" in raw:
        out["wetsuit"] = _wetsuit(path, raw["wetsuit"])
    chill = raw.get("wetsuit_chill", {})
    _check_keys(path, "[wetsuit_chill]", chill, _TABLES["wetsuit_chill"])
    for key, cfg_key, hi in (("air_delta_c", "wetsuit_chill_air_delta_c", 20),
                             ("wind_kn", "wetsuit_chill_wind_kn", 60)):
        v = _number(path, "[wetsuit_chill]", chill, key, 0, hi, required=False)
        if v is not None:
            out[cfg_key] = v
    return out


def apply_user_settings(path=None) -> bool:
    """Merges settings.toml over CFG. Returns False when there is no file,
    leaving CFG's example rider in place (the caller warns about that)."""
    path = Path(path) if path else settings_path()
    if not path.exists():
        return False
    CFG.update(load_settings(path))
    return True


def _toml(v):
    """One value in TOML. JSON's string escapes are valid in TOML basic
    strings, so no TOML writer is needed for this handful of types."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, float) and v.is_integer():
        return str(int(v)) if abs(v) < 1e15 else repr(v)
    return repr(v)


def dump_settings(values) -> str:
    """settings.toml text for CFG-style `values` (the keys load_settings
    returns; missing ones are taken from CFG). load_settings reads it back
    to the same values."""
    v = {k: values.get(k, CFG[k]) for k in ("rider", "quiver", "min_rideable_ha", "wetsuit",
                                            "wetsuit_chill_air_delta_c", "wetsuit_chill_wind_kn")}
    rider = v["rider"]
    kites = []
    for k in v["quiver"]:
        lines = ["[[kite]]", f"size_m2 = {_toml(k['size_m2'])}"]
        for key, val in (("name", k.get("name")), ("brand", k.get("brand")),
                         ("color", k.get("color")), ("year", k.get("year")),
                         ("wind_range_75kg", k.get("wind_range_75kg")),
                         ("photo", k.get("image")),
                         ("exempt_from_ceiling", k.get("exempt_from_ceiling") or None),
                         ("min_wind_kn", k.get("min_wind_kn")), ("note", k.get("note"))):
            if val not in (None, ""):
                lines.append(f"{key} = {_toml(val)}")
        kites.append("\n".join(lines) + "\n")
    suits = []
    for lo, hi, suit in v["wetsuit"]:
        lines = ["[[wetsuit]]"]
        if hi < _WARMEST:
            lines.append(f"below_c = {_toml(hi)}")
        lines.append(f"suit = {_toml(suit)}")
        suits.append("\n".join(lines) + "\n")
    height, board = rider.get("height_cm"), rider.get("board")
    return string.Template(_TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(
        weight_kg=_toml(rider["weight_kg"]), skill_ceiling_kn=_toml(rider["skill_ceiling_kn"]),
        height_cm_line=f"height_cm = {_toml(height)}" if height else "# height_cm = 180",
        board_line=f"board = {_toml(board)}" if board else "# board = \"your board\"",
        min_rideable_ha=_toml(v["min_rideable_ha"]),
        kites="\n".join(kites), wetsuit="\n".join(suits),
        air_delta_c=_toml(v["wetsuit_chill_air_delta_c"]),
        wind_kn=_toml(v["wetsuit_chill_wind_kn"]))


def save_settings(values, path=None):
    """Writes settings.toml, after checking it reads back: a file the next
    run would reject is never left behind."""
    path = Path(path) if path else settings_path()
    text = dump_settings(values)
    tmp = path.with_suffix(".toml.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding="utf-8")
    try:
        load_settings(tmp)
    except SettingsError:
        tmp.unlink()
        raise
    tmp.replace(path)
    return path
