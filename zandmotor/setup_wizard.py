"""`zandmotor --setup`: a few questions in Terminal that write settings.toml.

Written for people who have never edited a config file: every question has a
default in [brackets] that Enter keeps, numbers are checked as they are typed,
and a kite photo is added by dragging it into the window. Only the rider and
the kites are asked; the wetsuit table, the rideable-water threshold and the
advanced kite keys (storm kite, wind floor, notes) carry over from the
existing file untouched, and can be changed there by hand."""

from __future__ import annotations

import re
import shlex
import shutil
from pathlib import Path

from PIL import Image

from zandmotor.config import CFG, USER_DIR
from zandmotor.errors import SettingsError
from zandmotor.settings import load_settings, save_settings, settings_path


class _Prompter:
    def __init__(self, input_fn, print_fn):
        self.input, self.print = input_fn, print_fn

    def text(self, question, default=""):
        shown = f" [{default}]" if default not in ("", None) else ""
        answer = self.input(f"{question}{shown}: ").strip()
        return answer if answer else (default or "")

    def number(self, question, lo, hi, default=None, optional=False):
        while True:
            answer = self.text(question, "" if default is None else _fmt(default))
            if not answer and optional:
                return None
            try:
                v = float(answer.replace(",", "."))
            except ValueError:
                self.print(f"  Please type a number between {_fmt(lo)} and {_fmt(hi)}.")
                continue
            if lo <= v <= hi:
                return int(v) if v.is_integer() else v
            self.print(f"  That's outside {_fmt(lo)}-{_fmt(hi)}; please try again.")


def _fmt(v):
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def parse_range(answer):
    """"11-25", "11 25", "11 to 25" -> (11, 25); None if it isn't one."""
    nums = re.findall(r"\d+(?:[.,]\d+)?", answer)
    if len(nums) != 2:
        return None
    lo, hi = (float(n.replace(",", ".")) for n in nums)
    lo, hi = (int(x) if x.is_integer() else x for x in (lo, hi))
    return (lo, hi) if 0 < lo < hi <= 80 else None


def parse_dropped_path(answer):
    """A file dragged into Terminal arrives with its spaces escaped
    (/Users/me/My\\ Kite.jpg) or in quotes, plus a trailing space."""
    answer = answer.strip()
    try:
        parts = shlex.split(answer)
        answer = parts[0] if len(parts) == 1 else answer
    except ValueError:
        pass
    return Path(answer).expanduser()


def _ask_photo(p, kites_dir, current=None):
    while True:
        answer = p.text("  Photo: drag the picture into this window and press Enter "
                        "(optional)", current or "")
        if not answer or answer == current:
            return current or None
        src = parse_dropped_path(answer)
        try:
            with Image.open(src) as img:
                img.verify()
        except (OSError, ValueError):
            p.print("  Can't open that as a picture. Try again, or press Enter to skip.")
            continue
        kites_dir.mkdir(parents=True, exist_ok=True)
        dest = kites_dir / src.name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        p.print(f"  Copied to {dest}")
        return src.name


def _ask_kite(p, kites_dir, n):
    p.print(f"\nKite {n}")
    size = p.number("  Size in m² (press Enter when you have no more kites)", 1, 25,
                    optional=True)
    if size is None:
        return None
    name = p.text("  Name, e.g. Cabrinha Switchblade (optional)")
    while True:
        answer = p.text("  Wind range in knots from the maker's chart for a 75 kg rider, "
                        "e.g. 11-25 (Enter if you don't know)")
        rng = parse_range(answer) if answer else None
        if rng or not answer:
            break
        p.print("  Please type two numbers like 11-25, the low one first.")
    return {"name": name or "kite", "brand": "", "size_m2": size, "color": "", "year": None,
            "image": _ask_photo(p, kites_dir), "wind_range_75kg": rng}


def _describe(k):
    rng = k.get("wind_range_75kg")
    return (f"{_fmt(k['size_m2'])} m² {' '.join(x for x in (k.get('brand'), k['name']) if x)}"
            f", {f'{rng[0]}-{rng[1]} kn' if rng else 'range estimated'}")


def run_setup(input_fn=input, print_fn=print, path=None):
    """Asks the questions and writes settings.toml. Returns its path."""
    p = _Prompter(input_fn, print_fn)
    path = Path(path) if path else settings_path()
    kites_dir = path.parent / CFG["kite_images_dir"]
    current = None
    if path.exists():
        try:
            current = load_settings(path)
        except SettingsError as e:
            p.print(f"Your current settings can't be used ({e}); starting from scratch.\n")
    p.print("Zandmotor kite setup. Press Enter to keep the value in [brackets].\n")
    rider = dict((current or {}).get("rider") or CFG["rider"])
    rider["weight_kg"] = p.number("Your weight in kg", 30, 200,
                                  rider["weight_kg"] if current else None)
    p.print("Skill ceiling: the strongest gust you want to be out in. No kite is suggested "
            "past it.")
    rider["skill_ceiling_kn"] = p.number("Skill ceiling in knots", 10, 80,
                                         rider["skill_ceiling_kn"])

    kites = list(current["quiver"]) if current else []
    if kites:
        p.print("\nYour kites:")
        for i, k in enumerate(kites, 1):
            p.print(f"  {i}. {_describe(k)}")
        choice = p.text("Enter to keep them, n to enter them all again, a to add one").lower()
        if choice.startswith("n"):
            kites = []
        elif not choice.startswith("a"):
            choice = "keep"
    else:
        choice = "new"
        p.print("\nNow your kites, one at a time.")
    if choice != "keep":
        while True:
            k = _ask_kite(p, kites_dir, len(kites) + 1)
            if k is None:
                if kites:
                    break
                p.print("  You need at least one kite.")
                continue
            kites.append(k)

    values = dict(current or {})
    values.update(rider=rider, quiver=kites)
    saved = save_settings(values, path)
    p.print(f"\nSaved to {saved}. Edit that file for the wetsuit table, a storm kite and more.")
    return saved


def ensure_user_dir():
    USER_DIR.mkdir(parents=True, exist_ok=True)
    (USER_DIR / CFG["kite_images_dir"]).mkdir(exist_ok=True)
