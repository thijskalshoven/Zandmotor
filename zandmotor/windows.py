"""Multi-day rideable-window search over the wind forecast horizon."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np

from zandmotor.config import TZ, UTC
from zandmotor.flood import rideable_at
from zandmotor.wind import compass, is_daylight


def find_windows(now, water, wind, curve, daylight, max_windows=3, step_min=60):
    """Runs of consecutive daylight hours whose verdict is go or maybe.

    This is the question the tool is actually for: most hours are "no", so an
    8-hour slider showing thirteen refusals answers nothing. Searches the whole
    wind forecast horizon.

    Caveat worth remembering when reading the output: with the lagoon in lake
    mode its area is a constant, so what varies across the week is wind, tide
    (via the intertidal strip) and daylight - not the lagoon."""
    if wind is None:
        return []
    t_end = min(float(wind[0][-1]), water.covers_until())
    windows, cur = [], None
    te = now.timestamp()
    while te <= t_end:
        ok = is_daylight(daylight, te)
        if ok:
            _, _, w, quiver, verdict, _, _, _ = rideable_at(te, water, wind, curve)
            ok = verdict in ("go", "maybe")
        if ok:
            if cur is None:
                cur = {"start": te, "end": te, "verdict": verdict,
                       "kite": quiver["best"], "size": next(
                           (k["size_m2"] for k in quiver["kites"]
                            if k["name"] == quiver["best"]), None),
                       "spd": [w[0]], "gst": [w[1]], "deg": [w[2]]}
            else:
                cur["end"] = te
                cur["spd"].append(w[0])
                cur["gst"].append(w[1])
                cur["deg"].append(w[2])
                if verdict == "go":
                    cur["verdict"] = "go"
        elif cur is not None:
            windows.append(cur)
            cur = None
            if len(windows) >= max_windows:
                break
        te += step_min * 60
    if cur is not None and len(windows) < max_windows:
        windows.append(cur)
    for win in windows:
        win["mean_kn"] = float(np.mean(win["spd"]))
        win["gust_kn"] = float(np.max(win["gst"]))
        rad = np.radians(win["deg"])
        win["dir_deg"] = math.degrees(math.atan2(float(np.mean(np.sin(rad))),
                                                 float(np.mean(np.cos(rad))))) % 360
        for k in ("spd", "gst", "deg"):
            win.pop(k)
    return windows


def format_window(win, step_min=60):
    """One human line for a rideable window, in local time.

    The end is extended by one step: an hour that qualifies is rideable for
    that hour, so a single qualifying hour reads "15:00-16:00" rather than the
    zero-length "15:00-15:00"."""
    a = dt.datetime.fromtimestamp(win["start"], UTC).astimezone(TZ)
    b = dt.datetime.fromtimestamp(win["end"] + step_min * 60, UTC).astimezone(TZ)
    kite = f"{win['size']:g} m {win['kite']}" if win["kite"] else "no kite fits"
    hours = (win["end"] - win["start"]) / 3600 + step_min / 60
    same = a.date() == b.date()
    when = (f"{a:%a %d %b %H:%M}–{b:%H:%M}" if same
            else f"{a:%a %d %b %H:%M}–{b:%a %d %b %H:%M}")
    return (f"{when} ({hours:.0f} h) · {kite} · {win['mean_kn']:.0f} kn gusting "
            f"{win['gust_kn']:.0f}, {compass(win['dir_deg'])}")
