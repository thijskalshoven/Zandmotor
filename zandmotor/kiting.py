"""Kite quiver matching and the per-hour go/maybe/no verdict."""

from __future__ import annotations

import base64
import io

from PIL import Image

from zandmotor.config import CFG, HERE
from zandmotor.wind import compass, shore_class


def kite_advice(w, usable_ha, quiver):
    """Verdict for one hour: is there water, does a kite you own fit, and is
    anything about the wind unpleasant.

    usable_ha is the largest CONNECTED wet patch (see largest_patch_ha), not a
    total - the total is what let ~620 ha of North Sea, and later 250 ha of
    scattered beach, stand in for a rideable surface.

    The kite question is answered by the quiver (see quiver_advice), not by an
    abstract wind table. There used to be both, and they contradicted each
    other - the table called 26 kn "too strong for a beginner" while the
    quiver offered a 7 m as good to 37 kn.
    Returns (wind_txt, verdict, notes)."""
    notes = []
    water_ok = usable_ha >= CFG["min_rideable_ha"]
    if not water_ok:
        notes.append(f"largest patch of water only {usable_ha:.1f} ha "
                     f"(want {CFG['min_rideable_ha']:.0f})")
    if w is None:
        return "No wind data", "no", notes + ["no wind forecast"]

    spd, gst, deg = w[0], w[1], w[2]
    dir_cls = shore_class(deg)
    wind_txt = f"{spd:.0f} kn, gusts {gst:.0f} kn, {compass(deg)} ({dir_cls})"
    kite_ok = quiver["best"] is not None
    if not kite_ok:
        notes.append(quiver["conclusion"].lower())
    if quiver.get("over_ceiling"):
        notes.append(f"gusts past your {CFG['rider']['skill_ceiling_kn']:.0f} kn ceiling")
    elif quiver.get("past_ceiling"):
        # a ceiling-exempt kite fits: rideable, but never a plain "go"
        notes.append(f"gusts past your {CFG['rider']['skill_ceiling_kn']:.0f} kn ceiling, "
                     "storm kite only")
    if dir_cls == "offshore":
        notes.append("offshore wind, gusty over the dunes")
    if gst - spd >= CFG["gusty_delta_kn"]:
        notes.append("gusty")

    if not (water_ok and kite_ok) or quiver.get("over_ceiling"):
        verdict = "no"
    elif notes or quiver["best_status"] == "marginal":
        verdict = "maybe"
    else:
        verdict = "go"
    return wind_txt, verdict, notes


def quiver_image_uri(filename, warnings):
    """Embed a kite photo from CFG['kite_images_dir'] as a data URI, so the
    output HTML stays a single portable file. Swap in a replacement photo by
    keeping the same filename; a missing/renamed file degrades to a blank
    thumbnail plus a warning instead of a broken page.

    Downscaled to kite_thumb_px before embedding: the originals are up to
    1.6 MB each and are displayed at 56 px, which put 3.4 MB of invisible
    detail into a 4.6 MB page."""
    path = HERE / CFG["kite_images_dir"] / filename
    try:
        img = Image.open(io.BytesIO(path.read_bytes()))
    except (OSError, ValueError):
        warnings.append(f"Kite photo '{filename}' not found or unreadable in "
                        f"{CFG['kite_images_dir']}/; showing a blank thumbnail.")
        return ""
    n = CFG["kite_thumb_px"]
    img = img.convert("RGB")
    img.thumbnail((n, n), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def kite_wind_range(kite, rider):
    """This kite's usable wind range for this rider, in knots.

    Manufacturer charts assume a 75 kg rider. Shift the range's CENTRE by
    weight/75 (for a fixed kite size the usable wind scales with rider weight,
    per size_m2 = weight_kg*2.2/wind_kn) but keep its WIDTH - how wide a
    window a kite has is a property of its depower and the rider's skill, and
    does not stretch because the rider is heavier. Scaling both endpoints, as
    this used to, inflated an 11-25 kn chart into 12.9-29.3 and turned a
    formula-estimated 7 m into "good to 37 kn".

    The top is then capped at skill_ceiling_kn - what a kite can technically
    hold and what you want to be out in are different numbers - unless the
    kite is marked exempt_from_ceiling (a storm kite that only makes sense
    above it).
    Returns (lo, hi, estimated, capped)."""
    if kite["wind_range_75kg"] is not None:
        lo75, hi75 = kite["wind_range_75kg"]
        estimated = False
    else:
        centre75 = 75 * 2.2 / kite["size_m2"]
        f = CFG["kite_est_range_frac"]
        lo75, hi75 = centre75 * (1 - f), centre75 * (1 + f)
        estimated = True
    half = (hi75 - lo75) / 2
    centre = (lo75 + hi75) / 2 * rider["weight_kg"] / 75.0
    lo, hi = max(centre - half, 0.0), centre + half
    ceiling = rider.get("skill_ceiling_kn")
    capped = ceiling is not None and hi > ceiling and not kite.get("exempt_from_ceiling")
    if capped:
        hi = float(ceiling)
    return min(lo, hi), hi, estimated, capped


def quiver_advice(w, rider):
    """Given the current wind and the rider's own quiver (CFG['quiver']),
    work out which owned kite fits best right now.

    Ranges come from kite_wind_range (see there for the weight shift and the
    skill ceiling). The LOWER bound is tested against the mean wind and the
    UPPER against the gust: the gust is what actually overpowers you, so a
    10 kn mean gusting to 30 is not a 10 kn day for kite choice.

    A kite is "ideal" when the wind sits inside its range, "marginal" within a
    further kite_marginal_frac buffer, else "off". The buffer never extends
    past the skill ceiling - that bound is meant to be hard - except for a
    kite marked exempt_from_ceiling. Gusts past the ceiling only rule the hour
    out when no exempt kite fits them."""
    kites = []
    for k in CFG["quiver"]:
        lo, hi, estimated, capped = kite_wind_range(k, rider)
        kites.append({"name": k["name"], "size_m2": k["size_m2"],
                      "range": (round(lo, 1), round(hi, 1)),
                      "estimated": estimated, "capped": capped,
                      "exempt": bool(k.get("exempt_from_ceiling"))})

    if w is None:
        for k in kites:
            k["status"] = "off"
        return {"kites": kites, "best": None, "best_status": None,
                "over_ceiling": False, "past_ceiling": False, "conclusion": "No wind data"}

    spd, gst = w[0], w[1]
    ceiling = rider.get("skill_ceiling_kn")
    past_ceiling = ceiling is not None and gst > ceiling
    m = CFG["kite_marginal_frac"]
    for k in kites:
        lo, hi = k["range"]
        hi_buf = hi * (1 + m)
        if ceiling is not None and not k["exempt"]:
            hi_buf = min(hi_buf, float(ceiling))
        if lo <= spd and gst <= hi:
            k["status"] = "ideal"
        elif lo * (1 - m) <= spd and gst <= hi_buf:
            k["status"] = "marginal"
        else:
            k["status"] = "off"

    ideal = [k for k in kites if k["status"] == "ideal"]
    marginal = [k for k in kites if k["status"] == "marginal"]
    # past the ceiling only exempt kites can still fit (every other range
    # stops at it), so the hour is ruled out only if none of them do
    over_ceiling = past_ceiling and not (ideal or marginal)

    def center_dist(k):
        lo, hi = k["range"]
        return abs(spd - (lo + hi) / 2)

    best_status = None
    if ideal:
        best = min(ideal, key=center_dist)
        best_status = "ideal"
        conclusion = f"{best['size_m2']}m² {best['name']} — right in its sweet spot at {spd:.0f} kn"
    elif marginal:
        best = min(marginal, key=center_dist)
        best_status = "marginal"
        side = "underpowered" if spd < sum(best["range"]) / 2 else "overpowered"
        conclusion = (f"{best['size_m2']}m² {best['name']} is closest, but a little {side} "
                      f"at {spd:.0f} kn gusting {gst:.0f}")
    else:
        best = None
        biggest = max(kites, key=lambda k: k["size_m2"])
        if over_ceiling:
            conclusion = (f"Gusting {gst:.0f} kn, past your {ceiling:.0f} kn ceiling — "
                          "not a day for any of them")
        elif spd < biggest["range"][0]:
            conclusion = f"Too light for all {len(kites)} kites right now ({spd:.0f} kn)"
        else:
            conclusion = (f"Too strong for all {len(kites)} kites right now "
                          f"({spd:.0f} kn gusting {gst:.0f})")

    return {"kites": kites, "best": best["name"] if best else None,
            "best_status": best_status, "over_ceiling": over_ceiling,
            "past_ceiling": past_ceiling,
            "conclusion": conclusion}
