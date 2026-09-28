"""Assembles the frame data and writes the final self-contained HTML report."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from zandmotor.config import CFG, TZ
from zandmotor.kiting import quiver_image_uri

_TEMPLATE_PATH = Path(__file__).resolve().parent / "template.html"


def write_html(path, frames, grid, lagoon_ring, location, terrain_note, warnings, terrain_uri,
               windows, tides, lagoon_obs, lagoon_label, sources):
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    lats = [p[1] for p in lagoon_ring]
    lons = [p[0] for p in lagoon_ring]
    quiver_meta = [{"name": k["name"], "size_m2": k["size_m2"], "brand": k["brand"],
                    "color": k["color"], "year": k["year"],
                    "image": quiver_image_uri(k["image"], warnings)} for k in CFG["quiver"]]
    data = {"bounds": grid.latlon_bounds, "frames": frames, "terrain": terrain_uri,
            "lagoon": [[lat, lon] for lon, lat in lagoon_ring],
            "lagoon_bounds": [[min(lats), min(lons)], [max(lats), max(lons)]],
            "quiver_meta": quiver_meta, "rider": CFG["rider"],
            "min_rideable_ha": CFG["min_rideable_ha"],
            "windows": windows["list"], "windows_none": windows["none"],
            "windows_caveat": windows["caveat"],
            "tides": tides, "lagoon_obs": lagoon_obs, "lagoon_label": lagoon_label,
            "sources": sources}
    warn_html = "".join(f'<div class="warn">{w}</div>' for w in warnings)
    html = (template
            .replace("__DATA__", json.dumps(data))
            .replace("__MAXI__", str(len(frames) - 1))
            .replace("__GENERATED__", dt.datetime.now(TZ).strftime("%a %d %b %H:%M"))
            .replace("__LOCATION__", location)
            .replace("__TERRAIN__", terrain_note)
            .replace("__WARNINGS__", warn_html))
    path.write_text(html, encoding="utf-8")
