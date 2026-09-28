"""Rijkswaterstaat water level: measured, forecast, astronomical, with fallbacks."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import requests

from zandmotor.config import CFG, log
from zandmotor.errors import HarmonicFitUnavailableError, NoWaterLevelDataError
from zandmotor.time_utils import format_rws_datetime, parse_iso_datetime


def rws_series(location, start, end, proces):
    """Return (times[s since epoch], values[m NAP]) for one ProcesType
    ('meting', 'verwachting', 'astronomisch')."""
    def request(with_proces):
        aquo = {"Compartiment": {"Code": "OW"}, "Grootheid": {"Code": "WATHTE"}}
        if with_proces:
            aquo["ProcesType"] = proces
        body = {"Locatie": {"Code": location},
                "AquoPlusWaarnemingMetadata": {"AquoMetadata": aquo},
                "Periode": {"Begindatumtijd": format_rws_datetime(start),
                            "Einddatumtijd": format_rws_datetime(end)}}
        return requests.post(CFG["rws_url"], json=body, timeout=90)

    resp = request(True)
    if resp.status_code == 400:
        resp = request(False)              # filter on the response instead
    if resp.status_code == 204 or not resp.content:
        return np.array([]), np.array([])
    if not resp.ok:
        raise RuntimeError(f"RWS {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    pts = {}
    for wl in data.get("WaarnemingenLijst") or []:
        meta = wl.get("AquoMetadata") or {}
        pt = str(meta.get("ProcesType") or "").lower()
        if pt and pt != proces:
            continue
        hoed = ((meta.get("Hoedanigheid") or {}).get("Code") or "NAP").upper()
        if hoed != "NAP":
            continue
        unit = ((meta.get("Eenheid") or {}).get("Code") or "cm").lower()
        scale = 0.01 if unit == "cm" else 1.0
        for m in wl.get("MetingenLijst") or []:
            v = (m.get("Meetwaarde") or {}).get("Waarde_Numeriek")
            if v is None or abs(v) > 5000:
                continue
            pts[parse_iso_datetime(m["Tijdstip"]).timestamp()] = v * scale
    if not pts:
        return np.array([]), np.array([])
    t = np.array(sorted(pts))
    return t, np.array([pts[k] for k in t])


class WaterLevel:
    """Combines measured, forecast and astronomical water levels into one
    function level(t) with a source label, following a fallback chain."""

    def __init__(self, now, end, history_days=3):
        self.now = now
        self.meas = self.fc = self.astro = (np.array([]), np.array([]))
        self.location = None
        errors = []
        for loc in CFG["rws_locations"]:
            try:
                self.meas = rws_series(loc, now - dt.timedelta(days=history_days), now, "meting")
                self.fc = rws_series(loc, now - dt.timedelta(hours=6), end, "verwachting")
                self.astro = rws_series(loc, now - dt.timedelta(days=history_days), end,
                                        "astronomisch")
            except (requests.RequestException, RuntimeError, ValueError) as e:
                errors.append(f"{loc}: {e}")
                continue
            if len(self.meas[0]) or len(self.fc[0]):
                self.location = loc
                break
        if self.location is None:
            raise NoWaterLevelDataError(
                "No water level data from Rijkswaterstaat.\n  " + "\n  ".join(errors))
        log.info("Water level: %s (measured %d, forecast %d, astro %d points)",
                 self.location, len(self.meas[0]), len(self.fc[0]), len(self.astro[0]))
        self.surge_now = self._surge_now()
        self.harmonic = None

    def _surge_now(self):
        tm, vm = self.meas
        ta, va = self.astro
        if not len(tm) or not len(ta):
            return 0.0
        recent = tm > tm[-1] - 3600
        return float(np.mean(vm[recent] - np.interp(tm[recent], ta, va)))

    @staticmethod
    def _covers(series, t, tol=900):
        ts = series[0]
        return len(ts) > 1 and ts[0] - tol <= t <= ts[-1] + tol

    def level(self, t_epoch):
        """Returns (level m NAP, source)."""
        if self._covers(self.meas, t_epoch, tol=600):
            return float(np.interp(t_epoch, *self.meas)), "measured"
        if self._covers(self.fc, t_epoch):
            return float(np.interp(t_epoch, *self.fc)), "RWS forecast"
        if self._covers(self.astro, t_epoch):
            hrs = max(0.0, (t_epoch - self.now.timestamp()) / 3600)
            surge = self.surge_now * math.exp(-hrs / CFG["surge_decay_hours"])
            return float(np.interp(t_epoch, *self.astro) + surge), "tide + surge"
        return self._harmonic(t_epoch), "harmonic fit"

    def covers_until(self):
        """Last epoch any series can answer for without falling back to the
        harmonic fit. The window search runs over the week-long wind forecast,
        which reaches well past the tide data, and the harmonic fallback exits
        the program when utide isn't installed - so callers clamp to this."""
        ends = [s[0][-1] for s in (self.meas, self.fc, self.astro) if len(s[0])]
        return max(ends) if ends else self.now.timestamp()

    def low_water_datum(self):
        """A low-water reference level (near MLWS) from the astronomical tide.

        Used to decide which water is permanently submerged sea rather than
        intertidal beach. It must NOT be the lowest level in the run window:
        on a short run at high tide that sits far above real low water, and a
        wide band of permanently-submerged shoreface then gets counted as
        "intertidal beach" - ~250 ha of it, which is the same mistake as
        counting the open sea, one step further out."""
        t, v = self.astro if len(self.astro[0]) else self.meas
        if not len(t):
            return CFG["sea_seed_level"]
        return float(np.percentile(v, 2))

    def extremes(self, start, end, max_n=4):
        """Upcoming high and low waters as [{t, level, kind, source}].

        No extra API call: the forecast and astronomical series are already
        fetched, so the turning points are just sign changes in their slope.
        Prefers the RWS forecast (it includes surge) and falls back to the
        astronomical tide, which reaches further ahead."""
        half = 3 * 3600                    # half a tidal cycle is ~6.2 h
        for name, (t, v) in (("RWS forecast", self.fc), ("astronomical", self.astro)):
            sel = (t >= start) & (t <= end) if len(t) else np.zeros(0, bool)
            if sel.sum() < 12:
                continue
            ts, vs = t[sel], v[sel]
            # A turning point is the extreme value across half a tidal cycle,
            # not merely a sign change in the slope. 10-minute data wiggles
            # enough to produce dozens of spurious sign changes - which is how
            # this once reported two consecutive high waters, and two
            # consecutive lows, in the same four-row table.
            out = []
            for i in range(len(ts)):
                if ts[i] - half < ts[0] or ts[i] + half > ts[-1]:
                    continue               # truncated window: can't judge yet
                win = (ts >= ts[i] - half) & (ts <= ts[i] + half)
                if vs[i] >= vs[win].max():
                    kind = "high"
                elif vs[i] <= vs[win].min():
                    kind = "low"
                else:
                    continue
                # highs and lows must alternate, and sit at least half a cycle
                # apart; anything else is the same turning point twice
                if out and (ts[i] - out[-1]["t"] < half or out[-1]["kind"] == kind):
                    continue
                out.append({"t": float(ts[i]), "level": float(vs[i]), "kind": kind,
                            "source": name})
                if len(out) >= max_n:
                    break
            if out:
                return out
        return []

    def _harmonic(self, t_epoch):
        if self.harmonic is None:
            try:
                import utide
            except ImportError:
                raise HarmonicFitUnavailableError(
                    "No forecast or tide data available and utide is not installed "
                    "(pip install utide).") from None
            tm, vm = rws_series(self.location, self.now - dt.timedelta(days=35), self.now,
                                "meting")
            if len(tm) < 1000:
                raise HarmonicFitUnavailableError(
                    "Not enough measured water levels for a harmonic fit.")
            days = tm / 86400.0
            coef = utide.solve(days, vm, lat=52.05, method="ols", conf_int="none",
                               verbose=False)
            self.harmonic = (utide, coef)
        utide, coef = self.harmonic
        return float(utide.reconstruct(np.array([t_epoch / 86400.0]), coef,
                                       verbose=False).h[0])
