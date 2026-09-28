"""Open-Meteo wind/marine data, daylight, wetsuit advice, and compass helpers."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import requests

from zandmotor.config import CFG, UTC, log


def fetch_wind(lat, lon, forecast_days=7):
    """Hourly wind plus air temperature, and the daily sunrise/sunset pair, in
    one request. The week-long horizon is what the "next rideable window"
    search needs; daylight keeps it from offering sessions at 03:00.
    Returns (times, speed, gust, direction, air_temp, daylight) where daylight
    is a list of (sunrise_epoch, sunset_epoch)."""
    params = {"latitude": lat, "longitude": lon,
              "hourly": "wind_speed_10m,wind_gusts_10m,wind_direction_10m,temperature_2m",
              "daily": "sunrise,sunset",
              "wind_speed_unit": "kn", "timezone": "UTC", "past_days": 2,
              "forecast_days": forecast_days}
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        h = j["hourly"]
        t = np.array([dt.datetime.fromisoformat(s).replace(tzinfo=UTC).timestamp()
                      for s in h["time"]])
        day = j.get("daily") or {}
        daylight = [(dt.datetime.fromisoformat(a).replace(tzinfo=UTC).timestamp(),
                     dt.datetime.fromisoformat(b).replace(tzinfo=UTC).timestamp())
                    for a, b in zip(day.get("sunrise") or [], day.get("sunset") or [])]
        return (t, np.array(h["wind_speed_10m"], float), np.array(h["wind_gusts_10m"], float),
                np.array(h["wind_direction_10m"], float),
                np.array(h["temperature_2m"], float), daylight)
    except (requests.RequestException, KeyError, ValueError) as e:
        log.warning("No wind data (%s); the map still works without it.", e)
        return None


def wind_at(wind, t_epoch):
    """Wind interpolated to t_epoch, not snapped to the nearest hour - with
    --step 30 snapping gave consecutive frames identical wind. Direction is
    interpolated through its sin/cos components so 350 deg -> 10 deg crosses
    north instead of sweeping the long way round through south.
    Returns (speed, gust, direction, air_temp_c)."""
    if wind is None:
        return None
    t, spd, gst, dirn, air = wind[0], wind[1], wind[2], wind[3], wind[4]
    rad = np.radians(dirn)
    d = math.degrees(math.atan2(float(np.interp(t_epoch, t, np.sin(rad))),
                                float(np.interp(t_epoch, t, np.cos(rad))))) % 360
    return (float(np.interp(t_epoch, t, spd)), float(np.interp(t_epoch, t, gst)), d,
            float(np.interp(t_epoch, t, air)))


def is_daylight(daylight, t_epoch):
    """True if t_epoch falls between a sunrise and its sunset. No daylight
    data -> True, so a missing field never silently hides every window."""
    if not daylight:
        return True
    return any(a <= t_epoch <= b for a, b in daylight)


def fetch_sea_temp(lat, lon):
    """Hourly sea-surface temperature (Open-Meteo marine API, free, no key).
    Returns (times, temps_c) or None - a missing wetsuit line is a cosmetic
    loss, so this never aborts the run."""
    try:
        r = requests.get("https://marine-api.open-meteo.com/v1/marine", params={
            "latitude": lat, "longitude": lon, "hourly": "sea_surface_temperature",
            "timezone": "UTC", "past_days": 1, "forecast_days": 7}, timeout=30)
        r.raise_for_status()
        h = r.json()["hourly"]
        t, v = h["time"], h["sea_surface_temperature"]
        ok = [(s, x) for s, x in zip(t, v) if x is not None]
        if not ok:
            raise ValueError("no sea surface temperature values returned")
        return (np.array([dt.datetime.fromisoformat(s).replace(tzinfo=UTC).timestamp()
                          for s, _ in ok]),
                np.array([x for _, x in ok], float))
    except (requests.RequestException, KeyError, ValueError) as e:
        log.warning("No sea temperature (%s); the wetsuit advice is skipped.", e)
        return None


def sea_temp_at(sea_temp, t_epoch):
    if sea_temp is None:
        return None
    return float(np.interp(t_epoch, sea_temp[0], sea_temp[1]))


def wetsuit_advice(water_c, w):
    """Which wetsuit, from the water temperature. A cold hard wind bumps the
    call one step warmer-dressed: standing about soaked in 18 kn of 10 C air
    is the part that ends sessions, not the water itself.
    Returns (call, note) or (None, None) with no data."""
    if water_c is None:
        return None, None
    table = CFG["wetsuit"]
    idx = next((i for i, (lo, hi, _) in enumerate(table) if lo <= water_c < hi), len(table) - 1)
    note = None
    if w is not None and idx > 0:
        spd, air = w[0], w[3]
        if air <= water_c - CFG["wetsuit_chill_air_delta_c"] and spd >= CFG["wetsuit_chill_wind_kn"]:
            idx -= 1
            note = (f"one step warmer than {water_c:.0f} C water alone suggests: "
                    f"{air:.0f} C air in {spd:.0f} kn")
    return table[idx][2], note


def compass(deg):
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int((deg % 360) / 22.5 + 0.5) % 16]


def shore_class(deg):
    diff = abs((deg - CFG["shore_normal_deg"] + 180) % 360 - 180)
    if diff <= 45:
        return "onshore"
    if diff <= 100:
        return "side-shore"
    return "offshore"


def fetch_wind_history(lat, lon, start, end):
    """Hourly wind speed/gust archive (not forecast) for fitting against past
    satellite observations."""
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon,
        "start_date": start.date().isoformat(), "end_date": end.date().isoformat(),
        "hourly": "wind_speed_10m,wind_gusts_10m", "wind_speed_unit": "kn",
        "timezone": "UTC"}, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    t = np.array([dt.datetime.fromisoformat(s).replace(tzinfo=UTC).timestamp() for s in h["time"]])
    return t, np.array(h["wind_speed_10m"], float), np.array(h["wind_gusts_10m"], float)


def max_gust_window(t_arr, gst_arr, te, hours):
    """Peak gust in the `hours` before te. Used as the wind feature: a year
    of Zandmotor images showed this correlates with lagoon area (probably
    intermittent wave overtopping across the low sand berm) much better than
    sea level does - and better than any onshore/offshore direction split,
    so direction is deliberately not part of this."""
    mask = (t_arr >= te - hours * 3600) & (t_arr <= te)
    return float(gst_arr[mask].max()) if mask.any() else np.nan
