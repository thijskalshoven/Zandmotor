"""Open-Meteo wind/marine data, daylight, wetsuit advice, and compass helpers."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import requests

from zandmotor.config import CFG, UTC, log


def fetch_wind(lat, lon, forecast_days=7, past_days=2):
    """Hourly wind, air temperature and rain, and the daily sunrise/sunset
    pair, in one request. The week-long horizon is what the "next rideable
    window" search needs; daylight keeps it from offering sessions at 03:00.
    past_days reaches back far enough to total the rain since the newest
    lagoon image (the "may have filled since" flag).
    Returns (times, speed, gust, direction, air_temp, daylight, rain_mm) where
    daylight is a list of (sunrise_epoch, sunset_epoch)."""
    params = {"latitude": lat, "longitude": lon,
              "hourly": "wind_speed_10m,wind_gusts_10m,wind_direction_10m,temperature_2m,"
                        "precipitation",
              "daily": "sunrise,sunset",
              "wind_speed_unit": "kn", "timezone": "UTC", "past_days": int(past_days),
              "forecast_days": forecast_days}
    try:
        r = requests.get(CFG["open_meteo_forecast_url"], params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        h = j["hourly"]
        t = np.array([dt.datetime.fromisoformat(s).replace(tzinfo=UTC).timestamp()
                      for s in h["time"]])
        day = j.get("daily") or {}
        daylight = [(dt.datetime.fromisoformat(a).replace(tzinfo=UTC).timestamp(),
                     dt.datetime.fromisoformat(b).replace(tzinfo=UTC).timestamp())
                    for a, b in zip(day.get("sunrise") or [], day.get("sunset") or [])]
        rain = np.array([np.nan if v is None else v for v in h.get("precipitation") or
                         [None] * len(t)], float)

        def col(name):
            return np.array([np.nan if v is None else v for v in h[name]], float)
        spd, gst, dirn = col("wind_speed_10m"), col("wind_gusts_10m"), col("wind_direction_10m")
        # the last forecast hours sometimes come back empty; interpolated, a
        # missing value turned the kite advice and the windows into "nan kn"
        ok = np.isfinite(spd) & np.isfinite(gst) & np.isfinite(dirn)
        if not ok.any():
            raise ValueError("no wind values returned")
        return (t[ok], spd[ok], gst[ok], dirn[ok], col("temperature_2m")[ok], daylight,
                rain[ok])
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
        r = requests.get(CFG["open_meteo_marine_url"], params={
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


def fetch_waves(lat, lon, start=None, end=None, past_days=None, forecast_days=None):
    """Hourly waves from the Open-Meteo marine API: a date range (start/end,
    datetimes) for --calibrate, or past_days/forecast_days for a report. The
    API picks its nearest sea cell (~7 km offshore here), the same one for
    history and forecast, so the fitted wave term carries over. Returns
    (times, height_m, period_s, from_deg) or None - fill-and-drain then runs
    on still water."""
    params = {"latitude": lat, "longitude": lon, "timezone": "UTC",
              "hourly": "wave_height,wave_period,wave_direction"}
    if start is not None:
        params.update(start_date=f"{start:%Y-%m-%d}", end_date=f"{end:%Y-%m-%d}")
    else:
        params.update(past_days=past_days or 0, forecast_days=forecast_days or 7)
    try:
        r = requests.get(CFG["open_meteo_marine_url"], params=params, timeout=60)
        r.raise_for_status()
        h = r.json()["hourly"]
        ok = [i for i, x in enumerate(h["wave_height"]) if x is not None]
        if not ok:
            raise ValueError("no wave heights returned")

        def col(name):
            return np.array([np.nan if h[name][i] is None else h[name][i] for i in ok], float)
        return (np.array([dt.datetime.fromisoformat(h["time"][i]).replace(tzinfo=UTC).timestamp()
                          for i in ok]),
                col("wave_height"), col("wave_period"), col("wave_direction"))
    except (requests.RequestException, KeyError, ValueError) as e:
        log.warning("No wave heights (%s); the lagoon's surge fill uses still water only.", e)
        return None


def waves_at(waves, t_epoch):
    """Wave height (m) at t, or 0.0 without data (still water)."""
    if waves is None:
        return 0.0 if np.isscalar(t_epoch) else np.zeros(len(t_epoch))
    return np.interp(t_epoch, waves[0], waves[1])


def wave_forcing(waves, t_epoch, form):
    """The wave term of the flood level (flood.effective_level) at t, in one
    of four forms the --calibrate fit chooses between:
      height      significant wave height (m)
      height_dir  height x the onshore share of the wave direction
      runup       sqrt(height x deep-water wavelength) - the scaling of beach
                  run-up (Stockdon et al. 2006): long-period swell runs much
                  further up than a short chop of the same height
      runup_dir   runup x the onshore share
    The onshore share is cos(direction - shore_normal_deg), 0 for waves
    running along or off the coast. 0 without data (still water)."""
    if waves is None:
        return 0.0 if np.isscalar(t_epoch) else np.zeros(len(t_epoch))
    t, hs, tp, wd = waves
    h = np.interp(t_epoch, t, np.nan_to_num(hs))
    if form in ("runup", "runup_dir"):
        wavelength = 9.81 * np.interp(t_epoch, t, np.nan_to_num(tp)) ** 2 / (2 * np.pi)
        h = np.sqrt(h * wavelength)
    if form.endswith("_dir"):
        # interpolate the direction as a vector, so 350 and 10 degrees average to 0
        rad = np.radians(np.nan_to_num(wd))
        ang = np.arctan2(np.interp(t_epoch, t, np.sin(rad)), np.interp(t_epoch, t, np.cos(rad)))
        h = h * np.clip(np.cos(ang - np.radians(CFG["shore_normal_deg"])), 0.0, None)
    return h


def rain_sum(rain, t_epoch, hours):
    """Rain (mm) in the `hours` before each t, from an hourly (times, mm)
    series; 0 without data."""
    if rain is None:
        return 0.0 if np.isscalar(t_epoch) else np.zeros(len(t_epoch))
    t, mm = rain
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(mm))])
    return (csum[np.searchsorted(t, t_epoch, side="right")]
            - csum[np.searchsorted(t, np.asarray(t_epoch) - hours * 3600, side="right")])


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
    """Hourly wind speed/gust and rain archive (not forecast), for gating SAR
    scenes and for storing the weather context of each satellite observation.
    The archive lags a few days behind real time; recent hours come back NaN.
    Returns (times, speed, gust, rain_mm)."""
    r = requests.get(CFG["open_meteo_archive_url"], params={
        "latitude": lat, "longitude": lon,
        "start_date": start.date().isoformat(), "end_date": end.date().isoformat(),
        "hourly": "wind_speed_10m,wind_gusts_10m,precipitation", "wind_speed_unit": "kn",
        "timezone": "UTC"}, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    t = np.array([dt.datetime.fromisoformat(s).replace(tzinfo=UTC).timestamp() for s in h["time"]])

    def arr(name):
        return np.array([np.nan if v is None else v for v in h[name]], float)
    return t, arr("wind_speed_10m"), arr("wind_gusts_10m"), arr("precipitation")


def max_gust_window(t_arr, gst_arr, te, hours):
    """Peak gust in the `hours` before te, NaN without data. Used to gate SAR
    scenes (rough water reads as land) and as the candidate feature of the
    wind test in --calibrate. That test does NOT currently hold: it looked
    strong at n=20 (r=0.60) and collapsed at n=64 (r=0.11) - see the module
    docstring of zandmotor_lagoon.py. Direction is left out because the
    onshore component correlated worse than plain magnitude."""
    mask = (t_arr >= te - hours * 3600) & (t_arr <= te) & np.isfinite(gst_arr)
    return float(gst_arr[mask].max()) if mask.any() else np.nan


def sum_window(t_arr, v_arr, t0, t1):
    """Sum of v over t0 < t <= t1, ignoring NaN; NaN if nothing is covered."""
    mask = (t_arr > t0) & (t_arr <= t1) & np.isfinite(v_arr)
    return float(v_arr[mask].sum()) if mask.any() else np.nan
