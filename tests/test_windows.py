from zandmotor.windows import format_window


def test_format_window_same_day():
    win = {"start": 1_700_000_000, "end": 1_700_003_600, "kite": "Bandit",
           "size": 14, "mean_kn": 12.0, "gust_kn": 18.0, "dir_deg": 90.0}
    text = format_window(win, step_min=60)
    assert "Bandit" in text
    assert "14" in text
    assert "12 kn" in text


def test_format_window_no_kite_fits():
    win = {"start": 1_700_000_000, "end": 1_700_000_000, "kite": None,
           "size": None, "mean_kn": 30.0, "gust_kn": 40.0, "dir_deg": 0.0}
    text = format_window(win, step_min=60)
    assert "no kite fits" in text


class _FixedLagoon:
    """Stands in for LagoonWater: one constant area."""
    def __init__(self, ha):
        import numpy as np
        self.ha, self.mask = ha, np.zeros((2, 2), bool)
        self.unknown = False

    def at(self, te):
        return self.mask, self.ha, self.ha

    def range_at(self, te, area_ha=None):
        return None


def _wind(hours, spd, gst, deg=270.0):
    import numpy as np
    t = np.arange(hours) * 3600.0 + 1_700_000_000
    n = len(t)
    return (t, np.full(n, spd), np.full(n, gst), np.full(n, deg), np.full(n, 15.0), [],
            np.zeros(n))


def test_find_windows_follows_wind_not_tide():
    import datetime as dt
    from zandmotor.config import UTC
    from zandmotor.windows import find_windows
    now = dt.datetime.fromtimestamp(1_700_000_000, UTC)
    wind = _wind(6, 15.0, 18.0)
    # enough lagoon water: one window over the whole (daylight-less) horizon
    assert len(find_windows(now, _FixedLagoon(4.0), wind, [], float("inf"))) == 1
    # too little lagoon water: no window, whatever the wind
    assert find_windows(now, _FixedLagoon(1.0), wind, [], float("inf")) == []
