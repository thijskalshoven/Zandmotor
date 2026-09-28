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
