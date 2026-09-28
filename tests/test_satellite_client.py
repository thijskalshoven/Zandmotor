import datetime as dt

from zandmotor.satellite.client import dedupe_scenes_one_per_day


def _t(iso):
    return dt.datetime.fromisoformat(iso)


def test_dedupe_keeps_newest_per_day_newest_first():
    times = [
        _t("2026-01-01T08:00:00+00:00"),
        _t("2026-01-01T14:00:00+00:00"),  # same day, later - should win
        _t("2026-01-03T09:00:00+00:00"),
        _t("2026-01-02T09:00:00+00:00"),
    ]
    out = dedupe_scenes_one_per_day(times)
    assert [t.date().isoformat() for t in out] == ["2026-01-03", "2026-01-02", "2026-01-01"]
    assert out[2] == _t("2026-01-01T14:00:00+00:00")


def test_dedupe_empty():
    assert dedupe_scenes_one_per_day([]) == []


def test_dedupe_accepts_a_set():
    times = {_t("2026-01-01T08:00:00+00:00"), _t("2026-01-01T14:00:00+00:00")}
    out = dedupe_scenes_one_per_day(times)
    assert len(out) == 1
    assert out[0] == _t("2026-01-01T14:00:00+00:00")
