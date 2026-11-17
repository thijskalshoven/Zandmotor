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


def test_dedupe_with_key_keeps_records():
    class Scene:
        def __init__(self, t, orbit):
            self.t, self.orbit = t, orbit
    scenes = [Scene(_t("2026-09-29T05:57:00+00:00"), 110),
              Scene(_t("2026-09-30T05:49:00+00:00"), 37),
              Scene(_t("2026-09-30T05:57:00+00:00"), 110)]
    out = dedupe_scenes_one_per_day(scenes, key=lambda s: s.t)
    assert [s.orbit for s in out] == [110, 110]


class _FakeToken:
    def __init__(self):
        self.refreshes = 0

    def bearer(self):
        return f"tok{self.refreshes}"

    def refresh(self):
        self.refreshes += 1


def test_post_renews_an_expired_token_and_retries(monkeypatch):
    from zandmotor.satellite import client
    seen = []

    def fake(url, deadline_s, headers=None, **kw):
        seen.append(headers["Authorization"])
        return (401, b"") if len(seen) == 1 else (200, b"ok")
    monkeypatch.setattr(client, "post_with_deadline", fake)
    tok = _FakeToken()
    assert client._post(tok, "https://x") == b"ok"
    assert tok.refreshes == 1 and seen == ["Bearer tok0", "Bearer tok1"]


def test_post_gives_up_after_repeated_401(monkeypatch):
    import pytest
    import requests
    from zandmotor.satellite import client
    monkeypatch.setattr(client, "post_with_deadline", lambda url, d, headers=None, **kw: (401, b""))
    with pytest.raises(requests.HTTPError):
        client._post(_FakeToken(), "https://x")


def test_post_retries_a_stalled_request(monkeypatch):
    import requests
    from zandmotor.satellite import client
    calls = []

    def fake(url, deadline_s, headers=None, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise requests.Timeout("stalled")
        return 200, b"ok"
    monkeypatch.setattr(client, "post_with_deadline", fake)
    monkeypatch.setattr(client.time, "sleep", lambda s: None)
    assert client._post(_FakeToken(), "https://x") == b"ok" and len(calls) == 2
