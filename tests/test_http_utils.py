import time

import pytest
import requests

from zandmotor.http_utils import post_with_deadline


class _R:
    status_code, content = 200, b"ok"


def test_returns_status_and_body(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda url, **kw: _R())
    assert post_with_deadline("https://x", 5) == (200, b"ok")


def test_a_stalled_server_cannot_outlast_the_deadline(monkeypatch):
    def stalls(url, **kw):
        time.sleep(5)                       # a server that never finishes
        return _R()
    monkeypatch.setattr(requests, "post", stalls)
    t0 = time.monotonic()
    with pytest.raises(requests.Timeout):
        post_with_deadline("https://x", 0.2)
    assert time.monotonic() - t0 < 2


def test_request_errors_reach_the_caller(monkeypatch):
    def fails(url, **kw):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(requests, "post", fails)
    with pytest.raises(requests.ConnectionError):
        post_with_deadline("https://x", 5)
