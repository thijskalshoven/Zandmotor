"""HTTP helper shared by the RWS and CDSE clients."""

from __future__ import annotations

import threading

import requests


def post_with_deadline(url, deadline_s, connect_s=15, idle_s=30, **kw):
    """POST and get the whole response within deadline_s seconds, however the
    server misbehaves.

    requests' own timeout only bounds each wait for the next bytes, so a
    server that stalls or trickles its answer can hold one request open far
    longer: an RWS request ran for over half an hour and two --calibrate runs
    sat idle on a Sentinel Hub socket for 30 minutes. Checking the clock
    between chunks isn't enough either, because a single chunk read can
    block just as long. So the request runs in a daemon thread and the caller
    stops waiting at the deadline; an abandoned thread ends with its socket
    timeout or with the process.
    Returns (status_code, content_bytes); raises requests.Timeout when the
    deadline passes, or whatever the request itself raised."""
    box = {}

    def work():
        try:
            r = requests.post(url, timeout=(connect_s, idle_s), **kw)
            box["result"] = (r.status_code, r.content)
        except BaseException as e:          # handed to the caller below
            box["error"] = e

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(deadline_s)
    if worker.is_alive():
        raise requests.Timeout(f"no complete response from {url} after {deadline_s:.0f} s")
    if "error" in box:
        raise box["error"]
    return box["result"]
