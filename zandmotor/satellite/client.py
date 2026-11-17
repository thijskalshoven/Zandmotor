"""CDSE (Copernicus Data Space Ecosystem) authentication and the request
helpers shared by the optical (Sentinel-2) and SAR (Sentinel-1) modules."""

from __future__ import annotations

import datetime as dt
import json
import os
import time

import requests
from rasterio.io import MemoryFile

from zandmotor.config import CFG, log
from zandmotor.http_utils import post_with_deadline
from zandmotor.time_utils import format_rws_datetime


class CdseToken:
    """A CDSE access token that renews itself.

    The token lives ~10 minutes. A full calibration walks a year of optical
    and radar scenes and can take far longer (or the laptop sleeps mid-run),
    and a plain string token then turns every later request into a 401. That
    happened: 9 images in, every request failed and the run fitted a model
    on what it had. Every request asks bearer() and gets a fresh token when
    the old one is near expiry, or after a 401 via refresh()."""

    def __init__(self):
        self.cid = os.environ.get("CDSE_CLIENT_ID")
        self.secret = os.environ.get("CDSE_CLIENT_SECRET")
        if not self.cid or not self.secret:
            raise RuntimeError("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET for --calibrate "
                               "(see \"Satellite data\" in the README).")
        self._token, self._expires = None, 0.0
        self.refresh()

    def refresh(self):
        r = requests.post(CFG["cdse_token_url"],
                          data={"grant_type": "client_credentials",
                                "client_id": self.cid, "client_secret": self.secret},
                          timeout=30)
        r.raise_for_status()
        j = r.json()
        self._token = j["access_token"]
        # renew a minute early, so a request never leaves with a dying token
        self._expires = time.monotonic() + float(j.get("expires_in", 600)) - 60

    def bearer(self):
        if time.monotonic() >= self._expires:
            self.refresh()
        return self._token


def cdse_token():
    return CdseToken()


def _post(token, url, attempts=3, **kw):
    """POST with the token's current bearer and return the response body.

    Renews the token on a 401 and backs off on timeouts, connection drops,
    429 and 5xx. Each attempt must complete within cdse_request_deadline_s
    (see post_with_deadline: stalled requests used to sit for the full
    120 s read timeout). Raises the last error."""
    for i in range(attempts):
        last = i == attempts - 1
        try:
            status, content = post_with_deadline(
                url, CFG["cdse_request_deadline_s"],
                headers={"Authorization": f"Bearer {token.bearer()}"}, **kw)
        except (requests.Timeout, requests.ConnectionError):
            if last:
                raise
            time.sleep(5 * (i + 1))
            continue
        if status == 401 and not last:
            token.refresh()
            continue
        if (status == 429 or status >= 500) and not last:
            time.sleep(5 * (i + 1))
            continue
        if status >= 400:
            raise requests.HTTPError(f"{status} from {url}: "
                                     f"{content[:200].decode('utf-8', 'replace')}")
        return content
    raise RuntimeError("unreachable")


def catalog_search(token, collection, start, end, cql_filter, bbox=None, chunk_days=45):
    """All catalogue features for `collection` between start and end.

    Searched in chunks because one request returns at most 100 features. The
    chunking used to live only in the SAR listing; the optical listing made one
    request, which was fine with a strict tile-cloud filter but silently
    truncates once that filter is loosened (a year at <95% cloud is ~200
    Sentinel-2 granules here, two tiles overlap the bbox)."""
    body = {"bbox": list(bbox or CFG["bbox_wgs84"]), "collections": [collection],
            "limit": 100, "filter": cql_filter, "filter-lang": "cql2-json"}
    feats = []
    cursor_end = end
    while cursor_end > start:
        chunk_start = max(cursor_end - dt.timedelta(days=chunk_days), start)
        body["datetime"] = (f"{format_rws_datetime(chunk_start)[:19]}Z/"
                            f"{format_rws_datetime(cursor_end)[:19]}Z")
        page = json.loads(_post(token, CFG["cdse_catalog_url"], json=body))["features"]
        if len(page) >= body["limit"]:
            log.warning("Catalogue search for %s hit the %d-result cap between %s and %s; "
                        "some scenes may be missing (lower chunk_days)", collection,
                        body["limit"], chunk_start.date(), cursor_end.date())
        feats += page
        cursor_end = chunk_start
    return feats


def dedupe_scenes_one_per_day(items, key=None):
    """Keep only the newest item per calendar day, newest day first.

    `items` are datetimes, or records whose datetime `key` extracts (the SAR
    listing keeps the relative orbit with each scene). Shared by cdse_scenes
    and cdse_sar_scenes."""
    key = key or (lambda x: x)
    seen, out = set(), []
    for item in sorted(items, key=key, reverse=True):
        day = key(item).date()
        if day not in seen:
            seen.add(day)
            out.append(item)
    return out


def process_image(token, grid, t, *, collection, evalscript, extra_data_filter=None,
                  time_window=None):
    """Sentinel Hub Process API request for `collection` on `grid`, decoded
    to a (bands, H, W) float32 array. Shared by cdse_image (Sentinel-2) and
    cdse_sar_image (Sentinel-1).

    By default it asks for the whole UTC day's mosaic. `time_window` (a
    timedelta) narrows that to t +/- window, so a request returns exactly one
    satellite pass - which matters for SAR, where two tracks look at the
    lagoon from different angles and must not be mosaicked together."""
    if time_window is None:
        a = t.replace(hour=0, minute=0, second=0, microsecond=0)
        b = a + dt.timedelta(days=1)
    else:
        a, b = t - time_window, t + time_window
    data_filter = {"timeRange": {"from": a.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "to": b.strftime("%Y-%m-%dT%H:%M:%SZ")}}
    if extra_data_filter:
        data_filter.update(extra_data_filter)
    body = {"input": {"bounds": {"bbox": list(grid.bounds_3857),
                                 "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"}},
                      "data": [{"type": collection, "dataFilter": data_filter}]},
            "output": {"width": grid.width, "height": grid.height,
                       "responses": [{"identifier": "default",
                                      "format": {"type": "image/tiff"}}]},
            "evalscript": evalscript}
    content = _post(token, CFG["cdse_process_url"], json=body)
    with MemoryFile(content) as mem, mem.open() as src:
        return src.read().astype("float32")
