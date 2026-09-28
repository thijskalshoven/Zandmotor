"""CDSE (Copernicus Data Space Ecosystem) authentication."""

from __future__ import annotations

import os

import requests

from zandmotor.config import CFG


def cdse_token():
    cid, secret = os.environ.get("CDSE_CLIENT_ID"), os.environ.get("CDSE_CLIENT_SECRET")
    if not cid or not secret:
        raise RuntimeError("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET for --calibrate "
                           "(see top of file).")
    r = requests.post(CFG["cdse_token_url"], data={"grant_type": "client_credentials",
                                                  "client_id": cid, "client_secret": secret},
                      timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]
