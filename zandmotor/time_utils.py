"""Datetime helpers shared by the RWS, wind, and satellite fetchers."""

from __future__ import annotations

import datetime as dt

from zandmotor.config import UTC


def format_rws_datetime(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000+00:00")


def parse_iso_datetime(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)
