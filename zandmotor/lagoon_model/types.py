"""Tagged dataclasses for the lagoon response model, replacing the
polymorphic {"mode": "tidal"|"wind"|"lake", ...} dict fit_lagoon_response
used to return. Each type knows how to compute its own area_at(...), so
callers no longer need to branch on a mode string.

to_dict()/from_dict() keep the on-disk cache/lagoon_response.json format, and
from_dict() accepts caches written before a field existed (it takes that
field's default), so an older cache keeps loading without a re-`--calibrate`.

Every type carries the observation record itself: `times`/`areas` (the
original columns) plus `obs`, one dict per observation with the context
needed to re-test the model offline - sensor, track, seen fraction, sea
level, gust and rain at that moment. Before `obs` existed, every re-analysis
had to refetch a year of tide and weather data first. `sar_tracks` records
the per-track SAR threshold and how well it agreed with optical.
"""

from __future__ import annotations

import dataclasses

import numpy as np


class _Serializable:
    """Shared to_dict/from_dict/mode for the response dataclasses."""

    @property
    def mode(self):
        return self.MODE

    def to_dict(self):
        return {"mode": self.mode, **dataclasses.asdict(self)}

    @classmethod
    def from_dict(cls, d):
        kw = {}
        for f in dataclasses.fields(cls):
            if f.name in d:
                kw[f.name] = d[f.name]
            elif (f.default is dataclasses.MISSING
                  and f.default_factory is dataclasses.MISSING):
                raise KeyError(f"lagoon response cache lacks required field {f.name!r}")
        return cls(**kw)


@dataclasses.dataclass
class TidalResponse(_Serializable):
    delay_min: int
    damping: float
    offset: float
    floor_ha: float
    r: float
    n: int
    rws_location: str
    fitted: str
    times: list
    areas: list
    n_optical: int
    n_sar: int
    obs: list = dataclasses.field(default_factory=list)
    sar_tracks: dict = dataclasses.field(default_factory=dict)

    MODE = "tidal"

    def area_at(self, sea_level_fn, te, wind_hist=None, lake_area=None):
        sea_lag, _ = sea_level_fn(te - self.delay_min * 60)
        area = self.damping * sea_lag + self.offset
        return max(area, self.floor_ha)


@dataclasses.dataclass
class WindResponse(_Serializable):
    window_h: int
    slope: float
    offset: float
    area_min_ha: float
    area_max_ha: float
    r: float
    sea_r: float
    n: int
    fitted: str
    times: list
    areas: list
    n_optical: int
    n_sar: int
    obs: list = dataclasses.field(default_factory=list)
    sar_tracks: dict = dataclasses.field(default_factory=dict)

    MODE = "wind"

    def area_at(self, sea_level_fn, te, wind_hist=None, lake_area=None):
        from zandmotor.wind import max_gust_window
        if wind_hist is not None:
            gust = max_gust_window(wind_hist[0], wind_hist[1], te, self.window_h)
            if np.isfinite(gust):
                area = self.slope * gust + self.offset
                return float(np.clip(area, self.area_min_ha, self.area_max_ha))
        if lake_area is not None:
            return lake_area
        return float(np.interp(te, np.array(self.times), np.array(self.areas)))


@dataclasses.dataclass
class LakeResponse(_Serializable):
    r: float
    wind_r: float
    n: int
    fitted: str
    times: list
    areas: list
    n_optical: int
    n_sar: int
    obs: list = dataclasses.field(default_factory=list)
    sar_tracks: dict = dataclasses.field(default_factory=dict)

    MODE = "lake"

    def area_at(self, sea_level_fn, te, wind_hist=None, lake_area=None):
        if lake_area is not None:
            return lake_area
        return float(np.interp(te, np.array(self.times), np.array(self.areas)))


_BY_MODE = {"tidal": TidalResponse, "wind": WindResponse, "lake": LakeResponse}


def model_from_dict(d):
    """Reconstructs the right dataclass from a decoded lagoon_response.json."""
    return _BY_MODE[d["mode"]].from_dict(d)
