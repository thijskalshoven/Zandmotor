"""Tagged dataclasses for the lagoon response model, replacing the
polymorphic {"mode": "tidal"|"wind"|"lake", ...} dict fit_lagoon_response
used to return. Each type knows how to compute its own area_at(...), so
callers no longer need to branch on a mode string.

to_dict()/from_dict() keep the on-disk cache/lagoon_response.json format
unchanged, so an existing cache (potentially the product of a year of
satellite calibration) keeps loading without forcing a re-`--calibrate`.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class TidalResponse:
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

    MODE = "tidal"

    @property
    def mode(self):
        return self.MODE

    def area_at(self, sea_level_fn, te, wind_hist=None, lake_area=None):
        sea_lag, _ = sea_level_fn(te - self.delay_min * 60)
        area = self.damping * sea_lag + self.offset
        return max(area, self.floor_ha)

    def to_dict(self):
        return {"mode": self.mode, **dataclasses.asdict(self)}

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})


@dataclasses.dataclass
class WindResponse:
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

    MODE = "wind"

    @property
    def mode(self):
        return self.MODE

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

    def to_dict(self):
        return {"mode": self.mode, **dataclasses.asdict(self)}

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})


@dataclasses.dataclass
class LakeResponse:
    r: float
    wind_r: float
    n: int
    fitted: str
    times: list
    areas: list
    n_optical: int
    n_sar: int

    MODE = "lake"

    @property
    def mode(self):
        return self.MODE

    def area_at(self, sea_level_fn, te, wind_hist=None, lake_area=None):
        if lake_area is not None:
            return lake_area
        return float(np.interp(te, np.array(self.times), np.array(self.areas)))

    def to_dict(self):
        return {"mode": self.mode, **dataclasses.asdict(self)}

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})


_BY_MODE = {"tidal": TidalResponse, "wind": WindResponse, "lake": LakeResponse}


def model_from_dict(d):
    """Reconstructs the right dataclass from a decoded lagoon_response.json."""
    return _BY_MODE[d["mode"]].from_dict(d)
