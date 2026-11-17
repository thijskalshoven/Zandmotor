"""Consumes the fitted lagoon response model - runs on every invocation."""

from __future__ import annotations

import hashlib
import heapq
import json

import numpy as np
from scipy import ndimage

from zandmotor.config import CFG
from zandmotor.flood import largest_patch_ha, remove_small_patches


def observation_records(model):
    """The model's observations as dicts. Caches written before per-observation
    context existed only have the times/areas columns; those come back without
    sensor or track."""
    if model.obs:
        return list(model.obs)
    return [{"t": float(t), "area_ha": float(a), "sensor": None, "track": None}
            for t, a in zip(model.times, model.areas)]


def latest_observation(model, now):
    """The newest reliable observation at or before `now`, or None.

    This, not a median of recent images, is what lake mode shows. The median
    mixed two radar tracks that read ~2 ha apart and smeared the lagoon's
    real behaviour (a ~2.5-4.5 ha resting state with occasional fills to
    ~13 ha). Every observation in a current cache has already passed the
    seen-pixel rule and, for radar, a per-track calibration, so "reliable"
    here only drops implausibly small readings (a 0.02 ha detection failure
    is on record; the lagoon did not drain that day)."""
    te = now.timestamp()
    ok = [o for o in observation_records(model)
          if o["t"] <= te and o["area_ha"] >= CFG["lake_min_plausible_ha"]]
    return max(ok, key=lambda o: o["t"]) if ok else None


def describe_observation(o):
    """Short human label for where a lagoon area came from."""
    if o.get("sensor") == "optical":
        return "satellite photo"
    if o.get("sensor") == "landsat":
        return "Landsat photo"
    if o.get("sensor") == "sar":
        return f"radar, {o['track'].replace('S1 ', '')}" if o.get("track") else "radar"
    return "satellite image"


def possible_fill(since_te, now_te, sea, rain):
    """Whether the lagoon may have filled since the observation at since_te.

    Not a model - two fills in a year can't support one. Both 2025-26 fills
    followed stormy, wet spells, so this flags any such spell since the
    newest image (fill_flag_sea_m or fill_flag_rain_mm), looking back at most
    fill_flag_max_days. `sea` and `rain` are (t, values) arrays, or None.
    Returns {"max_sea_m", "rain_mm", "flagged"} (None where there's no data)."""
    t0 = max(since_te, now_te - CFG["fill_flag_max_days"] * 86400)
    out = {"max_sea_m": None, "rain_mm": None, "flagged": False}
    if sea is not None and len(sea[0]):
        m = (sea[0] > t0) & (sea[0] <= now_te)
        if m.any():
            out["max_sea_m"] = float(np.max(sea[1][m]))
    if rain is not None and len(rain[0]):
        m = (rain[0] > t0) & (rain[0] <= now_te) & np.isfinite(rain[1])
        if m.any():
            out["rain_mm"] = float(np.sum(rain[1][m]))
    out["flagged"] = bool(
        (out["max_sea_m"] is not None and out["max_sea_m"] >= CFG["fill_flag_sea_m"])
        or (out["rain_mm"] is not None and out["rain_mm"] >= CFG["fill_flag_rain_mm"]))
    return out


def wetness_fingerprint(lagoon, ring, basin=None):
    """Identifies the exact outline (and basin) a wetness frequency map was
    fitted for.

    The frequency map is only meaningful for the lagoon footprint it was built
    from: re-trace the outline and its per-pixel values refer to ground that
    is no longer in the mask, while newly-included cells have never been
    observed at all. Comparing array shapes does not catch this - the shape is
    fixed by bbox and resolution, so it matches no matter which outline was
    used. The basin is part of it because the map covers the basin, which
    grows with every --outline (geometry.basin_mask)."""
    h = hashlib.sha1()
    h.update(json.dumps({"bbox": list(CFG["bbox_wgs84"]), "res": CFG["grid_res_m"],
                         "ring": [[round(c, 6) for c in p] for p in ring]},
                        sort_keys=True).encode())
    h.update(np.packbits(lagoon).tobytes())
    if basin is not None:
        h.update(b"basin")
        h.update(np.packbits(basin).tobytes())
    return h.hexdigest()


def resting_area(model, te):
    """The lagoon's resting size at te: the median plausible image over the
    lagoon_rest_days before it. A median, so one fill can't move it; a fill
    drains back to this (flood.FillDrain)."""
    return resting_area_of(observation_records(model), te)


def resting_area_of(records, te):
    """resting_area for a plain list of observation dicts (the fit replays
    the record without a model object)."""
    t0 = te - CFG["lagoon_rest_days"] * 86400
    areas = [o["area_ha"] for o in records
             if t0 <= o["t"] <= te and o["area_ha"] >= CFG["lake_min_plausible_ha"]]
    return float(np.median(areas)) if areas else None


def wet_mask_from_area(freq, lagoon, target_ha, cell_area_m2):
    """Which lagoon cells are wet for a given target area: the ones most
    often observed wet in the satellite record fill in first - a direct
    measurement of where the water actually is, not a bathymetry guess.

    Grown outward from one seed (the most often wet cell, ties going to the
    one deepest inside the outline), always taking the most often wet cell
    that borders the water so far. The lagoon is one body of water; ranking
    cells over the whole outline instead could wet a far corner before the
    water that joins it, and draw the lagoon as separate patches."""
    target_px = min(int(round(target_ha * 1e4 / cell_area_m2)), int(lagoon.sum()))
    mask = np.zeros(lagoon.shape, bool)
    if target_px <= 0:
        return mask
    depth = ndimage.distance_transform_edt(lagoon)
    ys, xs = np.nonzero(lagoon)
    seed = max(zip(freq[ys, xs], depth[ys, xs], ys, xs))[2:]
    h, w = lagoon.shape
    queued = np.zeros(lagoon.shape, bool)
    queued[seed] = True
    heap = [(-freq[seed], -depth[seed], *seed)]
    n = 0
    while heap and n < target_px:
        _, _, y, x = heapq.heappop(heap)
        mask[y, x] = True
        n += 1
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w and lagoon[yy, xx] and not queued[yy, xx]:
                queued[yy, xx] = True
                heapq.heappush(heap, (-freq[yy, xx], -depth[yy, xx], yy, xx))
    # a dry cell boxed in by water is a frequency-map speck, not an island
    return ndimage.binary_fill_holes(mask) & lagoon


class LagoonWater:
    """The lagoon's water at any instant - the only water that counts.

    One object answers for both the hourly frames and the week-long window
    search, so the two can't disagree. (They used to read the rideable area
    from different places: the frames computed it exactly, the window search
    interpolated a sea-level curve, and a check had to compare their verdicts.
    With the tide-driven beach no longer counted, nothing needs interpolating.)

    Without a fitted model the answer is the whole outline, one constant. In
    lake mode the size starts at the newest image and, when `fill_drain` is
    given (a callable taking the lagoon mask at that image and returning a
    flood.FillDrain), fills on surges and drains back between them. The
    water is grown within `basin` (the satellite record covers the whole
    basin), or within the outline when no basin is given; after a surge,
    within what the surge filled, so it drains from the edges the record
    sees wet least often. In the tidal and wind modes the area comes from
    the model's area_at() for each instant.
    With `unknown` (a user's copy without the shared model) there is no
    lagoon at all: no water is drawn or counted, and rideable_at calls every
    hour "unknown". The whole-outline fallback is for the maintainer only:
    shown to a user it once painted a 1.4 km placeholder box over the sea as
    lagoon, when neither the model nor OpenStreetMap could be reached.
    at(te) returns (wet_mask, lagoon_ha, largest_patch_ha)."""

    def __init__(self, lagoon, grid, model=None, freq=None, sea_level_fn=None,
                 wind_hist=None, lake_area=None, basin=None, fill_drain=None, range_fn=None,
                 unknown=False):
        self.lagoon, self.grid = lagoon, grid
        self.unknown = unknown
        self.region = lagoon if basin is None else basin | lagoon
        self.model, self.freq = model, freq
        self.sea_level_fn, self.wind_hist, self.lake_area = sea_level_fn, wind_hist, lake_area
        self._fixed, self.fill_drain, self._sized = None, None, {}
        self.range_fn = range_fn
        if unknown:
            self._fixed = self._finish(np.zeros_like(lagoon))
        elif model is None or freq is None:
            self._fixed = self._finish(lagoon.copy())
        elif model.mode == "lake" and lake_area is not None:
            self._fixed = self._finish(
                wet_mask_from_area(freq, self.region, lake_area, grid.cell_area_m2))
            if fill_drain is not None:
                fd = fill_drain(self._fixed[0])
                self.fill_drain = fd if fd.varies else None

    @property
    def constant(self):
        return self._fixed is not None and self.fill_drain is None

    def range_at(self, te, area_ha=None):
        """(low, high) likely size at te from `range_fn` (FillDrainFit.range_ha:
        how far off the fit was on images it didn't see, wider after a surge),
        or None. The verdict tests the low end (flood.rideable_at)."""
        if self.range_fn is None:
            return None
        if area_ha is None:
            area_ha = self.at(te)[1]
        st = self.state(te)
        return self.range_fn(area_ha, st is not None and st[1] is not None)

    def state(self, te):
        """(area_ha, surge) where surge is the (level, te) of the surge that set
        the size, or None; None outside fill-and-drain."""
        if self.fill_drain is None:
            return None
        area, _, event = self.fill_drain.at(te)
        return area, event

    def _finish(self, wet):
        wet = remove_small_patches(wet, self.grid)
        cell = self.grid.cell_area_m2
        return wet, float(wet.sum() * cell / 1e4), largest_patch_ha(wet, cell)

    def at(self, te):
        if self.fill_drain is not None:
            area, region, _ = self.fill_drain.at(te)
            # keyed on the rounded size, so the week-long window search reuses masks
            key = (round(area, 2), None if region is None else id(region))
            if key not in self._sized:
                self._sized[key] = self._finish(wet_mask_from_area(
                    self.freq, self.region if region is None else region, area,
                    self.grid.cell_area_m2))
            return self._sized[key]
        if self._fixed is not None:
            return self._fixed
        area = self.model.area_at(self.sea_level_fn, te, wind_hist=self.wind_hist,
                                  lake_area=self.lake_area)
        return self._finish(wet_mask_from_area(self.freq, self.region, area,
                                               self.grid.cell_area_m2))
