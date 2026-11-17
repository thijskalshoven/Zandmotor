"""Flood classification of the beach and dunes around the lagoon, and the
per-instant verdict shared by the hourly frames and the window search."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from zandmotor.config import CFG
from zandmotor.kiting import kite_advice, quiver_advice
from zandmotor.wind import wind_at


def flats_wet_map(z, F, sea_level, coastal_zone, lagoon):
    """Which cells outside the lagoon hold water right now, split into
    (tidal, ponded).

    `tidal` is water connected to the sea - the intertidal beach. `ponded` is
    standing water with no tidal route to the sea: dune lakes, ditches, wet
    dune valleys. Both are drawn, neither is counted (see compute_wet_masks).
    Depth doesn't matter for kiting, only presence of water, so this is
    boolean, not a depth value - but a cell only counts as flooded if the
    lowest point on its route to the sea is at least flood_margin_m below
    the water level (a thin, averaged-down sand ridge shouldn't leak) and
    the resulting film is at least min_visible_depth_m deep (anything
    thinner is a 5 m-grid artefact, not real standing water). Restricted to
    coastal_zone (open sea + lagoon only, see coastal_zone_mask): elevation
    alone can't tell a real dune-backed beach from a diked, pumped-dry town
    that happens to sit below sea level too.

    The beach and dunes are normal dry land, occasionally wet - AHN's laser
    measures them directly, so this elevation-based bathtub model is right
    for them. It's deliberately NOT used for the lagoon: its bed is
    underwater and was never reliably measurable this way (see
    wet_mask_from_area for how the lagoon is handled instead)."""
    flats = coastal_zone & ~lagoon
    connected = flats & (F < sea_level - CFG["flood_margin_m"])
    trapped = flats & ~connected & (z < F - CFG["pond_min_depth"])
    thin = CFG["min_visible_depth_m"]
    tidal = connected & (sea_level - z >= thin)
    ponded = trapped & (F - CFG["pond_drawdown"] - z >= thin)
    return tidal, ponded


def remove_small_patches(wet, grid):
    """Drop wet blobs smaller than min_patch_ha: isolated puddles left by
    diagonal leaks or DEM/observation noise, none of it rideable anyway."""
    struct = ndimage.generate_binary_structure(2, 1)      # 4-connected
    lab, n = ndimage.label(wet, structure=struct)
    if n == 0:
        return wet
    min_px = CFG["min_patch_ha"] * 1e4 / grid.cell_area_m2
    counts = ndimage.sum(wet, lab, index=np.arange(1, n + 1))
    small_ids = np.nonzero(counts < min_px)[0] + 1
    if len(small_ids) == 0:
        return wet
    wet = wet.copy()
    wet[np.isin(lab, small_ids)] = False
    return wet


def largest_patch_ha(wet, cell_area_m2):
    """Area of the biggest connected wet patch.

    This, not the total, is what "enough water to ride" means: water spread
    over disconnected puddles is not a session, and one continuous sheet of
    4 ha is."""
    lab, n = ndimage.label(wet, structure=ndimage.generate_binary_structure(2, 1))
    if n == 0:
        return 0.0
    counts = ndimage.sum(wet, lab, index=np.arange(1, n + 1))
    return float(counts.max() * cell_area_m2 / 1e4)


def compute_wet_masks(z, F, sea_level, coastal_zone, lagoon, sea, dry_override, grid):
    """The water around the lagoon that is drawn on the map but never counted:
    flood-classify the flats, drop small patches, apply the satellite
    dry-override, then split the wet beach from the open sea and the ponds.

    Only the lagoon counts as rideable (see LagoonWater). The intertidal beach
    used to count too, and it swung 0-21 ha with every tide: merged with the
    lagoon at high water it tripled the "biggest patch" the verdict tests, and
    the rideable-water chart became a tide curve. Before that the open sea
    (~620 ha) and the dune ponds (~21 ha) had made the same mistake.
    Returns (beach, uncounted): the wet intertidal beach, and the open sea
    plus isolated ponds."""
    tidal_wet, ponded_wet = flats_wet_map(z, F, sea_level, coastal_zone, lagoon)
    wet = remove_small_patches(tidal_wet, grid)
    if dry_override is not None:
        wet &= ~dry_override
    beach = wet & ~sea
    uncounted = (wet & sea) | remove_small_patches(ponded_wet & ~sea, grid)
    return beach, uncounted


def probable_extent(lagoon_wet, beach, uncounted, sea, bound):
    """The lagoon plus the flooded beach and ponded water inside `bound` that
    join it, closed into one body. Used twice, with two bounds:

    * the drawn "probable water edge" (bound: within probable_reach_m of the
      basin) - where the water probably reaches at this hour;
    * the surge fill (bound: the basin) - what a surge leaves behind in the
      basin, counted until a newer satellite image (see SurgeFill).

    The lagoon shows no tidal signal at satellite times (see the module
    docstring), but at surge highs the sea floods the basin: on 2026-11-12
    (+1.81 m NAP at 17:00) lagoon and sea were one sheet. Drawn as separate
    pale patches that read as loose beach water rather than one lagoon. The
    hourly edge stays out of the numbers: counting tide-driven water is what
    once tripled the "biggest patch" and turned the area chart into a tide
    curve (see compute_wet_masks). The drawn edge is bounded because, from
    about +1.6 m, the model floods the whole 4 km beach strip and an
    unbounded edge traced the coast (60-76 ha)."""
    struct = ndimage.generate_binary_structure(2, 1)
    body = lagoon_wet | ((beach | uncounted) & bound & ~sea)
    k = CFG["probable_smooth_px"]
    if k > 0:
        # closed before picking what joins the lagoon, so a one-cell dry sliver
        # (a 5 m-grid artefact) doesn't cut water off; padded so the closing
        # doesn't eat the body where it meets the array edge. A square, not
        # the 4-connected cross: the cross's diamond fits into a one-cell
        # notch in the outer edge and leaves it open.
        body = ndimage.binary_closing(np.pad(body, k),
                                      structure=ndimage.generate_binary_structure(2, 2),
                                      iterations=k)[k:-k, k:-k]
    body = ndimage.binary_fill_holes(body) & (bound | lagoon_wet) & ~sea
    lab, _ = ndimage.label(body | lagoon_wet, structure=struct)
    ids = np.unique(lab[lagoon_wet])
    return np.isin(lab, ids[ids > 0])


def surge_fill(base_wet, beach, sea, basin):
    """What a surge leaves in the basin: the lagoon plus the sea-connected
    flood inside the basin that joins it (probable_extent without ponds).

    Ponds stay out because they don't come from the sea: the flood model
    holds water in any depression below its spill level whatever the tide,
    so with ponds counted, a lagoon grown into one "filled" at low tide and
    could never drain below the pond's size (it stuck at 6.3 ha in the
    2026-11-17 report once the fit raised the basin floor to 1.5 m)."""
    return probable_extent(base_wet, beach, np.zeros_like(beach), sea, basin)


def effective_level(sea_m, wave_m, wave_k, offset_m, rain_mm=0.0, rain_k=0.0):
    """The level the basin's flood is tested at: the sea plus wave_k times
    the wave term (wind.wave_forcing), plus rain_k times the recent rain,
    minus offset_m.

    Overwash is waves running up on top of the sea, so still water alone
    filled the basin on calm high tides that left no fill (2026-03-05:
    +1.50 m NAP, 0.4 m waves) and missed the one after a wave storm
    (2026-05-11: +1.26 m, 2.6 m swell of 7-8 s from NNW). The rain term is a
    proxy, not overwash: of two alike surges, 2026-04-04 (+1.56 m, 1.4 m
    oblique waves, 2 mm rain) left no fill and 2026-09-01 (+1.56 m, 1.7 m
    oblique waves, 54 mm) filled the basin - rain running off the dunes may
    raise the lagoon until it joins the surge water. The offset lets the
    fitted flood level sit above the DEM's. --calibrate fits all of them and
    keeps a term only when it predicts images it wasn't fitted on better
    (lagoon_model.fill_drain_fit)."""
    return sea_m + wave_k * wave_m + rain_k * rain_mm - offset_m


def step_times(t0, t1):
    """The fill-and-drain time steps from t0 to t1: every surge_fill_step_min,
    on the clock (:00, :15, ...), so a frame on the hour reads its own step."""
    step = CFG["surge_fill_step_min"] * 60
    return np.arange(np.ceil(t0 / step) * step, t1 + step, step)


def fill_drain_series(obs_te, obs_ha, rest_ha, ts, levels, fill_ha, drain_days, onset=None):
    """The lagoon's size at each of `ts`, starting from an image of obs_ha at
    obs_te: the excess over the resting size decays with time constant
    drain_days, and whenever `fill_ha(level)` (the basin's area with the sea
    at that level, or None when the sea doesn't reach it) exceeds the size,
    the size jumps to it. Never drains below the image itself.

    Shared by FillDrain (the report) and the --calibrate fit
    (lagoon_model.fill_drain_fit), so the fitted drain time is tested on the
    same steps the report takes. Returns (areas, events, tops): events the
    (peak level, peak te, start te) of the surge that set the size so far, or
    None - one surge keeps its start te while its peak rises, so the start
    identifies it; tops the highest level that has filled the basin so far,
    or None.

    `onset`, when given, is the lowest level at which fill_ha can return an
    area: steps below it are pure decay and are computed in one go. The fit
    replays the year thousands of times, and almost every step is one."""
    rest = min(float(obs_ha), float(rest_ha))
    tau = drain_days * 86400
    ts, levels = np.asarray(ts, float), np.asarray(levels, float)
    n = len(ts)
    areas, events, tops = np.empty(n), [None] * n, [None] * n
    a, prev, event, top, flooding = float(obs_ha), float(obs_te), None, None, False
    steps = np.flatnonzero(levels >= onset) if onset is not None else range(n)
    i0 = 0
    for k in list(steps) + [n]:
        if k > i0:
            # no fill possible on these steps: the lagoon only drains
            areas[i0:k] = rest + (a - rest) * np.exp(-(ts[i0:k] - prev) / tau)
            events[i0:k], tops[i0:k] = [event] * (k - i0), [top] * (k - i0)
            a, prev, flooding = areas[k - 1], ts[k - 1], False
        if k == n:
            break
        t, lvl = ts[k], levels[k]
        a = rest + (a - rest) * np.exp(-(t - prev) / tau)
        prev = t
        filled = fill_ha(lvl)
        if filled is None:
            flooding = False
        else:
            raises = filled >= a
            if raises:
                a = filled
                top = lvl if top is None else max(top, lvl)
            # one surge is one episode of the sea reaching the basin; the
            # label names its peak, not the last step that topped it up
            # (the drain starts from that one)
            if flooding and lvl > event[0]:
                event = (float(lvl), float(t), event[2])
            elif not flooding and raises:
                event, flooding = (float(lvl), float(t), float(t)), True
        areas[k], events[k], tops[k] = a, event, top
        i0 = k + 1
    return areas.tolist(), events, tops


class FillDrain:
    """The lagoon's size between satellite images: it fills fast when a surge
    floods the basin and drains slowly back to its resting size.

    Lake mode used to hold the newest image until the next one. That missed
    the 2026-11-12 surge (+1.83 m NAP): the count stayed at 4.3 ha while
    lagoon and sea were one sheet. A first fix held the surge's fill until
    the next image, and the 2026-11-15 image found 45% of it already dry.

    So the size steps through time from the newest image: every
    surge_fill_step_min the excess over the resting size decays with time
    constant lagoon_drain_days, and whenever the sea floods the basin the
    size jumps to what the basin holds at that level (probable_extent,
    bounded by the basin - the flooded beach beyond it drains with the
    tide). The resting size is the median image of the lagoon_rest_days up
    to the newest one; the lagoon never drains below the image itself.

    Fitted by replaying the 2025-26 record image to image with the measured
    sea level (28 images, 3 fills): mean error 0.90 ha, against 2.05 ha for
    holding the newest image and 5.4 ha for holding a surge's fill. Drain
    times of 0.5-1.5 days score within 0.1 ha of each other. Rain-driven
    fills (2026-05-14 at +1.18 m) are not modelled; neither are surges that
    left no fill (2026-03-05, +1.50 m).

    `level_fn(te)` returns (level, source); `masks_fn(level)` returns the
    (beach, uncounted) flood masks; `base_wet` is the lagoon at the newest
    image. `drain_days` is the fitted drain time (lagoon_model.fill_drain_fit),
    lagoon_drain_days when not given."""

    def __init__(self, obs_te, obs_ha, rest_ha, until_te, level_fn, masks_fn, sea, basin,
                 base_wet, cell_area_m2, drain_days=None):
        self.ts = step_times(obs_te, until_te)
        self.obs_ha, self.cell_ha = float(obs_ha), cell_area_m2 / 1e4
        self.masks_fn, self.sea, self.basin, self.base_wet = masks_fn, sea, basin, base_wet
        self.min_px = CFG["probable_min_extra_ha"] * 1e4 / cell_area_m2
        self._fills = {}
        levels = np.array([level_fn(t)[0] for t in self.ts], float)
        levels = np.round(np.where(np.isfinite(levels), levels, -99.0), 2)
        onset = self._onset(np.unique(levels))

        def fill_ha(lvl):
            m = self._fill(lvl) if onset is not None and lvl >= onset else None
            return None if m is None else m.sum() * self.cell_ha

        self.area, self.event, self.top = fill_drain_series(
            obs_te, obs_ha, rest_ha, self.ts, levels, fill_ha,
            CFG["lagoon_drain_days"] if drain_days is None else drain_days, onset=onset)

    def _fill(self, lvl):
        """The basin's water when the sea stands at lvl, or None if the sea
        doesn't reach the lagoon there."""
        if lvl not in self._fills:
            beach, _ = self.masks_fn(lvl)
            m = surge_fill(self.base_wet, beach, self.sea, self.basin)
            self._fills[lvl] = m if (m & ~self.base_wet).sum() >= self.min_px else None
        return self._fills[lvl]

    def _onset(self, levels):
        """Lowest level in `levels` that floods the basin, by bisection: the
        flood only grows with the level, and each test is a full-map flood."""
        if not len(levels) or self._fill(levels[-1]) is None:
            return None
        lo, hi = 0, len(levels) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._fill(levels[mid]) is None:
                lo = mid + 1
            else:
                hi = mid
        return levels[lo]

    @property
    def varies(self):
        return any(e is not None for e in self.event) or any(
            abs(a - self.obs_ha) >= 0.05 for a in self.area)

    def at(self, te):
        """(area_ha, region, event) at te: region is the mask the water lies in
        (the largest fill so far, or None for the whole basin) and event the
        (peak level, peak te, start te) of the surge that set the size, or
        None."""
        k = int(np.searchsorted(self.ts, te, side="right")) - 1
        if k < 0:
            return self.obs_ha, None, None
        top = self.top[k]
        return self.area[k], (None if top is None else self._fills[top]), self.event[k]


def rideable_at(te, lagoon_water, wind):
    """Everything one instant needs, shared by the frame loop and the window
    search: lagoon water from `lagoon_water` (a LagoonWater), the wind, and
    the verdict. Returns (lagoon_ha, usable_ha, w, quiver, verdict, notes,
    wind_txt, wet_mask).

    Between satellite images the size is an estimate with a range; the
    verdict tests the low end of it, so an hour isn't called rideable on the
    optimistic side of a guess."""
    wet, lagoon_ha, usable_ha = lagoon_water.at(te)
    r = lagoon_water.range_at(te)
    w = wind_at(wind, te)
    quiver = quiver_advice(w, CFG["rider"])
    if lagoon_water.unknown:
        # no lagoon model: the wind and kite notes still hold, but whether there
        # is water to ride can't be said, so no hour gets a go or a no
        wind_txt, _, notes = kite_advice(w, float("inf"), quiver)
        return (lagoon_ha, usable_ha, w, quiver, "unknown", ["lagoon unknown"] + notes,
                wind_txt, wet)
    wind_txt, verdict, notes = kite_advice(w, usable_ha if r is None else min(usable_ha, r[0]),
                                           quiver)
    return lagoon_ha, usable_ha, w, quiver, verdict, notes, wind_txt, wet
