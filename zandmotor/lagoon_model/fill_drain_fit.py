"""Fits the lagoon's fill-and-drain (flood.FillDrain) to the satellite record
(--calibrate only): how fast a fill drains, the basin floor under the AHN
gaps, and when a surge reaches the lagoon - the flood is tested at the
effective level sea + wave_k x wave term + rain_k x rain - flood_offset_m
(flood.effective_level), with the wave term in one of four forms
(wind.wave_forcing).

How: replay the record image to image. From each image, step the measured
sea level, waves and rain forward with flood.fill_drain_series - the same
steps the report takes - and compare the size reached at the next image with
what that image saw. Every combination of the fill_fit_* grids is tried.

Choosing what to keep: thousands of combinations on ~27 image pairs overfit
if the best score simply wins. So the combinations are grouped into families
(which wave form, rain or not) and each family is scored by leave-one-out:
every image is predicted with the family's best parameters fitted on all the
other images. The family that predicts unseen images best wins - a simpler
one if it is within fill_fit_tolerance_ha - and only then are its parameters
fitted on everything. The same out-of-sample errors give the report its
uncertainty range (fit.ranges).

Why a replay and not a regression: the record has three fills in a year
(2026-05-14, 2026-09-01, 2026-11-12) and images days to weeks apart, so the
drain can't be read off any pair of images. "One to three weeks", inferred
earlier from sparse images, scored worst; the fits find about a day.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np

from zandmotor.config import CFG, log
from zandmotor.flood import (
    compute_wet_masks, effective_level, fill_drain_series, step_times, surge_fill,
)
from zandmotor.lagoon_model.runtime import resting_area_of, wet_mask_from_area
from zandmotor.terrain import (
    coastal_zone_mask, permanent_sea_mask, prepare_terrain, spill_elevation,
)
from zandmotor.wind import rain_sum, wave_forcing

# simplest first: a family has to beat every simpler one to be chosen
FAMILY_ORDER = ["still", "height", "height_dir", "runup", "runup_dir"]


@dataclasses.dataclass
class FillDrainFit:
    """The fitted (or default) fill-and-drain parameters, with the replay that
    justifies them. `fitted` is None for the CFG defaults."""
    drain_days: float
    basin_gap_fill_m: float
    wave_k: float = 0.0                     # the wave term's weight in the flood level
    flood_offset_m: float = 0.0             # the flood level sits this much above the DEM's
    wave_form: str = "height"               # wind.wave_forcing form of the wave term
    rain_k: float = 0.0                     # m of flood level per mm of recent rain
    base_ha: float | None = None            # the lagoon size the surge was flooded onto
    fitted: str | None = None
    fingerprint: str | None = None          # the outline and basin it was fitted for
    n: int = 0                              # image-to-image steps replayed
    mae_ha: float | None = None             # in-sample, with the chosen parameters
    cv_mae_ha: float | None = None          # leave-one-out: images the fit didn't see
    hold_mae_ha: float | None = None        # holding the newest image, for comparison
    family: str = "still"
    cv: dict = dataclasses.field(default_factory=dict)       # family -> leave-one-out error
    ranges: dict = dataclasses.field(default_factory=dict)   # "calm"/"surge" -> [lo, hi, n]
    table: list = dataclasses.field(default_factory=list)    # best rows, for the log
    replay: list = dataclasses.field(default_factory=list)   # per image: observed vs replayed
    curve: list = dataclasses.field(default_factory=list)    # [level, ha or None] at the fit

    def fill_ha(self):
        """fill_ha for fill_drain_series from the stored curve, or None when the
        fit predates it. The report's size-over-time chart replays past images
        with it - the same replay the error in the caption was scored on."""
        if not self.curve:
            return None
        return curve_fn([c[0] for c in self.curve],
                        [np.nan if c[1] is None else c[1] for c in self.curve])

    def range_ha(self, area_ha, surged):
        """(low, high) around an estimated size: the 10-90% spread of how far
        off the leave-one-out replay was on images after a surge (`surged`)
        or in calm spells. None without a fit."""
        r = self.ranges.get("surge" if surged else "calm") or self.ranges.get("calm")
        if not r:
            return None
        return max(area_ha + r[0], 0.0), area_ha + r[1]

    @classmethod
    def defaults(cls):
        return cls(CFG["lagoon_drain_days"], CFG["basin_gap_fill_m"], CFG["lagoon_wave_k"],
                   CFG["lagoon_flood_offset_m"], CFG["lagoon_wave_form"], CFG["lagoon_rain_k"])

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def load(cls, path, fingerprint):
        """The cached fit if it was made for this outline and basin, else the
        defaults. Returns (fit, why_not) with why_not None when the cache was
        used. The fit depends on the basin (it floods it) just as the wetness
        map does, so it is invalidated by the same fingerprint."""
        if not path.exists():
            return cls.defaults(), "no fill-and-drain fit yet"
        d = json.loads(path.read_text())
        if d.get("fingerprint") != fingerprint:
            return cls.defaults(), "the outline or basin changed since the fill-and-drain fit"
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d}), None


def fill_curve(grid, z_raw, lagoon, basin, base_wet, gap_fill_m, levels, dry_override,
               low_water_m):
    """The basin's water (ha) with the sea at each of `levels`, NaN where the
    sea doesn't reach the lagoon - what FillDrain measures at each step, for
    one basin floor. The flood only grows with the level, so this is the
    whole of the model's dependence on the terrain."""
    z = prepare_terrain(z_raw, lagoon, basin, gap_fill_m)
    F = spill_elevation(z)
    coastal_zone = coastal_zone_mask(z, lagoon | basin)
    sea = permanent_sea_mask(z, lagoon, low_water_m)
    min_px = CFG["probable_min_extra_ha"] * 1e4 / grid.cell_area_m2
    out = []
    for lvl in levels:
        beach, _ = compute_wet_masks(z, F, lvl, coastal_zone, lagoon, sea, dry_override, grid)
        m = surge_fill(base_wet, beach, sea, basin)
        out.append(m.sum() * grid.cell_area_m2 / 1e4 if (m & ~base_wet).sum() >= min_px
                   else np.nan)
    return np.array(out)


def curve_fn(levels, areas):
    """fill_ha for fill_drain_series from a fill_curve: None below the level
    where the sea first reaches the lagoon, interpolated above it."""
    ok = np.isfinite(areas)
    if not ok.any():
        fn = lambda lvl: None                                         # noqa: E731
        fn.onset = None
        return fn
    lv, ar = np.asarray(levels)[ok], np.asarray(areas)[ok]

    def fn(lvl):
        return None if lvl < lv[0] else float(np.interp(lvl, lv, ar))
    fn.onset = float(lv[0])     # lets fill_drain_series skip the steps below it
    return fn


def levels_from(sea, waves, wave_k, offset_m, form="height", rain=None, rain_k=0.0):
    """levels_at for replay: the effective flood level (flood.effective_level)
    at times ts, from the measured sea series, the waves (None = still water)
    and the hourly rain (times, mm). NaN where the sea record has no data."""
    sea_t, sea_v = sea

    def levels_at(ts):
        lv = np.interp(ts, sea_t, sea_v, left=np.nan, right=np.nan)
        wave = wave_forcing(waves, ts, form) if wave_k else 0.0
        mm = rain_sum(rain, ts, CFG["fill_rain_window_h"]) if rain_k else 0.0
        return effective_level(lv, wave, wave_k, offset_m, mm, rain_k)
    return levels_at


def pair_levels(obs, levels_at):
    """For each image and the next: (o0, o1, steps, effective levels, resting
    size) - the inputs of one replay step that don't depend on the floor or
    the drain time, so the grid search computes them once."""
    out = []
    for o0, o1 in zip(obs[:-1], obs[1:]):
        ts = step_times(o0["t"], o1["t"])
        ts = ts[ts <= o1["t"]]
        levels = levels_at(ts) if len(ts) else np.array([])
        levels = np.round(np.where(np.isfinite(levels), levels, -99.0), 2)
        out.append((o0, o1, ts, levels, resting_area_of(obs, o0["t"])))
    return out


def replay_pairs(pairs, fill_ha, drain_days):
    """[(next_obs, predicted_ha, surged)] for pair_levels output: the size
    fill-and-drain reaches by the next image, and whether a surge filled the
    basin on the way."""
    rows = []
    for o0, o1, ts, levels, rest in pairs:
        if not len(ts):
            rows.append((o1, o0["area_ha"], False))
            continue
        areas, events, _ = fill_drain_series(o0["t"], o0["area_ha"],
                                             o0["area_ha"] if rest is None else rest,
                                             ts, levels, fill_ha, drain_days,
                                             onset=getattr(fill_ha, "onset", None))
        rows.append((o1, areas[-1], events[-1] is not None))
    return rows


def replay(obs, levels_at, fill_ha, drain_days):
    """From each image to the next, the size fill-and-drain reaches by the
    next image. `obs` sorted by time; `levels_at(ts)` the effective flood
    levels (levels_from). Returns [(next_obs, predicted_ha, surged)]."""
    return replay_pairs(pair_levels(obs, levels_at), fill_ha, drain_days)


def pair_errors(rows, basin_ha=None):
    """Absolute error of replayed against observed size, per image. An image
    that saw less than fill_fit_full_view of the basin brackets the size
    instead: at least what it saw, at most that plus the hidden part of the
    basin (basin_ha x the unseen fraction), and only a prediction outside
    that range counts. (Counting only predictions below it let the fit fill
    the basin on 2026-04-04, when the 65% that was visible held 2.7 ha.)"""
    errs = []
    for o, p, *_ in rows:
        seen = o.get("seen_basin", 1.0)
        if seen >= CFG["fill_fit_full_view"]:
            errs.append(abs(p - o["area_ha"]))
            continue
        hi = o["area_ha"] + (1.0 - seen) * basin_ha if basin_ha else float("inf")
        errs.append(max(o["area_ha"] - p, p - hi, 0.0))
    return np.array(errs, float)


def mean_error(rows, basin_ha=None):
    """Mean of pair_errors."""
    e = pair_errors(rows, basin_ha)
    return float(e.mean()) if len(e) else float("nan")


def _family(form, k, rain_k):
    return ("still" if not k else form) + ("+rain" if rain_k else "")


def _family_rank(name):
    base, rain = name.split("+")[0], "+rain" in name
    return (rain, FAMILY_ORDER.index(base))


def _spread(res):
    """[lo, hi, n] of signed residuals (observed - predicted): the 10-90%
    range with enough of them, the extremes with few, and never excluding 0."""
    if not len(res):
        return None
    lo, hi = (np.percentile(res, [10, 90]) if len(res) >= 8 else (res.min(), res.max()))
    return [round(float(min(lo, 0.0)), 2), round(float(max(hi, 0.0)), 2), int(len(res))]


def fit_drain(obs, levels_for, curves, drain_grid, wave_options=((None, 0.0),),
              offsets=(0.0,), rain_ks=(0.0,), basin_ha=None):
    """Grid search over basin floors (`curves` maps a floor in m NAP to
    (levels, areas) from fill_curve), drain times, wave terms (form, k),
    rain weights and flood offsets; `levels_for(form, k, offset, rain_k)`
    returns a levels_at (levels_from). The family is chosen by leave-one-out
    (see the module docstring). Returns a FillDrainFit without
    fitted/fingerprint, or None with too few images. `basin_ha` brackets
    partly seen images (pair_errors)."""
    obs = sorted((o for o in obs if o["area_ha"] >= CFG["lake_min_plausible_ha"]),
                 key=lambda o: o["t"])
    if len(obs) < CFG["response_min_scenes"]:
        return None
    fns = {gap: curve_fn(*c) for gap, c in curves.items()}
    combos, errors, preds, surged = [], [], [], []
    for form, k in wave_options:
        for rk in rain_ks:
            for off in offsets:
                pairs = pair_levels(obs, levels_for(form, k, off, rk))
                for gap in sorted(fns):
                    for d in drain_grid:
                        rows = replay_pairs(pairs, fns[gap], d)
                        combos.append({"family": _family(form, k, rk), "wave_form": form or "height",
                                       "wave_k": float(k), "rain_k": float(rk),
                                       "flood_offset_m": float(off),
                                       "basin_gap_fill_m": float(gap), "drain_days": float(d)})
                        errors.append(pair_errors(rows, basin_ha))
                        preds.append([p for _, p, _ in rows])
                        surged.append([s for *_, s in rows])
    E, P, S = np.array(errors), np.array(preds), np.array(surged)
    n = E.shape[1]
    observed = np.array([o["area_ha"] for o in obs[1:]])
    full = np.array([o.get("seen_basin", 1.0) >= CFG["fill_fit_full_view"] for o in obs[1:]])
    fams = sorted({c["family"] for c in combos}, key=_family_rank)
    cv, loo = {}, {}
    for fam in fams:
        idx = np.array([i for i, c in enumerate(combos) if c["family"] == fam])
        tot = E[idx].sum(axis=1)
        # for each image, the family's best parameters on all the other images
        pick = idx[np.argmin((tot[:, None] - E[idx]) / max(n - 1, 1), axis=0)]
        cv[fam] = round(float(E[pick, np.arange(n)].mean()), 3)
        loo[fam] = pick
    best_cv = min(cv.values())
    fam = next(f for f in fams if cv[f] <= best_cv + CFG["fill_fit_tolerance_ha"])
    idx = [i for i, c in enumerate(combos) if c["family"] == fam]
    mae = E[idx].mean(axis=1)
    near = [i for i, m in zip(idx, mae) if m <= mae.min() + CFG["fill_fit_tolerance_ha"]]
    # the plainest parameters close to the family's best
    best = min(near, key=lambda i: (combos[i]["wave_k"], combos[i]["rain_k"],
                                    combos[i]["flood_offset_m"], E[i].mean()))
    pick = loo[fam]
    res = observed - P[pick, np.arange(n)]
    was_surge = S[pick, np.arange(n)]
    ranges = {"calm": _spread(res[full & ~was_surge]), "surge": _spread(res[full & was_surge])}
    hold = pair_errors([(o1, o0["area_ha"]) for o0, o1 in zip(obs[:-1], obs[1:])], basin_ha)
    table = sorted(([*combos[i].values(), round(float(E[i].mean()), 3)] for i in range(len(combos))),
                   key=lambda r: r[-1])[:40]
    c = combos[best]
    return FillDrainFit(
        drain_days=c["drain_days"], basin_gap_fill_m=c["basin_gap_fill_m"], wave_k=c["wave_k"],
        flood_offset_m=c["flood_offset_m"], wave_form=c["wave_form"], rain_k=c["rain_k"],
        n=n, mae_ha=round(float(E[best].mean()), 3), cv_mae_ha=cv[fam],
        hold_mae_ha=round(float(hold.mean()), 3), family=fam, cv=cv, ranges=ranges,
        table=table,
        replay=[{"t": o["t"], "observed_ha": round(o["area_ha"], 2),
                 "replayed_ha": round(float(P[best, i]), 2), "full_view": bool(full[i]),
                 "surged": bool(S[best, i])} for i, o in enumerate(obs[1:])])


def fit_fill_drain(grid, z_raw, lagoon, basin, freq, obs, sea, waves, rain, dry_override, now):
    """Fit fill-and-drain to the record `obs` with the measured sea series
    `sea` = (t, level), the waves (wind.fetch_waves, or None to fit on still
    water only) and the hourly rain (t, mm) or None. Returns a FillDrainFit
    (fingerprint not set), or None when the record is too short."""
    sea_t, sea_v = sea
    lo, hi, step = CFG["fill_fit_levels_m"]
    levels = np.round(np.arange(lo, hi + step / 2, step), 2)
    plausible = [o["area_ha"] for o in obs if o["area_ha"] >= CFG["lake_min_plausible_ha"]]
    if not plausible:
        return None
    # the lagoon the surge joins: its typical size, grown where it is most often
    # wet. The report floods the same size (base_ha), so its surges are the
    # ones the fit was scored on: a bigger lagoon reaches nearer the basin's
    # edge and is joined by smaller surges (+1.40 m against +1.62 m effective
    # level for the 5.7 ha lagoon of 2026-11-15).
    base_ha = float(np.median(plausible))
    base = wet_mask_from_area(freq, basin | lagoon, base_ha, grid.cell_area_m2)
    low = float(np.nanpercentile(sea_v, 1))
    curves = {g: (levels, fill_curve(grid, z_raw, lagoon, basin, base, g, levels,
                                     dry_override, low))
              for g in CFG["fill_fit_gap_fill_m"]}
    wave_options = [(None, 0.0)]
    if waves is not None:
        wave_options += [(form, k) for form, ks in CFG["fill_fit_wave_forms"].items() for k in ks]
    fit = fit_drain(obs,
                    lambda form, k, off, rk: levels_from(sea, waves, k, off, form or "height",
                                                         rain, rk),
                    curves, CFG["fill_fit_drain_days"], wave_options, CFG["fill_fit_offset_m"],
                    CFG["fill_fit_rain_k"] if rain is not None else (0.0,),
                    basin.sum() * grid.cell_area_m2 / 1e4)
    if fit is None:
        log.warning("Too few lagoon images to fit fill-and-drain; keeping the defaults.")
        return None
    fit.fitted = now.isoformat()
    fit.base_ha = round(base_ha, 3)
    fit.curve = [[float(lvl), None if not np.isfinite(a) else round(float(a), 3)]
                 for lvl, a in zip(levels, curves[fit.basin_gap_fill_m][1])]
    log.info("  leave-one-out error by family: %s",
             "  ".join(f"{f} {e:.2f}" for f, e in fit.cv.items()))
    log.info("Fill-and-drain: %s - drain time %g d, basin floor %.2f m NAP, flood level = sea + "
             "%.3g x %s + %.3g x rain - %.2f m; error %.2f ha on images it was fitted on, "
             "%.2f ha on images left out (holding the newest image: %.2f ha); range %s",
             fit.family, fit.drain_days, fit.basin_gap_fill_m, fit.wave_k, fit.wave_form,
             fit.rain_k, fit.flood_offset_m, fit.mae_ha, fit.cv_mae_ha, fit.hold_mae_ha,
             fit.ranges)
    if fit.cv_mae_ha >= fit.hold_mae_ha:
        log.warning("Fill-and-drain predicts unseen images no better than holding the newest "
                    "image on this record; the report uses it anyway, so check the fit.")
    return fit
