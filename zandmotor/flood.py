"""Flood classification and the rideable-area curve shared by the hourly
frame loop and the multi-day window search."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from zandmotor.config import CFG
from zandmotor.kiting import kite_advice, quiver_advice
from zandmotor.wind import wind_at


def flats_wet_map(z, F, sea_level, coastal_zone, lagoon):
    """Which cells outside the lagoon hold water right now, split into
    (tidal, ponded).

    `tidal` is water connected to the sea - the intertidal beach, which is what
    "beach shallows" means and which counts as rideable. `ponded` is standing
    water with no tidal route to the sea: dune lakes, ditches, wet dune
    valleys. Those are drawn but never counted, for two reasons - they are not
    beach, and they don't change with the tide, so including them put a
    constant ~21 ha in front of the "enough water" test and made it pass every
    hour regardless of conditions. Same failure as counting the open sea.
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

    This, not the total, is what "enough water to ride" means: 250 ha spread
    over disconnected puddles is not a session, and one continuous sheet of
    4 ha is. Using the total made the test unfailable once the intertidal
    beach was included."""
    lab, n = ndimage.label(wet, structure=ndimage.generate_binary_structure(2, 1))
    if n == 0:
        return 0.0
    counts = ndimage.sum(wet, lab, index=np.arange(1, n + 1))
    return float(counts.max() * cell_area_m2 / 1e4)


def rideable_curve(z, F, coastal_zone, lagoon, sea, lagoon_wet, dry_override, grid, levels):
    """(total_ha, largest_patch_ha) at each sea level in `levels`.

    Precomputed so the week-long window search doesn't rerun the flood model at
    every candidate hour. Mirrors the frame loop step for step, so the sidebar
    and the "next window" line cannot disagree. Wet area is monotonic in sea
    level, so interpolating between samples is safe.

    Assumes the lagoon's own wet mask is fixed across the run - exactly true in
    lake mode (its area is one measured constant); an approximation in the tidal
    and wind modes, where it only affects the window search, not the frames."""
    tot, big = [], []
    for lv in levels:
        tidal, _ = flats_wet_map(z, F, float(lv), coastal_zone, lagoon)
        wet = remove_small_patches(tidal | lagoon_wet, grid)
        if dry_override is not None:
            wet &= ~dry_override
        wet &= ~sea
        tot.append(wet.sum() * grid.cell_area_m2 / 1e4)
        big.append(largest_patch_ha(wet, grid.cell_area_m2))
    return np.asarray(tot, float), np.asarray(big, float)


def rideable_at(te, water, wind, curve):
    """Everything one instant needs, shared by the frame loop and the window
    search. `curve` is (levels, total_ha, largest_ha) from rideable_curve."""
    levels, tot, big = curve
    sea_level, _ = water.level(te)
    total_ha = float(np.interp(sea_level, levels, tot))
    usable_ha = float(np.interp(sea_level, levels, big))
    w = wind_at(wind, te)
    quiver = quiver_advice(w, CFG["rider"])
    wind_txt, verdict, notes = kite_advice(w, usable_ha, quiver)
    return total_ha, usable_ha, w, quiver, verdict, notes, wind_txt, sea_level
