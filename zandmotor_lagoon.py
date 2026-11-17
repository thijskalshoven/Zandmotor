#!/usr/bin/env python3
"""
Zandmotor lagoon nowcast for kitesurfing
========================================

Builds an interactive HTML map (with a time slider) of the Zandmotor lagoon for
the current hour and the next few hours: where there is water, plus wind and
kite-size advice. Depth isn't modelled or shown - kiting only needs a surface
to ride on, not a particular depth, so the only question that matters is
whether a spot is wet.

How it works
------------
1. Terrain   : AHN elevation (PDOK WCS, m NAP) for the beach and dunes, with
               the Rijkswaterstaat 2026 coastal DTM laid over it where that
               survey has data (CFG["coastal_dem"]): it measures most of the
               basin floor AHN got no return from. The lagoon's bed is
               underwater, so it isn't modelled - see "Lagoon" below.
2. Water     : Rijkswaterstaat WaterWebservices (ddapi20): measured level,
               official forecast (incl. surge) and astronomical tide, with
               fallbacks (astro + decaying surge, or harmonic fit via utide).
               Draws the wet beach on the map (never counted), and feeds the
               "lagoon may have filled since" warning.
3. Lagoon    : tracked by presence, not elevation. --calibrate tests how much
               of the lagoon is wet against the sea (tidal) and against wind
               gusts, using a year of Sentinel-2 and Landsat 8-9 images (radar
               was dropped: see finding 5 below). On the current record neither
               explains it (see "What the calibration actually found" below),
               so the lagoon is treated as a slowly-varying lake: its area
               starts at the newest reliable satellite observation, then
               fills on surges and drains back (see "Lagoon basin,
               fill-and-drain and probable water edge"), labelled as an
               estimate whenever it differs from the image.
               The same images also build a per-pixel wetness frequency map,
               so an area can be turned back into an actual shape: the cells
               most often observed wet fill in first.
4. Wind      : Open-Meteo forecast (free, no key), a week ahead, plus
               sunrise/sunset so dark hours aren't offered as sessions, and
               rain since the newest lagoon image.
5. Sea state : Open-Meteo marine API: water temperature (the wetsuit call)
               and wave height, period and direction (when a surge reaches
               the lagoon).
6. Output    : zandmotor_lagoon.html, open it in any browser.

What counts as rideable
-----------------------
The lagoon only. Three things have been counted with it and each one swamped
it: the open sea (~620 ha against a ~2.8 ha lagoon, which silently disabled
the "is there enough water" test), the isolated dune ponds (~21 ha), and
last the intertidal beach strip, which swung 0-21 ha with every tide. Merged
with the lagoon at high water, that strip tripled the "biggest patch" the
verdict tests (4.2 -> 13 ha) and turned the rideable-water chart into a tide
curve. All three are still drawn, in paler tones, so it's obvious they
aren't counted.

Kite advice comes from the quiver in CFG below: the question is whether a kite
you actually own works right now, capped by rider["skill_ceiling_kn"]. Upper
limits are tested against the gust, not the mean, since that's what you get
hit by.

Install
-------
See README.md: a one-line installer for non-technical Mac users (install.sh,
which sets up uv, the app, double-click launchers and the setup wizard), and
`pip install -e ".[dev,tide]"` from a checkout. The user's own rider, kites and
wetsuit table live in settings.toml (zandmotor.settings), not in CFG; installed
copies download the maintainer's lagoon model (zandmotor.model_bundle).

Run
---
    python zandmotor_lagoon.py                  # map for now + next 8 hours
    python zandmotor_lagoon.py --hours 12 --step 30
    python zandmotor_lagoon.py --outline        # trace the lagoon from satellite
                                                # (or OpenStreetMap without account)
    python zandmotor_lagoon.py --calibrate      # refit the lagoon's wetness model and
                                                # its fill-and-drain (needs CDSE
                                                # credentials, below)

This is now a thin entry point; the implementation lives in the zandmotor/
package (see zandmotor/cli.py for the orchestration, and zandmotor/config.py
for CFG). `python -m zandmotor` and the installed `zandmotor` command work
the same way.

Satellite calibration (optional)
--------------------------------
Create a free account at https://dataspace.copernicus.eu, then in the dashboard
create an OAuth client (Sentinel Hub). Set:
    export CDSE_CLIENT_ID=...
    export CDSE_CLIENT_SECRET=...

Lagoon outline
--------------
--outline traces the lagoon automatically and writes lagoon.geojson plus a
check image (lagoon_outline_check.png, north up) so you can verify it:
  * sentinel : picks a recent cloud-free Sentinel-2 image taken near high
               water, detects water (NDWI) and separates the lagoon from the
               open sea and the dune lake. Needs CDSE credentials (below).
  * osm      : uses the water polygons in OpenStreetMap. No account needed,
               but may be outdated.
  * auto     : sentinel if credentials are set, otherwise osm (default).
If the wrong water body is picked, set CFG["lagoon_hint"] to a (lon, lat) point
inside the lagoon and run again. You can also draw the outline yourself on
https://geojson.io and save it as lagoon.geojson.

Lagoon basin, fill-and-drain and probable water edge
----------------------------------------------------
The outline is one satellite trace, and it follows the newest --outline: a
trace from a quiet day is small (4.5 ha on 2026-11-05, against 13.4 ha at the
2026-09-01 fill). Every trace is also merged into lagoon_basin.geojson, which
only grows, so the basin keeps the largest extent seen; AHN no-data patches
touching it (water when surveyed) are added too. Delete the file if Zandmotor
reshapes so much that the old basin is wrong.
  * Satellite areas are measured over the basin, counting the water joined
    to the lagoon, so a fill isn't capped at the outline.
  * Fill-and-drain (flood.FillDrain): from the newest image on, the lagoon
    fills fast when the sea floods the basin (to what the basin holds at
    that level) and drains back to its resting size, the median image of
    the past 60 days, with a drain time constant. The flood is tested at
    sea level + k x wave term + rain term - offset (flood.effective_level):
    overwash is waves on top of the sea. --calibrate refits all of it by
    replaying the record image to image with the measured sea and the
    Open-Meteo wave and rain history (lagoon_model.fill_drain_fit; result in
    cache/fill_drain.json), and picks the wave form and whether to use rain
    by leave-one-out - how well it predicts images it wasn't fitted on. On
    the 2025-26 record (37 images): wave height x onshore share, k 0.6,
    offset 0.3 m, 0.5-day drain - 0.90 ha off on unseen images, against
    1.06 ha on still water and 1.67 ha for holding the newest image. The
    same out-of-sample errors give a likely range around the estimate, and
    the verdict tests its low end. Labelled an estimate, and drawn over time
    in the report's size-over-time chart.
  * Probable water edge: a dotted line per frame, the lagoon plus the
    flooded water joined to it within CFG["probable_reach_m"] of the basin.
    It moves with the tide and is drawn only, never counted (see "What
    counts as rideable").
AHN gaps in the basin are set to the basin floor (fitted, or
CFG["basin_gap_fill_m"]) so the tide can reach them, instead of being filled
as 20 m land. Only sea-connected water counts as a fill (flood.surge_fill):
a fitted floor of 1.5 m leaves the lower measured floor as ponds, which hold
water at any tide and once stopped the drain at 6.3 ha.

What the calibration actually found
-----------------------------------
The record is refitted with --calibrate; the cache (cache/lagoon_response.json)
holds the current fit and, per observation, its sensor, track, sea level, gust
and rain. This section records what the 2026-10 review of the 2025-26 record
established, so the same leads aren't chased twice.

1. The lagoon does not follow the tide. On a clean optical-only record (n=22
   images, each showing >=97% of the lagoon), sea level at image time gives
   r=+0.28 (p=0.23); in the older pooled record r was ~0.1 in every sensor
   subset. The tide was sampled across its full range (-1.26 to +1.28 m NAP).
   So the lagoon is a lake on the hour scale and lake mode is right for it.

2. It changes in events instead. It rests at ~2.5-4.5 ha and filled to ~13 ha
   twice: mid-May 2026 (13.5 ha on 05-14) and 2026-09-01 (13.7 ha; measured
   over the basin since 2026-11). They drain in days, not weeks: the replay
   fits a 1-day time constant, and surges that seemed to leave no fill
   (2025-11-25 at +1.58 m, 2026-03-05 at +1.50 m) had drained before the
   next image.
   Both fills followed stormy spells; 09-01 came after four days at +1.44 to
   +1.56 m NAP, 40 kn gusts and 64 mm of rain. Other such spells (2025-11-25:
   +1.58 m, 45 mm) left no fill that the next image saw - now explained by
   the fast drain. Surges are modelled (above); the May fill came at only
   +1.18 m with 19 mm of rain, so rain alone still only raises "may have
   filled since".
   On 2026-11-12 (measured peak +1.83 m NAP, 16:21) lagoon and sea were seen
   as one sheet: the overwash route is real. Before it, the 2026-11-10 image
   saw 4.5 ha across the whole basin; the next clear image (11-15, a third of
   the basin under cloud) still saw at least 5.7 ha, with an arm beyond the
   outline. Fill-and-drain replays this to 6.0 ha at the 11-15 image. The
   hourly tide still stays out of the numbers, since the record shows no
   tidal signal at image times.

3. Rain looked like a predictor and isn't (yet). Rain in the previous 3 days
   gave r=0.76 on the clean record, but r=0.02 without the two fills: the
   correlation was those two points. Same trap as the wind result below.
   2026-11-17: as a term in the fill model, rain lowered the in-sample error
   (0.88 -> 0.68 ha) but not the error on images left out (every "+rain"
   family scored worse by leave-one-out), so the fit doesn't use it. The
   case for it is 2026-09-01 (filled, 54 mm) against 2026-04-04 (no fill,
   2 mm) with near-identical sea and waves; one pair can't carry a term.

4. The old record was distorted by how it was measured, all now fixed:
   * Two radar tracks disagreed. Sentinel-1 relative orbit 37 (~05:49 UTC)
     sees the lagoon ~3 dB darker than orbit 110 (~05:57 UTC); with one fixed
     threshold their medians were 4.05 ha (n=27) and 1.79 ha (n=11), orbit 110
     lower in 6 of 6 pairs a day apart. Thresholds are now fitted per track
     against same-week optical images.
   * Half the lagoon was ignored in every radar scene: pixels in the ambiguity
     band (3.0-3.9 of 7.45 ha) counted as dry.
   * Unseen pixels counted as dry in optical images too: 2025-12-25 was
     recorded at 0.99 ha with ~2.5 ha of water under thin cirrus. Images now
     need >=90% of the lagoon seen, and only seen pixels count.
   * The tile-level cloud filter (<30%) dropped clear images, including the
     05-14 fill; the screen is now per pixel over the lagoon itself.
   * The outline trim at 0.5 m NAP cut 5.9 of 13.4 ha of real lagoon (AHN puts
     its margins at 1.1-1.5 m NAP: the lagoon is perched), capping every
     measured area at 7.45 ha. The trim now only removes dune (>3 m NAP).

5. Radar can't map this lagoon, and Landsat replaces it. Over the flat, wet,
   sandy basin, Sentinel-1 VV backscatter of water and sand overlaps: on 19
   passes within a day of a Sentinel-2 image a wet pixel was darker than a
   dry one 46-73% of the time (orbit 37 ~0.7, orbit 110 ~0.5 = chance; the
   13.5 ha fill of 2026-05-14 read -18.8 dB against -18.2 for the sand). VH
   and VV-VH were no better; per-track thresholds, per-pixel change
   detection against each pixel's own median, and both combined all fitted
   to "nothing is wet". So radar is off for the lagoon (CFG
   sar_in_calibration). Landsat 8-9 adds an optical pass every 8 days: 9
   scenes saw the lagoon in the year, 5 on dates no Sentinel-2 did; its
   NDWI threshold (0.175) is fitted against Sentinel-2 of the same days and
   misses it by 0.48 ha (2026-09-01: 13.6 ha against 13.7).

6. What filled the lagoon on 2026-05-14 (13.5 ha at only +1.18 m) is still
   open. Two days before, 2.6 m swell of 7-8 s came from NNW (onshore) at
   +1.26 m, then 16 mm of rain; the fill model replays it to ~4 ha. Wave
   period (a run-up form) did not predict unseen images better than plain
   height; the onshore share of the direction did. The 2026 coastal DTM also
   shows a channel from the lagoon's north-west corner to the sea, no
   higher than +0.69 m (AHN: +1.27 m) - possibly cut by an earlier storm and
   part of how fills drain in a day or two. The fitted offset (0.3 m) keeps
   ordinary tides from flooding through it, consistent with the record
   showing no tidal signal at image times.

An earlier run on a much smaller sample (n=20) found what looked like a strong
wind-gust relationship (12h trailing max gust, r=0.60, p=0.005) and this file
used to describe it as validated. It did not survive the larger sample - at
n=64 the same correlation is r=0.11. Treat the earlier finding as a false
positive from a small sample, not as a suppressed effect.

The SAR wind-gating is a separate matter and does still hold for the flats
dry-override, the one remaining radar use: a strong gust shortly before
acquisition makes water look like land to radar, so those scenes are skipped
rather than believed.

Not a safety tool. Always check conditions on the spot.
"""

from zandmotor.cli import main

if __name__ == "__main__":
    main()
