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
1. Terrain   : AHN elevation (PDOK WCS, m NAP) for the beach and dunes, which
               AHN can measure directly. The lagoon's bed is underwater and
               was never reliably measurable this way, so it isn't modelled
               at all - see "Lagoon" below.
2. Water     : Rijkswaterstaat WaterWebservices (ddapi20): measured level,
               official forecast (incl. surge) and astronomical tide, with
               fallbacks (astro + decaying surge, or harmonic fit via utide).
               Drives tidal flooding of the beach/dunes.
3. Lagoon    : tracked by presence, not elevation. --calibrate tests how much
               of the lagoon is wet against the sea (tidal) and against wind
               gusts, using a year of Sentinel-2 (optical) and Sentinel-1
               (SAR, not cloud-limited) images. On the current record neither
               explains it (see "What the calibration actually found" below),
               so the lagoon is treated as a slowly-varying lake: its area is
               the median of recent satellite observations, reported with the
               spread of those observations rather than as a false forecast.
               The same images also build a per-pixel wetness frequency map,
               so an area can be turned back into an actual shape: the cells
               most often observed wet fill in first.
4. Wind      : Open-Meteo forecast (free, no key), a week ahead, plus
               sunrise/sunset so dark hours aren't offered as sessions.
5. Water temp: Open-Meteo marine API, turned into a wetsuit call.
6. Output    : zandmotor_lagoon.html, open it in any browser.

What counts as rideable
-----------------------
The lagoon plus the intertidal beach strip - never the open sea, which fills
most of the map area and would otherwise dominate every total (it measured
~620 ha against a ~2.8 ha lagoon, which silently disabled the
"is there enough water" test altogether). The sea is still drawn, in a paler
tone, so it's obvious it isn't being counted.

Kite advice comes from the quiver in CFG below: the question is whether a kite
you actually own works right now, capped by rider["skill_ceiling_kn"]. Upper
limits are tested against the gust, not the mean, since that's what you get
hit by.

Install
-------
    pip install requests numpy rasterio pyproj scikit-image pillow scipy
    pip install utide            # optional, extra fallback for the tide

Run
---
    python zandmotor_lagoon.py                  # map for now + next 8 hours
    python zandmotor_lagoon.py --hours 12 --step 30
    python zandmotor_lagoon.py --outline        # trace the lagoon from satellite
                                                # (or OpenStreetMap without account)
    python zandmotor_lagoon.py --calibrate      # refit the lagoon's wetness model
                                                # (needs CDSE credentials, below)

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

What the calibration actually found
-----------------------------------
Current fit (cache/lagoon_response.json, n=64 images: 20 optical + 44 SAR):

    mode = lake      sea level r = -0.05      wind gust r = 0.11

That is: neither the tide nor the wind explains the lagoon's wet area. It sits
around 2.6 ha (observed range 0.02-4.35 ha over a year) and drifts slowly.

An earlier run on a much smaller sample (n=20) found what looked like a strong
wind-gust relationship (12h trailing max gust, r=0.60, p=0.005) and this file
used to describe it as validated. It did not survive the larger sample - at
n=64 the same correlation is r=0.11. Treat the earlier finding as a false
positive from a small sample, not as a suppressed effect. It is recorded here
only so the same lead isn't chased twice.

The SAR wind-gating below is a separate matter and does still hold: a strong
gust shortly before acquisition makes water look like land to radar, so those
scenes are skipped rather than believed.

Not a safety tool. Always check conditions on the spot.
"""

from zandmotor.cli import main

if __name__ == "__main__":
    main()
