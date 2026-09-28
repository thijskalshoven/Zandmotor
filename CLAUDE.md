# CLAUDE.md

## Ask when something is unclear

**Always ask the user a question when something is unclear. Don't guess.** This applies to:

- Ambiguous requirements, or a request that can reasonably be read more than one way.
- Choices that change behaviour or results, such as model assumptions, thresholds in `CFG`, which data source to trust, or what counts as "rideable".
- Missing information: credentials, file paths, expected output, or which of several approaches the user prefers.
- Anything that conflicts with the documented findings (for example, reintroducing a tide or wind model for the lagoon; see "What the calibration actually found" in [zandmotor_lagoon.py](zandmotor_lagoon.py)).

Ask one focused question with concrete options, and recommend one when you have a view. Only go ahead without asking when the answer can be checked in the code or there is an obvious conventional default. If you do that, state the assumption you made.

## Project

This project is a nowcast for kitesurfing on the Zandmotor lagoon. It builds an interactive HTML map (`zandmotor_lagoon.html`) that shows where there is water, the wind, kite-size advice and a wetsuit call for the current hour and the next few hours. The full design rationale is in the module docstring of [zandmotor_lagoon.py](zandmotor_lagoon.py). Read it before changing any modelling logic.

It is not a safety tool, and nothing in the output should suggest that it is.

## Layout

- [zandmotor_lagoon.py](zandmotor_lagoon.py): thin entry point plus the main documentation docstring.
- [zandmotor/cli.py](zandmotor/cli.py): argument parsing and run orchestration. Typed errors are caught here, once.
- [zandmotor/config.py](zandmotor/config.py): `CFG` holds every tunable, URL and filename. Add new constants here rather than hardcoding them.
- [zandmotor/errors.py](zandmotor/errors.py): `ZandmotorError` subclasses. Library code raises these and never calls `sys.exit()`.
- [zandmotor/flood.py](zandmotor/flood.py): `compute_wet_masks` is the single flood-mask pipeline.
- [zandmotor/lagoon_model/](zandmotor/lagoon_model/): lagoon response model. `types.py` holds the tagged dataclasses `TidalResponse`, `WindResponse` and `LakeResponse`, each with its own `area_at()`.
- [zandmotor/satellite/](zandmotor/satellite/): CDSE client plus Sentinel-2 (optical) and Sentinel-1 (SAR), with shared request-building.
- `terrain.py` (AHN), `tides.py` (RWS), `wind.py` (Open-Meteo), `kiting.py`, `windows.py`, `outline.py`, `rendering.py`, `html_report/`.
- `cache/`, `lagoon.geojson`, `*_check.png` and `zandmotor_lagoon.html` are generated and gitignored.

## Commands

```bash
source .venv/bin/activate
pip install -e ".[dev,tide]"

python zandmotor_lagoon.py                 # map for now + next 8 hours
python zandmotor_lagoon.py --hours 12 --step 30 --no-open
python zandmotor_lagoon.py --outline       # trace lagoon (auto | sentinel | osm)
python zandmotor_lagoon.py --calibrate     # needs CDSE_CLIENT_ID / CDSE_CLIENT_SECRET

pytest                                     # run the tests in tests/
```

A normal run calls live external APIs (PDOK, Rijkswaterstaat, Open-Meteo and optionally CDSE). Tests should not depend on the network.

## Conventions

- Keep the existing refactor direction: centralized config, typed exceptions, one shared flood pipeline, tagged dataclasses instead of polymorphic dicts, and no duplicated CDSE request code.
- Match the surrounding style: `from __future__ import annotations`, docstrings that explain *why*, and comments that record past findings so the same leads aren't chased twice.
- Only the lagoon plus the intertidal strip counts as rideable. The open sea is never included in area totals.
- Test kite upper limits against the gust, not the mean wind.
- Commit only when asked.
