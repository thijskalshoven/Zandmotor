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
- [zandmotor/config.py](zandmotor/config.py): `CFG` holds every tunable, URL and filename. Add new constants here rather than hardcoding them. It also resolves the two layouts: a git checkout keeps everything in the project root; an installed copy keeps settings, kite photos and the report in `USER_DIR` (`~/Zandmotor Kite`) and the model and cache in `DATA` (`~/Library/Caches/zandmotor`). Never build paths from the code's own location.
- [zandmotor/settings.py](zandmotor/settings.py): the user's `settings.toml` (rider, kites, wetsuit table, `min_rideable_ha`), validated and merged over `CFG` at the start of `cli.main()`. `CFG` only holds an example rider. [zandmotor/setup_wizard.py](zandmotor/setup_wizard.py) is `--setup`.
- [zandmotor/model_bundle.py](zandmotor/model_bundle.py): the shared lagoon model snapshot. `--export-model` and [scripts/publish-model.sh](scripts/publish-model.sh) publish it as the `model` pre-release; installed copies download it. A new file that `--calibrate` or `--outline` produces and the report needs must be added to `model_files()`.
- [zandmotor/errors.py](zandmotor/errors.py): `ZandmotorError` subclasses. Library code raises these and never calls `sys.exit()`.
- [zandmotor/flood.py](zandmotor/flood.py): `compute_wet_masks` is the single flood-mask pipeline for the drawn, uncounted water (wet beach, sea, ponds). `rideable_at` is the one per-instant verdict.
- [zandmotor/lagoon_model/](zandmotor/lagoon_model/): lagoon response model. `types.py` holds the tagged dataclasses `TidalResponse`, `WindResponse` and `LakeResponse`, each with its own `area_at()` and the observation record with per-observation context (`obs`). `runtime.LagoonWater` is the only counted water, shared by the hourly frames and the window search; in lake mode it fills on surges and drains back via `flood.FillDrain`, whose drain time, wave weight, flood offset and basin floor `fill_drain_fit.py` refits on every `--calibrate` (both run `flood.fill_drain_series` on `flood.effective_level`, so keep them on those functions). The report's size-over-time chart (`cli._lagoon_trend`) replays the past with the fit's stored curve.
- [zandmotor/satellite/](zandmotor/satellite/): CDSE client plus Sentinel-2 and Landsat 8-9 (optical) and Sentinel-1 (SAR, now only for the flats dry-override: it can't map the lagoon), with shared request-building.
- `terrain.py` (AHN), `tides.py` (RWS), `wind.py` (Open-Meteo), `kiting.py`, `windows.py`, `outline.py`, `rendering.py`, `html_report/`.
- [install.sh](install.sh) and [zandmotor/launchers/](zandmotor/launchers/): the macOS installer (uv tool install from the `main` archive) and the double-click launchers. Installs come from `main`, so keep it working.
- `settings.toml`, `cache/`, `lagoon.geojson`, `lagoon_basin.geojson`, `*_check.png` and `zandmotor_lagoon.html` are personal or generated, and gitignored.

## Commands

```bash
source .venv/bin/activate
pip install -e ".[dev,tide]"

python zandmotor_lagoon.py                 # map for now + next 8 hours
python zandmotor_lagoon.py --hours 12 --step 30 --no-open
python zandmotor_lagoon.py --outline       # trace lagoon (auto | sentinel | osm)
python zandmotor_lagoon.py --calibrate     # needs CDSE_CLIENT_ID / CDSE_CLIENT_SECRET
python zandmotor_lagoon.py --setup         # rider and kites -> settings.toml
scripts/publish-model.sh                   # publish the lagoon model for installed copies (needs gh)

pytest                                     # run the tests in tests/
```

A normal run calls live external APIs (PDOK, Rijkswaterstaat, Open-Meteo and optionally CDSE). Tests should not depend on the network.

## Conventions

- Keep the existing refactor direction: centralized config, typed exceptions, one shared flood pipeline, tagged dataclasses instead of polymorphic dicts, and no duplicated CDSE request code.
- Match the surrounding style: `from __future__ import annotations`, docstrings that explain *why*, and comments that record past findings so the same leads aren't chased twice.
- Only the lagoon counts as rideable. The wet intertidal beach and the open sea are drawn but never included in area totals or the verdict.
- Satellite areas count only pixels the satellite actually saw. A second sensor (Landsat, and radar when it was used) gets its threshold fitted against Sentinel-2 images of the same days, per sensor or track; never one fixed threshold across them.
- Model terms in the fill-and-drain fit are chosen by leave-one-out error, not in-sample error: the record has ~30 images and 3 fills.
- Test kite upper limits against the gust, not the mean wind.
- Personal settings (rider, quiver, wetsuit) belong in `settings.toml`, never as new defaults in `CFG`. A new personal key needs a place in `settings.py`'s `_TABLES`, `load_settings`, `dump_settings` and the template.
- Warnings that tell the user to run `--calibrate` or `--outline` go through `cli._hint()`: users of an installed copy can't run them.
- Commit only when asked.
