"""Typed exceptions for conditions that used to call sys.exit() deep inside
library code. Raised from terrain.py/tides.py, caught once in cli.main()."""

from __future__ import annotations


class ZandmotorError(Exception):
    """Base class for errors the CLI turns into a clean exit message."""


class TerrainFetchError(ZandmotorError):
    """Could not obtain a usable AHN elevation tile."""


class NoSeaFoundError(ZandmotorError):
    """No open sea found on the map edge; the grid/bbox doesn't reach the coast."""


class NoWaterLevelDataError(ZandmotorError):
    """No RWS water level data (measured, forecast, or astronomical) at all."""


class HarmonicFitUnavailableError(ZandmotorError):
    """The harmonic-tide fallback can't run (utide missing, or not enough
    measured data to fit it)."""


class CalibrationIncompleteError(ZandmotorError):
    """Too many satellite requests failed during --calibrate to trust the
    record. Raised instead of fitting on what little arrived, so the cached
    model is left untouched: a run whose token expired mid-way once fitted a
    "wind" model on 9 images and overwrote a year-long record."""
