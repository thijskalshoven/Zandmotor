import numpy as np
import pytest

from zandmotor.errors import NoSeaFoundError
from zandmotor.terrain import spill_elevation


def test_spill_elevation_raises_typed_error_not_systemexit():
    # An all-dry grid: nothing on the border sits below sea_seed_level, so
    # there is no open sea to seed the flood-fill from.
    z = np.full((10, 10), 5.0, dtype="float32")
    with pytest.raises(NoSeaFoundError):
        spill_elevation(z)
