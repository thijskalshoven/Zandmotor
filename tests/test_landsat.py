import datetime as dt

import numpy as np

from zandmotor.config import UTC
from zandmotor.lagoon_model.calibration import _calibrate_landsat
from zandmotor.satellite.landsat import landsat_clear

T0 = dt.datetime(2026, 9, 1, 10, 40, tzinfo=UTC)


def test_cloud_shadow_does_not_hide_the_lagoon():
    # QA values seen over the clear, full lagoon on 2026-09-01: shadow bits set
    assert landsat_clear([23888, 21952]).all()
    assert not landsat_clear([0b1000, 0b10]).any()      # cloud, dilated cloud


def test_landsat_threshold_is_fitted_against_sentinel2_and_measures_the_basin():
    lag = np.zeros((20, 20), bool)
    lag[5:15, 5:15] = True
    reg = np.zeros((20, 20), bool)
    reg[2:18, 2:18] = True
    wet = np.zeros((20, 20), bool)
    wet[5:15, 5:17] = True                              # lagoon plus an arm in the basin
    ndwi = np.where(wet, 0.3, -0.2)
    ndwi[0, 0] = 0.5                                    # outside the basin: not counted
    clear = reg.copy()
    raw = [(T0 + dt.timedelta(days=8 * i), ndwi, clear) for i in range(3)]
    optical = {(T0 + dt.timedelta(days=8 * i, hours=1)).timestamp(): (wet[reg], reg[reg])
               for i in range(3)}
    obs, masks, info = _calibrate_landsat(raw, optical, lag, reg, 25 / 1e4)
    assert info["used"] and -0.2 < info["ndwi_threshold"] < 0.3 and info["mae_ha"] == 0.0
    assert len(obs) == 3 and obs[0]["sensor"] == "landsat"
    assert obs[0]["area_ha"] == wet.sum() * 25 / 1e4


def test_landsat_needs_enough_sentinel2_pairs():
    lag = np.ones((4, 4), bool)
    raw = [(T0, np.zeros((4, 4)), lag)]
    obs, masks, info = _calibrate_landsat(raw, {}, lag, lag, 25 / 1e4)
    assert obs == [] and not info["used"]
