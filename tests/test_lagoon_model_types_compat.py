import pytest

from zandmotor.lagoon_model.types import LakeResponse, model_from_dict


def test_cache_written_before_obs_existed_still_loads():
    old = {"mode": "lake", "r": 0.15, "wind_r": 0.16, "n": 2, "fitted": "2026-10-06T20:00:00+00:00",
           "times": [1.0, 2.0], "areas": [3.0, 4.0], "n_optical": 1, "n_sar": 1}
    m = model_from_dict(old)
    assert isinstance(m, LakeResponse)
    assert m.obs == [] and m.sar_tracks == {}


def test_new_fields_round_trip():
    m = LakeResponse(r=0.1, wind_r=0.1, n=1, fitted="x", times=[1.0], areas=[3.0],
                     n_optical=1, n_sar=0,
                     obs=[{"t": 1.0, "area_ha": 3.0, "sensor": "optical", "track": "S2"}],
                     sar_tracks={"37": {"threshold_db": -19.5, "used": True}})
    assert model_from_dict(m.to_dict()) == m


def test_missing_required_field_is_an_error():
    with pytest.raises(KeyError):
        model_from_dict({"mode": "lake", "r": 0.1})
