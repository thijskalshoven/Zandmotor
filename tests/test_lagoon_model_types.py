from zandmotor.lagoon_model.types import LakeResponse, TidalResponse, WindResponse, model_from_dict


def test_lake_response_round_trip_and_area_at():
    model = LakeResponse(r=-0.05, wind_r=0.11, n=64, fitted="2026-09-24T09:00:00+00:00",
                         times=[1.0, 2.0], areas=[2.5, 3.0], n_optical=20, n_sar=44)
    d = model.to_dict()
    assert d["mode"] == "lake"
    restored = model_from_dict(d)
    assert isinstance(restored, LakeResponse)
    assert restored == model
    # lake mode: a precomputed lake_area always wins over interpolation
    assert restored.area_at(None, 1.5, lake_area=9.9) == 9.9
    # without a precomputed lake_area, falls back to interpolating times/areas
    assert restored.area_at(None, 1.5) == 2.75


def test_tidal_response_round_trip_and_area_at():
    model = TidalResponse(delay_min=30, damping=2.0, offset=1.0, floor_ha=0.5, r=0.6, n=10,
                          rws_location="scheveningen", fitted="2026-01-01T00:00:00+00:00",
                          times=[], areas=[], n_optical=5, n_sar=5)
    d = model.to_dict()
    assert d["mode"] == "tidal"
    restored = model_from_dict(d)
    assert isinstance(restored, TidalResponse)

    def sea_level_fn(te):
        return (0.5, "measured")

    # area = damping * sea_lag + offset = 2*0.5 + 1 = 2.0, above the floor
    assert restored.area_at(sea_level_fn, 100.0) == 2.0

    def sea_level_fn_low(te):
        return (-10.0, "measured")

    # damping*sea_lag+offset goes very negative, clamped to floor_ha
    assert restored.area_at(sea_level_fn_low, 100.0) == 0.5


def test_wind_response_round_trip_and_area_at():
    model = WindResponse(window_h=12, slope=0.1, offset=1.0, area_min_ha=0.5, area_max_ha=4.0,
                         r=0.6, sea_r=0.1, n=20, fitted="2026-01-01T00:00:00+00:00",
                         times=[0.0, 3600.0], areas=[1.0, 2.0], n_optical=10, n_sar=10)
    d = model.to_dict()
    assert d["mode"] == "wind"
    restored = model_from_dict(d)
    assert isinstance(restored, WindResponse)
    # no wind_hist and no lake_area: falls back to interpolating times/areas
    assert restored.area_at(None, 1800.0) == 1.5
    # a precomputed lake_area still wins when wind_hist is absent
    assert restored.area_at(None, 1800.0, lake_area=3.3) == 3.3
