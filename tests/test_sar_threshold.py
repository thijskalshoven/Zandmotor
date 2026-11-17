import numpy as np

from zandmotor.satellite.sar import fit_track_threshold, threshold_grid


def _pair(rng, water_db, land_db, n=400, wet_frac=0.5):
    """One synthetic SAR/optical pair over n lagoon pixels: the optical image
    says which pixels are wet, the SAR image is darker there."""
    opt_wet = np.zeros(n, bool)
    opt_wet[: int(n * wet_frac)] = True
    vv = np.where(opt_wet, water_db, land_db) + rng.normal(0, 0.8, n)
    seen = np.ones(n, bool)
    return vv, seen, opt_wet, seen


def test_threshold_lands_between_the_tracks_water_and_land_levels():
    rng = np.random.default_rng(0)
    # a "dark" track (like orbit 37) and a "bright" one (like orbit 110):
    # the same lagoon, ~3 dB apart - each needs its own threshold
    dark = [_pair(rng, -22.0, -15.0) for _ in range(4)]
    bright = [_pair(rng, -19.0, -12.5) for _ in range(4)]
    thr_dark, err_dark = fit_track_threshold(dark, threshold_grid())
    thr_bright, err_bright = fit_track_threshold(bright, threshold_grid())
    assert -21.0 < thr_dark < -16.0
    assert -18.0 < thr_bright < -13.5
    assert thr_bright > thr_dark
    # once calibrated, both tracks agree with optical on area
    assert max(abs(e) for e in err_dark + err_bright) < 0.05 * 400


def test_one_pair_across_a_fill_does_not_drag_the_threshold():
    rng = np.random.default_rng(1)
    pairs = [_pair(rng, -22.0, -15.0) for _ in range(4)]
    # optical saw the lagoon full, SAR two days later saw it drained
    vv, seen, opt_wet, _ = _pair(rng, -22.0, -15.0, wet_frac=0.2)
    pairs.append((vv, seen, np.ones_like(opt_wet), seen))
    thr, _ = fit_track_threshold(pairs, threshold_grid())
    assert -21.0 < thr < -16.0


def test_only_jointly_seen_pixels_are_compared():
    rng = np.random.default_rng(2)
    vv, seen, opt_wet, opt_seen = _pair(rng, -22.0, -15.0)
    opt_seen = opt_seen.copy()
    opt_seen[:100] = False                  # cloud over part of the optical image
    opt_wet = opt_wet.copy()
    opt_wet[:100] = False                   # ...where it would otherwise read dry
    thr, errors = fit_track_threshold([(vv, seen, opt_wet, opt_seen)] * 3, threshold_grid())
    assert -21.0 < thr < -16.0
    assert abs(errors[0]) < 20


def test_no_pairs():
    assert fit_track_threshold([], threshold_grid()) == (None, [])
