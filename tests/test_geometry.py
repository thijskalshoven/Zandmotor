import numpy as np
import pytest

from zandmotor.geometry import Grid, _simplify, make_grid, mask_to_ring, polygon_mask


def test_simplify_keeps_short_lines_untouched():
    pts = [(0, 0), (1, 1), (2, 2)]
    out = _simplify(pts, tol=0.1)
    assert len(out) == 3


def test_simplify_drops_collinear_points():
    # A straight line with redundant midpoints should collapse to endpoints.
    pts = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    out = _simplify(pts, tol=0.1)
    assert len(out) == 2
    assert tuple(out[0]) == (0, 0)
    assert tuple(out[-1]) == (4, 0)


def test_simplify_keeps_a_real_corner():
    # An L-shape: the corner is far enough from the chord to survive.
    pts = [(0, 0), (5, 0), (5, 5)]
    out = _simplify(pts, tol=0.1)
    assert len(out) == 3


def test_make_grid_matches_requested_bbox_resolution():
    bbox = (4.165, 52.035, 4.225, 52.070)
    grid = make_grid(bbox, res_m=5.0)
    assert isinstance(grid, Grid)
    assert grid.width > 0 and grid.height > 0
    assert grid.cell_area_m2 == pytest.approx(25.0)
    # latlon_bounds should roughly bracket the requested bbox
    (lat_min, lon_min), (lat_max, lon_max) = grid.latlon_bounds
    assert lat_min == pytest.approx(bbox[1], abs=0.01)
    assert lat_max == pytest.approx(bbox[3], abs=0.01)


def test_polygon_mask_and_mask_to_ring_roundtrip():
    bbox = (4.165, 52.035, 4.225, 52.070)
    grid = make_grid(bbox, res_m=5.0)
    # A small square roughly in the middle of the bbox.
    ring = [
        (4.19, 52.05), (4.20, 52.05), (4.20, 52.06), (4.19, 52.06), (4.19, 52.05),
    ]
    mask = polygon_mask(grid, ring)
    assert mask.dtype == bool
    assert mask.any(), "expected some cells inside the traced square"

    traced = mask_to_ring(mask, grid)
    assert traced[0] == traced[-1], "ring must be closed"
    # Re-rasterizing the traced ring should recover a similar area (within 10%).
    remask = polygon_mask(grid, traced)
    assert remask.sum() == pytest.approx(mask.sum(), rel=0.1)
