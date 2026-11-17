import numpy as np

from zandmotor.geometry import crop_grid, make_grid, polygon_mask


def test_crop_grid_is_aligned_with_the_full_grid():
    grid = make_grid((4.165, 52.035, 4.225, 52.070), res_m=5.0)
    ring = [(4.19, 52.05), (4.20, 52.05), (4.20, 52.06), (4.19, 52.06), (4.19, 52.05)]
    full = polygon_mask(grid, ring)
    sub, (rs, cs) = crop_grid(grid, full, pad_px=12)
    assert (sub.height, sub.width) == full[rs, cs].shape
    assert sub.width < grid.width and sub.height < grid.height
    # rasterizing the same polygon on the sub-grid gives exactly the cut-out
    assert np.array_equal(polygon_mask(sub, ring), full[rs, cs])
    assert sub.cell_area_m2 == grid.cell_area_m2
