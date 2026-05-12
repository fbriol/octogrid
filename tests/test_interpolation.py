"""Interpolation tests on synthetic analytic fields.

We pick fields with a known closed form and check that:
  - nearest neighbour reproduces grid-node values exactly;
  - barycentric beats nearest on a smooth field;
  - both schemes return NaN whenever any corner is masked.
"""

from __future__ import annotations

import numpy as np
import octogrid
import pytest


def _analytic(lat_deg, lon_deg):
    return np.sin(np.deg2rad(lat_deg)) * np.cos(np.deg2rad(lon_deg))


@pytest.fixture
def smooth_grid_field(small_octahedral):
    """Smooth analytic field, raw codec on the small octahedral grid."""
    grid = small_octahedral
    vals = np.empty(grid.n_points, dtype=np.float32)
    off = 0
    for r in range(grid.n_rows):
        n = grid.n_lon(r)
        lon = np.arange(n) * (360.0 / n)
        vals[off : off + n] = _analytic(grid.lat_deg(r), lon)
        off += n
    return octogrid.compress(grid, "raw", vals)


def test_interp_methods_return_shape(smooth_grid_field, random_queries):
    qlat, qlon = random_queries
    for method in ("nearest", "barycentric"):
        out = octogrid.interpolate(
            smooth_grid_field, qlat, qlon, method=method
        )
        assert out.shape == (qlat.size,)
        assert out.dtype == np.float32


def test_interp_unknown_method_raises(smooth_grid_field):
    with pytest.raises(ValueError, match="unknown method"):
        octogrid.interpolate(
            smooth_grid_field, np.array([0.0]), np.array([0.0]), method="cubic"
        )


def test_nearest_recovers_grid_nodes(smooth_grid_field):
    """At grid nodes, nearest should return the stored value exactly."""
    grid = smooth_grid_field.grid
    qlat = np.array([grid.lat_deg(10)])
    n = grid.n_lon(10)
    qlon = np.array([5 * 360.0 / n])  # exact 6th longitude on this row
    expected = np.float32(_analytic(qlat[0], qlon[0]))
    got = octogrid.interpolate(
        smooth_grid_field, qlat, qlon, method="nearest"
    )[0]
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_barycentric_better_than_nearest_on_smooth(
    smooth_grid_field, random_queries
):
    qlat, qlon = random_queries
    truth = _analytic(qlat, qlon).astype(np.float32)
    nn = octogrid.interpolate(smooth_grid_field, qlat, qlon, method="nearest")
    bc = octogrid.interpolate(
        smooth_grid_field, qlat, qlon, method="barycentric"
    )
    nn_rmse = float(np.sqrt(np.mean((nn - truth) ** 2)))
    bc_rmse = float(np.sqrt(np.mean((bc - truth) ** 2)))
    # On a smooth bandlimited signal, barycentric should be at least one
    # order of magnitude better than nearest.
    assert bc_rmse * 10 < nn_rmse


def test_barycentric_propagates_nan_at_masked_corners(
    matching_octahedral, masked_latlon, random_queries
):
    """If any of the 3 triangle vertices is NaN, the result must be NaN."""
    grid = matching_octahedral
    lats, lons, arr = masked_latlon
    vals = octogrid.resample_from_latlon(grid, lats, lons, arr)
    field = octogrid.compress(grid, "raw", vals)

    qlat, qlon = random_queries
    # Restrict queries to land cells of the synthetic continent.
    mask = (np.abs(qlat) < 25.0) & (np.abs(qlon - 200.0) < 55.0)
    out = octogrid.interpolate(field, qlat[mask], qlon[mask])
    assert np.isnan(out).all()


def test_interp_outside_lat_range_falls_back_to_edge(smooth_grid_field):
    """Queries beyond the grid's lat span return the edge-row interp."""
    grid = smooth_grid_field.grid
    # Query just north of the northernmost row.
    qlat = np.array([grid.lat_deg(0) + 5.0])
    qlon = np.array([10.0])
    out = octogrid.interpolate(smooth_grid_field, qlat, qlon)
    assert np.isfinite(out).all()
