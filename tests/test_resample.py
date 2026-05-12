"""Tests for the C++ bilinear resampling from a regular lat/lon source."""

from __future__ import annotations

import numpy as np
import octogrid
import pytest


def _analytic(lat_deg, lon_deg):
    return np.sin(np.deg2rad(lat_deg)) * np.cos(np.deg2rad(lon_deg))


def test_resample_reproduces_smooth_source(matching_octahedral, smooth_latlon):
    """Resampled values match the closed-form analytic field."""
    grid = matching_octahedral
    lats, lons, arr = smooth_latlon
    values = octogrid.resample_from_latlon(grid, lats, lons, arr)
    assert values.shape == (grid.n_points,)
    assert values.dtype == np.float32

    # Compare against the analytic field evaluated at the grid nodes.
    expected = np.empty(grid.n_points, dtype=np.float32)
    off = 0
    for r in range(grid.n_rows):
        n = grid.n_lon(r)
        lon = np.arange(n) * (360.0 / n)
        expected[off : off + n] = _analytic(grid.lat_deg(r), lon)
        off += n
    # The 1° source samples a smoothly bandlimited signal — bilinear should
    # match the analytic value to ~source-cell resolution.
    err = np.abs(values - expected)
    assert err.max() < 5e-3


def test_resample_handles_negative_longitude(smooth_latlon):
    """A source given in [-180, 180] is shifted into [0, 360) transparently."""
    lats, lons, arr = smooth_latlon
    # Build a shifted source: lons in [-180, 180), same arr columns reordered.
    shifted = ((lons + 180.0) % 360.0) - 180.0
    order = np.argsort(shifted)
    lons_shifted = shifted[order]
    arr_shifted = arr[:, order]

    grid = octogrid.octahedral_matching_latlon(lons)
    out_ref = octogrid.resample_from_latlon(grid, lats, lons, arr)
    out_neg = octogrid.resample_from_latlon(
        grid, lats, lons_shifted, arr_shifted
    )
    np.testing.assert_allclose(out_ref, out_neg, atol=1e-6)


def test_resample_handles_reversed_latitude(smooth_latlon):
    """Source with increasing lats is auto-flipped."""
    lats, lons, arr = smooth_latlon
    lats_inc = lats[::-1]
    arr_inc = arr[::-1, :]
    grid = octogrid.octahedral_matching_latlon(lons)
    out_ref = octogrid.resample_from_latlon(grid, lats, lons, arr)
    out_inc = octogrid.resample_from_latlon(grid, lats_inc, lons, arr_inc)
    np.testing.assert_allclose(out_ref, out_inc, atol=1e-6)


def test_resample_propagates_nan(matching_octahedral, masked_latlon):
    """NaN among the bracketing source cells produces NaN in the output."""
    lats, lons, arr = masked_latlon
    grid = matching_octahedral
    values = octogrid.resample_from_latlon(grid, lats, lons, arr)
    # The synthetic continent covers |lat|<30 ∧ |lon-200|<60 → roughly
    # 30/180 * 120/360 = ~5.5% of the area; require at least 1% NaN.
    n_nan = int(np.isnan(values).sum())
    assert n_nan / values.size > 0.01


def test_resample_rejects_shape_mismatch(matching_octahedral):
    """Wrong source shape should error rather than silently corrupt memory."""
    grid = matching_octahedral
    lats = np.linspace(89.5, -89.5, 180)
    lons = np.linspace(0.5, 359.5, 360)
    bad = np.zeros((180, 359), dtype=np.float32)
    with pytest.raises(ValueError):
        octogrid.resample_from_latlon(grid, lats, lons, bad)
