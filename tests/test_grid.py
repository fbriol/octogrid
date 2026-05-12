"""Unit tests for the ReducedGrid topology layer."""

from __future__ import annotations

import numpy as np
import octogrid
import pytest


def test_octahedral_factory_point_count():
    """Octahedral grid: N_lon(d) = 20 + 4d, mirrored around the equator.

    For n_lat=64 → half=32, total points = 2 * sum_{d=0..31} (20 + 4d)
    = 2 * (20*32 + 4*31*32/2) = 2 * (640 + 1984) = 5248.
    """
    grid = octogrid.ReducedGrid.octahedral(n_lat=64, base=20)
    assert grid.n_rows == 64
    assert grid.n_points == 5248


def test_octahedral_factory_rejects_odd_n_lat():
    with pytest.raises(ValueError):
        octogrid.ReducedGrid.octahedral(n_lat=63, base=20)


def test_octahedral_lats_decreasing():
    grid = octogrid.ReducedGrid.octahedral(n_lat=32, base=20)
    lats = np.array([grid.lat_deg(r) for r in range(grid.n_rows)])
    assert np.all(np.diff(lats) < 0)


def test_regular_factory_uniform_n_lon():
    grid = octogrid.ReducedGrid.regular(n_lat=8, n_lon=16)
    assert grid.n_rows == 8
    assert all(grid.n_lon(r) == 16 for r in range(grid.n_rows))
    assert grid.n_points == 8 * 16


def test_explicit_constructor_round_trip():
    """User-provided lats/n_lon round-trips through accessors."""
    lats = [80.0, 40.0, 0.0, -40.0, -80.0]
    n_lon = [12, 24, 36, 24, 12]
    grid = octogrid.ReducedGrid(latitudes_deg=lats, n_lon=n_lon)
    assert [grid.lat_deg(r) for r in range(grid.n_rows)] == lats
    assert [grid.n_lon(r) for r in range(grid.n_rows)] == n_lon


def test_octahedral_matching_latlon_equator(regular_axes):
    """Helper picks an octahedral n_lat ≥ source's lon count at equator."""
    _, lons = regular_axes  # 1° spacing → 360 lon points at equator
    grid = octogrid.octahedral_matching_latlon(lons)
    # The equator-row N_lon must be at least 360.
    equator_row = grid.n_rows // 2
    assert grid.n_lon(equator_row) >= 360
