"""Shared pytest fixtures for the octogrid Python test suite.

All fixtures use synthetic analytic fields — no third-party data file is
required. Tests that opt-in to a real CMEMS/CNES file should mark themselves
``@pytest.mark.real_data`` so that they are skipped by default in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# Make the built extension importable without installing the wheel.
_BUILD_PKG = Path(__file__).resolve().parents[1] / "build" / "python_pkg"
if _BUILD_PKG.exists():
    sys.path.insert(0, str(_BUILD_PKG))


import octogrid  # noqa: E402 (imported after sys.path tweak)


# ---------------------------------------------------------------------------
# Analytic fields
# ---------------------------------------------------------------------------


def _smooth(lat_deg, lon_deg):
    """f(lat, lon) = sin(lat) * cos(lon) — bounded in [-1, 1], smooth on S²."""
    return np.sin(np.deg2rad(lat_deg)) * np.cos(np.deg2rad(lon_deg))


def _smooth_with_continents(lat_deg, lon_deg):
    """Same smooth field, masked over a rectangular pseudo-continent.

    NaN over |lat| < 30 AND |lon - 200| < 60 — emulates a land mask
    spanning low latitudes (no overlap with poles).
    """
    lat_g, lon_g = np.meshgrid(lat_deg, lon_deg, indexing="ij")
    arr = _smooth(lat_g, lon_g).astype(np.float32)
    mask = (np.abs(lat_g) < 30.0) & (np.abs(lon_g - 200.0) < 60.0)
    arr[mask] = np.nan
    return arr


# ---------------------------------------------------------------------------
# Source grids (regular lat/lon, in canonical orientation)
# ---------------------------------------------------------------------------


@pytest.fixture
def regular_axes():
    """A 1deg by 1deg regular lat/lon grid: lats decreasing, lons increasing."""
    lats = np.linspace(89.5, -89.5, 180)
    lons = np.linspace(0.5, 359.5, 360)
    return lats, lons


@pytest.fixture
def smooth_latlon(regular_axes):
    """Smooth analytic field on the regular axes, no NaN."""
    lats, lons = regular_axes
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    arr = _smooth(lat_g, lon_g).astype(np.float32)
    return lats, lons, arr


@pytest.fixture
def masked_latlon(regular_axes):
    """Analytic field with a rectangular NaN mask (synthetic continent)."""
    lats, lons = regular_axes
    arr = _smooth_with_continents(lats, lons)
    return lats, lons, arr


# ---------------------------------------------------------------------------
# Reduced grids
# ---------------------------------------------------------------------------


@pytest.fixture
def small_octahedral():
    """A small octahedral grid (n_lat=64) for fast unit tests."""
    return octogrid.ReducedGrid.octahedral(n_lat=64, base=20)


@pytest.fixture
def matching_octahedral(regular_axes):
    """An octahedral grid sized to the regular lat/lon source."""
    _, lons = regular_axes
    return octogrid.octahedral_matching_latlon(lons)


@pytest.fixture
def smooth_field_raw(matching_octahedral, smooth_latlon):
    """A raw-codec field built by resampling the smooth source."""
    grid = matching_octahedral
    lats, lons, arr = smooth_latlon
    values = octogrid.resample_from_latlon(grid, lats, lons, arr)
    return octogrid.compress(grid, "raw", values), values


# ---------------------------------------------------------------------------
# Query batches
# ---------------------------------------------------------------------------


@pytest.fixture
def random_queries():
    """Reproducible random off-grid query points away from the poles."""
    rng = np.random.default_rng(0)
    n = 2_000
    qlat = rng.uniform(-80.0, 80.0, n)
    qlon = rng.uniform(0.0, 360.0, n)
    return qlat, qlon
