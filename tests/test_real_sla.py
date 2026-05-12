"""Opt-in tests against a real (non-distributable) CMEMS L4 SLA file.

The file is not redistributed in the repository for licensing reasons.
Provide it locally as ``dt_global_allsat_phy_l4_*.nc`` next to the working
directory and these tests will run; otherwise they skip silently.

Run with::

    pytest -m real_data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import octogrid
import pytest


xr = pytest.importorskip("xarray")

_CANDIDATES = sorted(Path.cwd().glob("dt_global_allsat_phy_l4_*.nc"))
_SOURCE = _CANDIDATES[0] if _CANDIDATES else None

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        _SOURCE is None,
        reason="no local dt_global_allsat_phy_l4_*.nc file available",
    ),
]


def test_real_sla_topology_only_reduces_point_count():
    da = xr.open_dataset(_SOURCE)["sla"]
    field = octogrid.from_xarray(da)
    src_points = int(np.prod([da.sizes[d] for d in ("latitude", "longitude")]))
    assert field.n_points < src_points * 0.6


def test_real_sla_adaptive_round_trip_within_tolerance(tmp_path):
    da = xr.open_dataset(_SOURCE)["sla"]
    grid = octogrid.octahedral_matching_latlon(da.longitude.values)
    values = octogrid.resample_from_latlon(
        grid,
        da.latitude.values,
        da.longitude.values,
        np.asarray(da.isel(time=0).values, dtype=np.float32),
    )
    epsilon = 0.05
    field = octogrid.compress(
        grid, "zfp_adaptive", values, epsilon=epsilon, max_outlier_frac=0.01
    )

    octogrid.to_zarr(field, str(tmp_path / "sla.zarr"))
    reloaded = octogrid.open(str(tmp_path / "sla.zarr"))

    rng = np.random.default_rng(0)
    qlat = rng.uniform(-80.0, 80.0, 5_000)
    qlon = rng.uniform(0.0, 360.0, 5_000)
    a = octogrid.interpolate(field, qlat, qlon)
    b = octogrid.interpolate(reloaded, qlat, qlon)
    finite = np.isfinite(a) & np.isfinite(b)
    np.testing.assert_array_equal(a[finite], b[finite])
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
