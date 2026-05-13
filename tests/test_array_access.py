"""Array-like accessors: field.to_numpy() and grid.latitudes/longitudes."""

from __future__ import annotations

import numpy as np
import octogrid


def test_grid_latitudes_match_per_row(small_octahedral):
    """latitudes() repeats the per-row latitude across the row's points."""
    lats_flat = small_octahedral.latitudes()
    assert lats_flat.shape == (small_octahedral.n_points,)
    assert lats_flat.dtype == np.float64
    off = 0
    for r in range(small_octahedral.n_rows):
        n = small_octahedral.n_lon(r)
        np.testing.assert_array_equal(
            lats_flat[off : off + n], np.full(n, small_octahedral.lat_deg(r))
        )
        off += n


def test_grid_longitudes_evenly_spaced_per_row(small_octahedral):
    """Each row's longitudes span [0, 360) with step 360/n_lon."""
    lons_flat = small_octahedral.longitudes()
    assert lons_flat.shape == (small_octahedral.n_points,)
    assert lons_flat.dtype == np.float64
    off = 0
    for r in range(small_octahedral.n_rows):
        n = small_octahedral.n_lon(r)
        expected = np.arange(n) * (360.0 / n)
        np.testing.assert_allclose(lons_flat[off : off + n], expected)
        off += n


def test_to_numpy_round_trip_raw(small_octahedral):
    """Raw codec: bit-exact round-trip."""
    rng = np.random.default_rng(0)
    vals = rng.standard_normal(small_octahedral.n_points).astype(np.float32)
    field = octogrid.compress(small_octahedral, "raw", vals)
    np.testing.assert_array_equal(field.to_numpy(), vals)


def test_to_numpy_round_trip_with_nan(small_octahedral):
    """zfp_adaptive preserves NaN positions via to_numpy."""
    rng = np.random.default_rng(0)
    vals = rng.standard_normal(small_octahedral.n_points).astype(np.float32)
    nan_idx = rng.choice(vals.size, vals.size // 10, replace=False)
    vals[nan_idx] = np.nan
    field = octogrid.compress(
        small_octahedral,
        "zfp_adaptive",
        vals,
        epsilon=0.05,
        max_outlier_frac=0.01,
    )
    decoded = field.to_numpy()
    assert decoded.shape == vals.shape
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(decoded), np.isnan(vals))


def test_time_series_streaming_mean_workflow(small_octahedral):
    """Recipe sanity check: streaming mean across a synthetic time series."""
    rng = np.random.default_rng(1)
    n = small_octahedral.n_points
    # 10 "daily" snapshots: same underlying smooth field plus noise.
    base = rng.standard_normal(n).astype(np.float32)
    snaps = [
        base + (0.1 * rng.standard_normal(n).astype(np.float32))
        for _ in range(10)
    ]
    fields = [octogrid.compress(small_octahedral, "raw", s) for s in snaps]

    total = np.zeros(n, dtype=np.float64)
    n_obs = np.zeros(n, dtype=np.int64)
    for f in fields:
        v = f.to_numpy()
        finite = np.isfinite(v)
        total[finite] += v[finite]
        n_obs[finite] += 1
    mean = (total / n_obs).astype(np.float32)

    # Compare with the dense mean of the source snapshots.
    expected = np.mean(np.stack(snaps), axis=0)
    np.testing.assert_allclose(mean, expected, atol=1e-6)
