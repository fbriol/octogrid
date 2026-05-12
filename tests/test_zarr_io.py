"""Zarr v3 persistence: round-trip every codec and check layout invariants."""

from __future__ import annotations

import numpy as np
import octogrid
import pytest
import zarr


CODEC_PARAMS = [
    ("raw", {}),
    ("bfloat16", {}),
    ("uint16", {}),
    ("zfp", {"zfp_rate": 8}),
    ("zfp_adaptive", {"epsilon": 0.05, "max_outlier_frac": 0.01}),
]


@pytest.fixture
def smooth_values(matching_octahedral, smooth_latlon):
    grid = matching_octahedral
    lats, lons, arr = smooth_latlon
    return grid, octogrid.resample_from_latlon(grid, lats, lons, arr)


@pytest.mark.parametrize(("codec", "kwargs"), CODEC_PARAMS)
def test_zarr_round_trip_preserves_decoded_values(
    tmp_path, smooth_values, codec, kwargs
):
    """Saving then reloading must yield bit-identical interpolation output."""
    grid, values = smooth_values
    field = octogrid.compress(grid, codec, values, **kwargs)
    store = str(tmp_path / f"{codec}.zarr")
    octogrid.to_zarr(field, store)

    reloaded = octogrid.open(store)

    rng = np.random.default_rng(0)
    qlat = rng.uniform(-80.0, 80.0, 1000)
    qlon = rng.uniform(0.0, 360.0, 1000)
    a = octogrid.interpolate(field, qlat, qlon)
    b = octogrid.interpolate(reloaded, qlat, qlon)
    np.testing.assert_array_equal(a, b)


def test_zarr_layout_is_self_describing(tmp_path, smooth_values):
    """The on-disk store exposes the grid topology as plain Zarr arrays."""
    grid, values = smooth_values
    field = octogrid.compress(grid, "raw", values)
    store = str(tmp_path / "topo.zarr")
    octogrid.to_zarr(field, store)

    root = zarr.open_group(store, mode="r")
    expected_keys = {"grid_latitudes_deg", "grid_n_lon", "codec_blob"}
    assert expected_keys.issubset(set(root.array_keys()))
    assert root.attrs["format_version"] == octogrid.ZARR_FORMAT_VERSION
    assert root.attrs["codec_name"] == "raw"
    assert root.attrs["n_points"] == grid.n_points
    assert root.attrs["n_rows"] == grid.n_rows
    np.testing.assert_array_equal(
        np.asarray(root["grid_latitudes_deg"][:]),
        np.asarray(
            [grid.lat_deg(r) for r in range(grid.n_rows)], dtype=np.float64
        ),
    )


def test_zarr_open_rejects_unknown_version(tmp_path, smooth_values):
    """A wrong format_version attribute must raise rather than mis-decode."""
    grid, values = smooth_values
    field = octogrid.compress(grid, "raw", values)
    store = str(tmp_path / "bad.zarr")
    octogrid.to_zarr(field, store)
    root = zarr.open_group(store, mode="a")
    root.attrs["format_version"] = 99
    with pytest.raises(ValueError, match="unsupported"):
        octogrid.open(store)


def test_zarr_handles_nan_field(tmp_path, matching_octahedral, masked_latlon):
    """NaN mask survives a Zarr round-trip with the adaptive ZFP codec."""
    grid = matching_octahedral
    lats, lons, arr = masked_latlon
    values = octogrid.resample_from_latlon(grid, lats, lons, arr)
    field = octogrid.compress(
        grid, "zfp_adaptive", values, epsilon=0.05, max_outlier_frac=0.01
    )
    store = str(tmp_path / "nan.zarr")
    octogrid.to_zarr(field, store)
    reloaded = octogrid.open(store)

    rng = np.random.default_rng(2)
    # Query directly over the masked region — both fields must agree on NaN.
    qlat = rng.uniform(-20.0, 20.0, 500)
    qlon = rng.uniform(150.0, 250.0, 500)
    a = octogrid.interpolate(field, qlat, qlon)
    b = octogrid.interpolate(reloaded, qlat, qlon)
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    np.testing.assert_array_equal(nan_a, nan_b)
    # Wherever both are finite, values must match exactly.
    finite = ~nan_a
    np.testing.assert_array_equal(a[finite], b[finite])
