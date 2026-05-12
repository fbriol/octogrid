"""Tests for the ``from_xarray`` ergonomic entry point."""

from __future__ import annotations

import numpy as np
import octogrid
import pytest


xr = pytest.importorskip("xarray")


def _make_dataarray(
    lats, lons, arr, lat_name="latitude", lon_name="longitude"
):
    return xr.DataArray(
        arr,
        coords={lat_name: lats, lon_name: lons},
        dims=(lat_name, lon_name),
        name="field",
    )


def test_from_xarray_topology_only(smooth_latlon):
    lats, lons, arr = smooth_latlon
    da = _make_dataarray(lats, lons, arr)
    field = octogrid.from_xarray(da)
    assert field.codec_name == "raw"
    # Reduced grid has ~half the points of the regular source.
    assert field.n_points < lats.size * lons.size


def test_from_xarray_accepts_short_coord_names(smooth_latlon):
    lats, lons, arr = smooth_latlon
    da = _make_dataarray(lats, lons, arr, "lat", "lon")
    field = octogrid.from_xarray(da)
    assert field.codec_name == "raw"


def test_from_xarray_rejects_missing_coords(smooth_latlon):
    lats, lons, arr = smooth_latlon
    da = xr.DataArray(arr, coords={"y": lats, "x": lons}, dims=("y", "x"))
    with pytest.raises(ValueError, match="latitude"):
        octogrid.from_xarray(da)


def test_from_xarray_collapses_singleton_time(smooth_latlon):
    """A leading time=1 dim is implicitly squeezed before resampling."""
    lats, lons, arr = smooth_latlon
    arr3 = arr[np.newaxis, :, :]
    da = xr.DataArray(
        arr3,
        coords={
            "time": np.array(["2020-01-01"], dtype="datetime64[ns]"),
            "latitude": lats,
            "longitude": lons,
        },
        dims=("time", "latitude", "longitude"),
    )
    field = octogrid.from_xarray(da)
    assert field.codec_name == "raw"


# ---------------------------------------------------------------------------
# DataArray accessor (da.octogrid.*)
# ---------------------------------------------------------------------------


def test_accessor_registered_on_dataarray(smooth_latlon):
    """``import octogrid`` is enough to expose ``da.octogrid``."""
    lats, lons, arr = smooth_latlon
    da = _make_dataarray(lats, lons, arr)
    assert hasattr(da, "octogrid")


def test_accessor_compress_default_is_raw(smooth_latlon):
    lats, lons, arr = smooth_latlon
    da = _make_dataarray(lats, lons, arr)
    field = da.octogrid.compress()
    assert field.codec_name == "raw"


def test_accessor_compress_passes_codec_kwargs(smooth_latlon):
    lats, lons, arr = smooth_latlon
    da = _make_dataarray(lats, lons, arr)
    field = da.octogrid.compress(
        "zfp_adaptive", epsilon=0.05, max_outlier_frac=0.01
    )
    assert field.codec_name in {"zfp-adaptive", "zfp-adaptive+mask"}


def test_accessor_matches_from_xarray(smooth_latlon):
    """The accessor and the free function build identical fields."""
    lats, lons, arr = smooth_latlon
    da = _make_dataarray(lats, lons, arr)
    a = da.octogrid.compress()
    b = octogrid.from_xarray(da)
    assert a.n_points == b.n_points
    # Interpolation at the same points must produce identical output.
    qlat = np.linspace(-60.0, 60.0, 200)
    qlon = np.linspace(0.0, 359.0, 200)
    np.testing.assert_array_equal(
        octogrid.interpolate(a, qlat, qlon),
        octogrid.interpolate(b, qlat, qlon),
    )


def test_accessor_to_zarr_round_trip(tmp_path, smooth_latlon):
    """End-to-end persist with a single call, then reload."""
    lats, lons, arr = smooth_latlon
    da = _make_dataarray(lats, lons, arr)
    store = str(tmp_path / "via_accessor.zarr")
    da.octogrid.to_zarr(
        store, codec="zfp_adaptive", epsilon=0.05, max_outlier_frac=0.01
    )
    reloaded = octogrid.open(store)
    assert reloaded.codec_name in {"zfp-adaptive", "zfp-adaptive+mask"}
