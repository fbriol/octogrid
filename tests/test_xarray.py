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
