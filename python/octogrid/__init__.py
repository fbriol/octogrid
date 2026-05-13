"""octogrid — compact in-memory reduced grids with simple interpolation.

The primary goal: store geophysical fields on a grid whose km-spacing is
roughly constant (reduced/octahedral) instead of a regular lat/lon matrix
that massively oversamples the poles. Compression is a secondary layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

from ._octogrid import (
    CompressedField,
    ReducedGrid,
    _field_from_blob,
    _serialize_codec,
    compress,
    interpolate,
)
from ._octogrid import resample_from_latlon as _resample_native
from ._stack import STACK_FORMAT_VERSION, FieldStack, open_stack
from ._xarray_accessor import register as _register_accessor


# Best-effort xarray accessor registration. Silent no-op when xarray is
# not importable in the current environment.
_register_accessor()


if TYPE_CHECKING:
    import xarray as xr
    import zarr.storage
    from numpy.typing import ArrayLike, NDArray

    Store = Union[str, "zarr.storage.Store"]

#: On-disk format version. Bump on incompatible schema changes.
ZARR_FORMAT_VERSION = 1


def octahedral_matching_latlon(
    src_lons: ArrayLike, *, base: int = 20
) -> ReducedGrid:
    """Build an octahedral grid sized to a regular lat/lon source.

    Picks ``n_lat`` such that the equatorial longitude count of the
    octahedral grid is greater than or equal to the source's lon count.
    The number of latitude lines is constrained to be even, as the
    octahedral formula requires.
    """
    arr = np.asarray(src_lons, dtype=np.float64)
    dlon = float(np.abs(np.diff(arr)).min())
    nlon_eq_target = round(360.0 / dlon)
    n_lat = 2 * max(1, (nlon_eq_target - base + 3) // 4 + 1)
    return ReducedGrid.octahedral(n_lat=n_lat, base=base)


def resample_from_latlon(
    grid: ReducedGrid,
    src_lats: ArrayLike,
    src_lons: ArrayLike,
    src_arr: ArrayLike,
) -> NDArray[np.float32]:
    """Bilinearly resample a regular lat/lon array onto a reduced grid.

    Normalizes axis orientation (lats decreasing N→S, lons increasing in
    ``[0, 360)``) before dispatching to the C++ bilinear core. NaN in the
    source propagates to the output via the IEEE rules.

    Returns a flat float32 array of length ``grid.n_points``, ready to
    be passed to :func:`octogrid.compress`.
    """
    lats = np.ascontiguousarray(np.asarray(src_lats, dtype=np.float64))
    lons = np.ascontiguousarray(np.asarray(src_lons, dtype=np.float64))
    arr = np.ascontiguousarray(np.asarray(src_arr, dtype=np.float32))

    if lats[0] < lats[-1]:
        lats = np.ascontiguousarray(lats[::-1])
        arr = np.ascontiguousarray(arr[::-1, :])
    if lons.min() < 0:
        shifted = (lons + 360.0) % 360.0
        order = np.argsort(shifted, kind="stable")
        keep = np.concatenate(([True], np.diff(shifted[order]) > 0))
        order = order[keep]
        lons = np.ascontiguousarray(shifted[order])
        arr = np.ascontiguousarray(arr[:, order])
    return _resample_native(grid, lats, lons, arr)


def from_xarray(da: xr.DataArray) -> CompressedField:
    """Build a topology-only field from a regular lat/lon DataArray.

    Detects the lat/lon coordinates by name (``latitude``/``lat``,
    ``longitude``/``lon``), builds a matching octahedral grid, resamples
    via the C++ bilinear core, and wraps the result with the raw float32
    codec (no compression).

    Call :func:`octogrid.compress` afterwards to add a compression layer.
    """
    lat_name = next((n for n in ("latitude", "lat") if n in da.coords), None)
    lon_name = next((n for n in ("longitude", "lon") if n in da.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError(
            "DataArray must have latitude/longitude (or lat/lon) coords"
        )
    if "time" in da.dims:
        da = da.isel(time=0)
    # Make sure the spatial axes are last and in the expected order.
    da = da.transpose(lat_name, lon_name)
    lats = np.asarray(da.coords[lat_name].values, dtype=np.float64)
    lons = np.asarray(da.coords[lon_name].values, dtype=np.float64)
    arr = np.asarray(da.values, dtype=np.float32)
    grid = octahedral_matching_latlon(lons)
    values = resample_from_latlon(grid, lats, lons, arr)
    return compress(grid, "raw", values)


# ---------------------------------------------------------------------------
# Zarr v3 persistence
# ---------------------------------------------------------------------------
#
# Layout (a Zarr v3 group, autodescriptive):
#
#   <store>/
#     ├── (group attrs:
#     │       format_version, codec_name,
#     │       grid_kind, n_points, n_rows)
#     ├── grid_latitudes_deg   float64  (n_rows,)
#     ├── grid_n_lon           uint32   (n_rows,)
#     └── codec_blob           uint8    (n_bytes,)   opaque per-codec payload
#
# The codec blob layout is owned by the C++ Codec subclass; consumers without
# octogrid can still inspect grid topology via the standard Zarr arrays.


def to_zarr(
    field: CompressedField,
    store: Store,
    *,
    mode: str = "w",
) -> None:
    """Persist a ``CompressedField`` to a Zarr v3 store.

    ``store`` accepts everything ``zarr.open_group`` accepts: a local path,
    an ``fsspec`` URL, or a pre-opened ``zarr.storage.Store``.
    """
    import zarr  # noqa: PLC0415

    grid = field.grid
    lats = np.asarray(
        [grid.lat_deg(r) for r in range(grid.n_rows)], dtype=np.float64
    )
    n_lon = np.asarray(
        [grid.n_lon(r) for r in range(grid.n_rows)], dtype=np.uint32
    )
    blob = _serialize_codec(field)

    root = zarr.open_group(store, mode=mode, zarr_format=3)
    root.attrs.update(
        {
            "format_version": ZARR_FORMAT_VERSION,
            "codec_name": field.codec_name,
            "grid_kind": "reduced",
            "n_points": int(grid.n_points),
            "n_rows": int(grid.n_rows),
        }
    )
    root.create_array(
        "grid_latitudes_deg",
        shape=lats.shape,
        dtype=lats.dtype,
    )[:] = lats
    root.create_array(
        "grid_n_lon",
        shape=n_lon.shape,
        dtype=n_lon.dtype,
    )[:] = n_lon
    root.create_array(
        "codec_blob",
        shape=blob.shape,
        dtype=blob.dtype,
    )[:] = blob


def open(store: Store) -> CompressedField:
    """Reload a ``CompressedField`` previously written by :func:`to_zarr`.

    The codec is reconstructed by name from the group attributes and the
    payload in ``codec_blob``. The grid topology is read from the explicit
    ``grid_latitudes_deg`` / ``grid_n_lon`` arrays.
    """
    import zarr  # noqa: PLC0415

    root = zarr.open_group(store, mode="r")
    attrs = dict(root.attrs)
    version = attrs.get("format_version")
    if version != ZARR_FORMAT_VERSION:
        raise ValueError(f"unsupported octogrid format version: {version!r}")
    lats = np.asarray(root["grid_latitudes_deg"][:], dtype=np.float64)
    n_lon = np.asarray(root["grid_n_lon"][:], dtype=np.uint32)
    blob = np.asarray(root["codec_blob"][:], dtype=np.uint8)

    grid = ReducedGrid(
        latitudes_deg=lats.tolist(),
        n_lon=n_lon.tolist(),
    )
    codec_name = attrs["codec_name"]
    # NaN-aware AdaptiveZfp reports a slightly different runtime name; the
    # serialized form distinguishes by an internal flag, so route both
    # variants through the same deserializer.
    if codec_name.startswith("zfp-adaptive"):
        codec_name = "zfp-adaptive"
    return _field_from_blob(grid, codec_name, blob)


__all__ = [
    "STACK_FORMAT_VERSION",
    "ZARR_FORMAT_VERSION",
    "CompressedField",
    "FieldStack",
    "ReducedGrid",
    "compress",
    "from_xarray",
    "interpolate",
    "octahedral_matching_latlon",
    "open",
    "open_stack",
    "resample_from_latlon",
    "to_zarr",
]
