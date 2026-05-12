"""xarray DataArray accessor: ``da.octogrid.{compress,to_zarr}``.

Auto-registered on ``import octogrid`` when xarray is importable. The
registration is a no-op (and silent) on systems where xarray is absent —
the rest of octogrid keeps working with plain NumPy arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import numpy as np


if TYPE_CHECKING:
    import xarray as xr
    import zarr.storage

    Store = Union[str, "zarr.storage.Store"]

    from ._octogrid import CompressedField


def _resample_da_to_grid(da: xr.DataArray) -> tuple[Any, np.ndarray]:
    """Return ``(grid, flat_values)`` after locating lat/lon coords.

    Picks the standard short / long coord names, squeezes a singleton
    ``time`` axis, builds a matching octahedral grid, and dispatches to
    the C++ bilinear resampler. Mirrors :func:`octogrid.from_xarray`
    without going through ``compress``.
    """
    from . import (  # noqa: PLC0415
        octahedral_matching_latlon,
        resample_from_latlon,
    )

    lat_name = next((n for n in ("latitude", "lat") if n in da.coords), None)
    lon_name = next((n for n in ("longitude", "lon") if n in da.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError(
            "DataArray must have latitude/longitude (or lat/lon) coords"
        )
    if "time" in da.dims:
        da = da.isel(time=0)
    da = da.transpose(lat_name, lon_name)
    lats = np.asarray(da.coords[lat_name].values, dtype=np.float64)
    lons = np.asarray(da.coords[lon_name].values, dtype=np.float64)
    arr = np.asarray(da.values, dtype=np.float32)
    grid = octahedral_matching_latlon(lons)
    values = resample_from_latlon(grid, lats, lons, arr)
    return grid, values


class OctogridAccessor:
    """``DataArray.octogrid`` namespace.

    Ergonomic entry point for the most common workflow:

    * ``da.octogrid.compress()`` — resample to a matching octahedral grid,
      keep the values uncompressed (raw float32). Halves the point count
      vs the regular lat/lon source without any quality loss beyond the
      bilinear resampling step.

    * ``da.octogrid.compress("zfp_adaptive", epsilon=0.05)`` — same plus
      an in-memory compression layer with a bounded-error guarantee.

    * ``da.octogrid.to_zarr(path, ...)`` — end-to-end pipeline that
      resamples, compresses, and persists in one call.
    """

    def __init__(self, da: xr.DataArray) -> None:
        self._da = da

    def compress(
        self,
        codec: str = "raw",
        **kwargs: Any,  # noqa: ANN401 — codec-specific params (zfp_rate, …)
    ) -> CompressedField:
        """Resample the DataArray onto an octahedral grid and compress.

        ``codec`` defaults to ``"raw"`` (no value-level compression). Pass
        ``"bfloat16"``, ``"uint16"``, ``"zfp"`` (with ``zfp_rate``), or
        ``"zfp_adaptive"`` (with ``epsilon`` and ``max_outlier_frac``) to
        activate a codec.
        """
        from . import compress as _compress  # noqa: PLC0415

        grid, values = _resample_da_to_grid(self._da)
        return _compress(grid, codec, values, **kwargs)

    def to_zarr(
        self,
        store: Store,
        *,
        codec: str = "raw",
        mode: str = "w",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Resample, compress, and persist to a Zarr v3 store in one go."""
        from . import to_zarr as _to_zarr  # noqa: PLC0415

        field = self.compress(codec, **kwargs)
        _to_zarr(field, store, mode=mode)


def register() -> bool:
    """Register the accessor with xarray. Idempotent and silent on failure.

    Returns ``True`` if registration succeeded, ``False`` if xarray is
    not importable.
    """
    try:
        import xarray as xr  # noqa: PLC0415
    except ImportError:
        return False
    xr.register_dataarray_accessor("octogrid")(OctogridAccessor)
    return True
