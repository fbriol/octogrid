"""rgrid — compact in-memory reduced grids with simple interpolation.

The primary goal: store geophysical fields on a grid whose km-spacing is
roughly constant (reduced/octahedral) instead of a regular lat/lon matrix
that massively oversamples the poles. Compression is a secondary layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._rgrid import CompressedField, ReducedGrid, compress, interpolate


if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


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

    NaN in the source (continents, masked points) propagate to the output
    via the standard IEEE rules: any NaN among the 4 bracketing source
    cells produces NaN at that target point.

    Returns a flat float32 array of length ``grid.n_points``, ready to be
    passed to :func:`rgrid.compress`.
    """
    from scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415

    lats = np.asarray(src_lats, dtype=np.float64)
    lons = np.asarray(src_lons, dtype=np.float64)
    arr = np.asarray(src_arr, dtype=np.float32)

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        arr = arr[::-1, :]
    if lons.min() < 0:
        shifted = (lons + 360.0) % 360.0
        order = np.argsort(shifted, kind="stable")
        keep = np.concatenate(([True], np.diff(shifted[order]) > 0))
        order = order[keep]
        lons = shifted[order]
        arr = arr[:, order]

    lats_q = np.empty(grid.n_points, dtype=np.float64)
    lons_q = np.empty(grid.n_points, dtype=np.float64)
    off = 0
    for r in range(grid.n_rows):
        nl = grid.n_lon(r)
        lats_q[off : off + nl] = grid.lat_deg(r)
        lons_q[off : off + nl] = np.arange(nl) * (360.0 / nl)
        off += nl

    interp = RegularGridInterpolator(
        (lats, lons),
        arr,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    pts = np.column_stack([lats_q, lons_q])
    return interp(pts).astype(np.float32)


__all__ = [
    "CompressedField",
    "ReducedGrid",
    "compress",
    "interpolate",
    "octahedral_matching_latlon",
    "resample_from_latlon",
]
