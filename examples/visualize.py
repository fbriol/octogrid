"""Visualize an octogrid field.

Two complementary renderings:

1. **Scatter** — every grid point as a coloured dot, plotted directly at
   its native (lon, lat). Honest representation of the reduced topology:
   you can see the cells thinning out near the poles.

2. **Heatmap on a regular lat/lon viewport** — interpolate the field
   onto a uniform grid (any resolution you choose) and show it as an
   image. This is what most readers expect; it also lets you compare
   visually with NetCDF/GRIB renderings.

Run with::

    python examples/visualize.py [path-to-source.nc] [var]

Default source is the local CMEMS L4 MSLA file we use for the real-data
tests; falls back to a synthetic analytic field when that file is
missing, so the script always produces a plot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import octogrid


def _load_field(path: str | None, var: str) -> octogrid.CompressedField:
    """Build a CompressedField from a NetCDF file or, as a fallback, from
    a smooth analytic field on a 1° × 1° regular grid.
    """
    if path is not None and Path(path).exists():
        import xarray as xr  # noqa: PLC0415

        da = xr.open_dataset(path)[var]
        return octogrid.from_xarray(da)
    # Fallback: synthetic field f(lat, lon) = sin(lat) * cos(2 * lon).
    lats = np.linspace(89.5, -89.5, 180)
    lons = np.linspace(0.5, 359.5, 360)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    arr = (
        np.sin(np.deg2rad(lat_g)) * np.cos(2 * np.deg2rad(lon_g))
    ).astype(np.float32)
    grid = octogrid.octahedral_matching_latlon(lons)
    values = octogrid.resample_from_latlon(grid, lats, lons, arr)
    return octogrid.compress(grid, "raw", values)


def _render_scatter(field, ax):
    """Plot the raw grid as a scatter — shows the reduced topology."""
    values = field.to_numpy()
    lats = field.grid.latitudes()
    lons = field.grid.longitudes()
    sc = ax.scatter(lons, lats, c=values, s=1.0, cmap="RdBu_r")
    ax.set_title(
        f"Native octahedral grid — {field.grid.n_points:,} points",
        fontsize=10,
    )
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    return sc


def _render_heatmap(field, ax, *, nlat=360, nlon=720):
    """Resample the compressed field onto a regular lat/lon viewport."""
    lats = np.linspace(89.75, -89.75, nlat)
    lons = np.linspace(0.25, 359.75, nlon)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    values = octogrid.interpolate(
        field, lat_g.ravel(), lon_g.ravel(), method="barycentric"
    ).reshape(nlat, nlon)
    im = ax.imshow(
        values,
        extent=(0, 360, -90, 90),
        origin="upper",
        cmap="RdBu_r",
        aspect="auto",
    )
    ax.set_title(
        f"Rendered on a regular {nlat}×{nlon} viewport (barycentric)",
        fontsize=10,
    )
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    return im


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else (
        "dt_global_allsat_phy_l4_20190105_20190515.nc"
    )
    var = argv[2] if len(argv) > 2 else "sla"
    field = _load_field(path, var)
    print(
        f"field: codec={field.codec_name}  "
        f"n_points={field.n_points:,}  "
        f"in-RAM={field.footprint_bytes / 1e6:.2f} MB"
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    sc = _render_scatter(field, axes[0])
    fig.colorbar(sc, ax=axes[0], shrink=0.85)
    im = _render_heatmap(field, axes[1])
    fig.colorbar(im, ax=axes[1], shrink=0.85)
    fig.suptitle(f"{var} — octogrid visualization", fontsize=12)

    out = Path("octogrid_visualization.png")
    fig.savefig(out, dpi=130)
    print(f"saved → {out.resolve()}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
