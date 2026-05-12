"""End-to-end smoke test of the Python bindings.

Builds an octahedral grid, encodes a smooth analytic field with each codec,
batch-interpolates at random off-grid points, and reports footprint and
accuracy. Mirrors tests/test_smoke.cpp on the Python side.
"""

import sys
import numpy as np
import rgrid


def analytic(lat_deg, lon_deg):
    return np.sin(np.deg2rad(lat_deg)) * np.cos(np.deg2rad(lon_deg))


def main() -> int:
    grid = rgrid.ReducedGrid.octahedral(n_lat=128, base=20)
    n = grid.n_points
    print(
        f'grid: {grid.n_rows} rows, {n} points, '
        f'raw float32 = {n * 4 / 1024:.1f} KB'
    )

    # Fill grid with the analytic field.
    values = np.empty(n, dtype=np.float32)
    for r in range(grid.n_rows):
        lat = grid.lat_deg(r)
        nl = grid.n_lon(r)
        lons = np.arange(nl) * (360.0 / nl)
        # Compute the row-offset via cumulative sum of n_lon.
        offset = sum(grid.n_lon(rr) for rr in range(r))
        values[offset : offset + nl] = analytic(lat, lons)

    # Random off-grid queries.
    rng = np.random.default_rng(0)
    n_queries = 50_000
    qlat = rng.uniform(-80, 80, size=n_queries)
    qlon = rng.uniform(0, 360, size=n_queries)
    truth = analytic(qlat, qlon).astype(np.float32)

    failures = 0
    for codec_name, tol in [('bfloat16', 5e-3), ('uint16', 5e-3)]:
        field = rgrid.compress(grid, codec_name, values)
        ratio = (n * 4) / field.footprint_bytes
        kb = field.footprint_bytes / 1024
        print(f"\n[{codec_name}] {kb:.1f} KB (ratio = {ratio:.2f}x)")

        for method in ('nearest', 'barycentric'):
            pred = rgrid.interpolate(field, qlat, qlon, method=method)
            err = np.abs(pred - truth)
            rmse = float(np.sqrt(np.mean(err**2)))
            print(f'  {method:11s}  rmse = {rmse:.4e}  max = {err.max():.4e}')
            if method == 'barycentric' and rmse > tol:
                print(f'    !! exceeded tolerance {tol}')
                failures += 1

    print('\nOK' if failures == 0 else f'\n{failures} failure(s)')
    return failures


if __name__ == '__main__':
    sys.exit(main())
