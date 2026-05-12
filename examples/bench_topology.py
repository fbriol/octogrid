"""Demonstrate the primary win: stop storing fields on regular lat/lon
matrices. Resample to a reduced (octahedral) grid that keeps ~constant
km-spacing globally, then optionally compress.

The empreinte stacks: topology saves ~50 % of points (no polar
oversampling), then per-value compression cuts another factor of ~5.
"""
import sys
import time
import numpy as np
import xarray as xr
import rgrid


def load_msla(path, var):
    ds = xr.open_dataset(path)
    da = ds[var]
    if "time" in da.dims:
        da = da.isel(time=0)
    arr = da.values.astype(np.float32)
    lats = ds["latitude"].values.astype(np.float64)
    lons = ds["longitude"].values.astype(np.float64)
    return lats, lons, arr


def header(label):
    print(f"\n{'='*78}\n  {label}\n{'='*78}")


def measure_query_error(field, qlat, qlon, ref_arr, ref_lats, ref_lons):
    """Compare barycentric interpolation against scipy bilinear on src."""
    from scipy.interpolate import RegularGridInterpolator
    lats_inc = ref_lats[::-1] if ref_lats[0] > ref_lats[-1] else ref_lats
    arr_inc = ref_arr[::-1, :] if ref_lats[0] > ref_lats[-1] else ref_arr
    rgi = RegularGridInterpolator((lats_inc, ref_lons), arr_inc,
                                  method="linear", bounds_error=False,
                                  fill_value=np.nan)
    truth = rgi(np.column_stack([qlat, qlon])).astype(np.float32)
    pred = rgrid.interpolate(field, qlat, qlon, method="barycentric")
    valid = np.isfinite(truth) & np.isfinite(pred)
    err = np.abs(pred[valid] - truth[valid])
    if err.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.sqrt(np.mean(err**2))), float(err.max()), int(err.size)


def benchmark(label, path, var, epsilons=(None, 0.05)):
    header(f"{label}: {var}")
    lats, lons, arr = load_msla(path, var)
    n_src = arr.size
    n_nan = int(np.isnan(arr).sum())
    print(f"source: {arr.shape} lat/lon regular, "
          f"{n_src:,} points  raw float32 = {n_src*4/1e6:.1f} MB  "
          f"NaN = {n_nan:,} ({100*n_nan/n_src:.1f} %)")

    # Build matching octahedral grid.
    oct_grid = rgrid.octahedral_matching_latlon(lons)
    # And the "trivial" regular grid for comparison (uses the source axes).
    if lons.min() < 0:
        # Shift to [0,360) for the lat/lon grid path
        shifted = (lons + 360.0) % 360.0
        order = np.argsort(shifted, kind="stable")
        keep = np.concatenate(([True], np.diff(shifted[order]) > 0))
        order = order[keep]
        lons_pos = shifted[order]
        arr_pos = arr[:, order]
    else:
        lons_pos = lons
        arr_pos = arr
    if lats[0] < lats[-1]:
        lats_dec = lats[::-1]
        arr_dec = arr_pos[::-1, :]
    else:
        lats_dec = lats
        arr_dec = arr_pos
    reg_grid = rgrid.ReducedGrid(latitudes_deg=lats_dec.tolist(),
                                 n_lon=[lons_pos.size] * lats_dec.size)
    flat_reg = arr_dec.flatten()

    # Resample to octahedral.
    t0 = time.perf_counter()
    flat_oct = rgrid.resample_from_latlon(oct_grid, lats, lons, arr)
    t_resample = time.perf_counter() - t0
    print(f"octahedral grid: n_lat={oct_grid.n_rows}, "
          f"n_points={oct_grid.n_points:,}  "
          f"({100*oct_grid.n_points/n_src:.1f} % of source)  "
          f"resample took {t_resample:.2f} s")
    n_nan_oct = int(np.isnan(flat_oct).sum())
    print(f"  resampled NaN: {n_nan_oct:,} "
          f"({100*n_nan_oct/oct_grid.n_points:.1f} %)")

    # Random off-grid queries
    rng = np.random.default_rng(0)
    Q = 50_000
    qlat = rng.uniform(lats_dec[-1] + 1, lats_dec[0] - 1, Q)
    qlon = rng.uniform(0, 360, Q)

    header_fmt = "\n{:<48} {:>7} {:>7} {:>10} {:>10} {:>8}"
    print(header_fmt.format("config", "MB", "%raw", "rmse", "maxerr", "Mq/s"))
    baseline_mb = n_src * 4 / 1e6
    print(f"{'lat/lon float32 (raw)':<48} {baseline_mb:>7.2f} "
          f"{100.0:>7.1f} {'-':>10} {'-':>10} {'-':>8}")

    for eps in epsilons:
        for which in ("regular", "octahedral"):
            grid = reg_grid if which == "regular" else oct_grid
            flat = flat_reg if which == "regular" else flat_oct
            if eps is None:
                # Use bfloat16 as a near-lossless baseline.
                codec = "bfloat16"
                kw = {}
                tag = "bfloat16"
            else:
                codec = "zfp_adaptive"
                kw = {"epsilon": eps, "max_outlier_frac": 0.01}
                tag = f"zfp_adaptive ε={eps}"
            f = rgrid.compress(grid, codec, flat, **kw)
            rmse, mx, _ = measure_query_error(
                f, qlat, qlon, arr, lats, lons)
            # warmup + perf
            rgrid.interpolate(f, qlat[:500], qlon[:500], method="barycentric")
            t0 = time.perf_counter()
            rgrid.interpolate(f, qlat, qlon, method="barycentric")
            dt = time.perf_counter() - t0
            label_full = f"{which:<11} {tag}"
            mb = f.footprint_bytes / 1e6
            pct = 100 * mb / baseline_mb
            print(f"{label_full:<48} {mb:>7.2f} {pct:>7.1f} "
                  f"{rmse:>10.3e} {mx:>10.3e} {Q/dt/1e6:>8.2f}")


if __name__ == "__main__":
    var = sys.argv[1] if len(sys.argv) > 1 else "sla"
    benchmark("MSLA L4", "dt_global_allsat_phy_l4_20190105_20190515.nc", var)
