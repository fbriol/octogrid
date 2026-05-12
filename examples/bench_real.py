"""Benchmark on real CNES/CLS geophysical fields.

For each field: load, convert to float32, fill NaN with the field's mean
(NaN handling is a separate concern outside this proto's scope), build a
ReducedGrid that matches the file's lat/lon, then run each codec.

We report compression ratio, decode-roundtrip error (excluding originally-
NaN points), and barycentric interpolation throughput.
"""
import sys
import time
import numpy as np
import xarray as xr
import octogrid


def load_field(path, var, lat_name="latitude", lon_name="longitude",
               transpose=False, subsample=1):
    """Load a 2D variable as float32; return (lats, lons, values, mask)."""
    ds = xr.open_dataset(path)
    da = ds[var]
    if "time" in da.dims:
        da = da.isel(time=0)
    if transpose:
        # File stores (lon, lat) instead of (lat, lon).
        da = da.transpose(lat_name, lon_name)
    arr = da.values.astype(np.float32)
    lats = ds[lat_name].values.astype(np.float64)
    lons = ds[lon_name].values.astype(np.float64)
    # Drop duplicates at the wrap edge (e.g. 360 == 0 or 180/-180 both).
    if subsample > 1:
        lats = lats[::subsample]
        lons = lons[::subsample]
        arr = arr[::subsample, ::subsample]
    if lats[0] < lats[-1]:
        lats = lats[::-1]
        arr = arr[::-1, :]
    if lons.min() < 0:
        shifted = (lons + 360.0) % 360.0
        order = np.argsort(shifted, kind="stable")
        # Remove duplicate longitudes after shift (e.g., -180 and 180).
        keep = np.concatenate(([True], np.diff(shifted[order]) > 0))
        order = order[keep]
        lons = shifted[order]
        arr = arr[:, order]
    nan_mask = ~np.isfinite(arr)
    return lats, lons, arr, nan_mask


def build_grid_from_axes(lats, lons):
    nlat = lats.size
    nlon = lons.size
    return octogrid.ReducedGrid(
        latitudes_deg=lats.tolist(), n_lon=[nlon] * nlat,
    )


def run(name, path, var, **load_kw):
    print(f"\n{'='*72}\n  {name}: {path}\n{'='*72}")
    lats, lons, arr, nan_mask = load_field(path, var, **load_kw)
    print(f"shape: {arr.shape}, lat span: [{lats[-1]:.2f}, {lats[0]:.2f}], "
          f"lon span: [{lons[0]:.2f}, {lons[-1]:.2f}]")
    n_total = arr.size
    n_nan = int(nan_mask.sum())
    print(f"points: {n_total:,}  NaN: {n_nan:,} ({100*n_nan/n_total:.1f}%)  "
          f"raw float32: {n_total*4/1e6:.1f} MB")

    finite = arr[~nan_mask]
    vmin, vmax = float(finite.min()), float(finite.max())
    std = float(finite.std())
    print(f"finite range: [{vmin:.3f}, {vmax:.3f}]  std: {std:.4f}")
    # Codecs that support NaN natively (zfp_adaptive) get the raw array.
    # bfloat16/uint16/zfp need NaN replaced with mean (lossy on land).
    fill = float(finite.mean()) if finite.size else 0.0
    flat_raw = arr.flatten()  # includes NaN
    flat_filled = arr.copy()
    flat_filled[nan_mask] = fill
    flat_filled = flat_filled.flatten()

    grid = build_grid_from_axes(lats, lons)

    # Random off-grid queries within the valid lat band, away from poles
    # (poles often coincide with NaN bands).
    rng = np.random.default_rng(0)
    Q = 50_000
    qlat = rng.uniform(lats[-1] + 1, lats[0] - 1, size=Q)
    qlon = rng.uniform(0, 360, size=Q)

    configs = [
        ("bfloat16",      {}),
        ("zfp r=8",       {"codec": "zfp", "zfp_rate": 8}),
        ("zfp r=4",       {"codec": "zfp", "zfp_rate": 4}),
        ("zfp_ad ε=σ/100", {"codec": "zfp_adaptive", "epsilon": std/100.0}),
        ("zfp_ad ε=σ/30",  {"codec": "zfp_adaptive", "epsilon": std/30.0}),
        ("zfp_ad ε=σ/10",  {"codec": "zfp_adaptive", "epsilon": std/10.0}),
    ]
    print(f"\n{'codec':<18} {'MB':>8} {'ratio':>8} {'rmse':>11} "
          f"{'max err':>11} {'pct ≤ε':>8} {'Mpts/s':>8}")

    # Reference: uint16 on the filled array (it doesn't handle NaN).
    ref = octogrid.compress(grid, "uint16", flat_filled)
    ref_pred = octogrid.interpolate(ref, qlat, qlon, method="barycentric")
    print(f"{'uint16 (ref)':<18} {ref.footprint_bytes/1e6:>8.1f} "
          f"{(n_total*4)/ref.footprint_bytes:>7.2f}x {'-':>11} {'-':>11} "
          f"{'-':>8} {'-':>8}")

    for label, kw in configs:
        codec_name = kw.pop("codec", label.split()[0])
        # zfp_adaptive accepts NaN natively; others need the filled array.
        src = flat_raw if codec_name == "zfp_adaptive" else flat_filled
        try:
            f = octogrid.compress(grid, codec_name, src, **kw)
        except Exception as e:
            print(f"{label:<18} ERR: {e}")
            continue
        ratio = (n_total * 4) / f.footprint_bytes

        # warmup
        octogrid.interpolate(f, qlat[:500], qlon[:500], method="barycentric")
        t0 = time.perf_counter()
        pred = octogrid.interpolate(f, qlat, qlon, method="barycentric")
        dt = time.perf_counter() - t0
        # Only compare on points where neither reference nor codec returned NaN.
        valid = np.isfinite(pred) & np.isfinite(ref_pred)
        err = np.abs(pred[valid] - ref_pred[valid])
        rmse = float(np.sqrt(np.mean(err ** 2))) if err.size else 0.0
        max_err = float(err.max()) if err.size else 0.0
        # For zfp_adaptive: fraction of queries within configured epsilon.
        if "epsilon" in kw:
            pct_ok = 100.0 * float((err <= kw["epsilon"]).mean())
        else:
            pct_ok = float("nan")
        print(f"{label:<18} {f.footprint_bytes/1e6:>8.1f} {ratio:>7.2f}x "
              f"{rmse:>11.2e} {max_err:>11.2e} "
              f"{pct_ok if not np.isnan(pct_ok) else '-':>8} "
              f"{Q/dt/1e6:>8.2f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    geco = "../geco_sad"
    if which in ("all", "msla"):
        msla_path = "dt_global_allsat_phy_l4_20190105_20190515.nc"
        for var in ("sla", "adt", "ugos", "vgos"):
            run(f"MSLA::{var}", msla_path, var)
    if which in ("all", "mdt"):
        mdt_path = (
            f"{geco}/mean_dynamic_topography"
            f"_cnes_cls_2024_20240429T000000_v100.nc"
        )
        run("MDT", mdt_path, "mdt")
    if which in ("all", "mss"):
        mss_path = (
            f"{geco}/mean_sea_surface"
            f"_cnes_cls_hydrid_2023_20240425T000000_v101.nc"
        )
        run("MSS", mss_path, "mss")
    if which in ("all", "bathy"):
        run(
            "BATHY",
            f"{geco}/depth_or_elevation_20210215T000000_v100.nc",
            "elevation",
            lat_name="lat", lon_name="lon",
            transpose=True, subsample=4,
        )
