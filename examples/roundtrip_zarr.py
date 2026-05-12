"""Round-trip test: load a real MSLA field, store to Zarr v3, reload,
compare interpolations. Exercises every codec the library exposes.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

import octogrid


def main() -> int:
    src_path = "dt_global_allsat_phy_l4_20190105_20190515.nc"
    var = sys.argv[1] if len(sys.argv) > 1 else "sla"
    da = xr.open_dataset(src_path)[var]

    # Topology-only field via the high-level helper.
    field0 = octogrid.from_xarray(da)
    print(f"raw  field: {field0.footprint_bytes/1e6:6.2f} MB  "
          f"codec='{field0.codec_name}'  n_points={field0.n_points}")

    # Build other variants by passing the resampled values through compress.
    grid = field0.grid
    lats = da.coords.get("latitude", da.coords.get("lat")).values
    lons = da.coords.get("longitude", da.coords.get("lon")).values
    arr = np.asarray(da.values, dtype=np.float32)
    if "time" in da.dims:
        arr = arr[0]
    values = octogrid.resample_from_latlon(grid, lats, lons, arr)

    variants = [
        ("raw", {}),
        ("bfloat16", {}),
        ("uint16", {}),
        ("zfp", {"zfp_rate": 8}),
        ("zfp_adaptive", {"epsilon": 0.05, "max_outlier_frac": 0.01}),
    ]

    rng = np.random.default_rng(0)
    qlat = rng.uniform(-80, 80, 5_000)
    qlon = rng.uniform(0, 360, 5_000)

    failures = 0
    tmpdir = Path(tempfile.mkdtemp(prefix="octogrid_roundtrip_"))
    try:
        for codec, kw in variants:
            field = octogrid.compress(grid, codec, values, **kw)
            ref_pred = octogrid.interpolate(field, qlat, qlon)
            store = tmpdir / f"{codec}.zarr"
            octogrid.to_zarr(field, str(store))
            on_disk_bytes = sum(p.stat().st_size for p in store.rglob("*")
                                if p.is_file())
            reloaded = octogrid.open(str(store))
            re_pred = octogrid.interpolate(reloaded, qlat, qlon)

            valid = np.isfinite(ref_pred) & np.isfinite(re_pred)
            diff = np.abs(ref_pred[valid] - re_pred[valid])
            max_diff = float(diff.max()) if diff.size else 0.0
            ok = max_diff < 1e-6
            failures += 0 if ok else 1

            ram_mb = field.footprint_bytes / 1e6
            disk_mb = on_disk_bytes / 1e6
            mark = "OK" if ok else "FAIL"
            print(f"[{mark}] {codec:<14} "
                  f"in-RAM={ram_mb:6.2f} MB  on-disk={disk_mb:6.2f} MB  "
                  f"round-trip max diff = {max_diff:.2e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return failures


if __name__ == "__main__":
    sys.exit(main())
