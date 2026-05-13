# octogrid

> **Status: experimental (0.1.0.dev).** API may change before the first
> stable release. The C++ core is feature-complete and tested.

`octogrid` stores 2-D geophysical fields on **octahedral reduced grids**
(the same family used by ECMWF's IFS since 2016) instead of regular
lat/lon matrices. The grid keeps a roughly constant kilometric spacing
at every latitude, which halves the point count without losing
resolution at the equator. A pluggable codec layer can compress the
stored values on top, with a strict error bound for the adaptive ZFP
backend.

The library deliberately does not handle disk I/O beyond
**Zarr v3** persistence; NetCDF/HDF5 ingestion stays the
responsibility of `xarray`.

---

## Why a reduced grid?

A global 0.25° lat/lon grid (1440 × 720) puts ≈ 4 000 cells in every
longitude row, regardless of latitude. At 80° N the same cells are
only ≈ 4.8 km wide; at the equator they are 28 km wide. The polar
oversampling is pure waste — it cannot resolve anything the equatorial
spacing cannot. An *octahedral* reduced grid shrinks the row width with
`cos φ` so each cell stays close to its equatorial size at the ground.

For the CMEMS L4 MSLA product we tested, this is a constant **×2
reduction in point count, with no loss in effective resolution**:

```
regular 1° × 1° lat/lon → 1 036 800 points
matching octahedral      →   519 760 points   (50.1 %)
```

Adding an in-memory codec on top can take the empreinte further — see
the numbers below.

---

## Installation

For development right now (no PyPI release yet):

```sh
git clone <repo>
cd octogrid
cmake -S . -B build
cmake --build build -j
pip install nanobind xarray "zarr>=3"   # if not already in your env
export PYTHONPATH=$PWD/build/python_pkg
```

Once the 0.1 lands on PyPI and conda-forge, the install will collapse to:

```sh
pip install octogrid       # or
conda install -c conda-forge octogrid
```

ZFP support is fetched automatically by CMake (`FetchContent`).

---

## Quickstart

```python
import xarray as xr
import octogrid                              # registers the xarray accessor

src = xr.open_dataset("dt_global_..._sla.nc").sla

# 1. Topology only — halves the point count, no value-level loss.
field = src.octogrid.compress()

# 2. Topology + compression with a bounded-error guarantee.
field = src.octogrid.compress("zfp_adaptive", epsilon=0.05)
print(f"{field.codec_name}: {field.footprint_bytes/1e6:.2f} MB")

# 3. Persist and reload (Zarr v3 with a small, autodescriptive schema).
src.octogrid.to_zarr("sla.zarr", codec="zfp_adaptive", epsilon=0.05)
field = octogrid.open("sla.zarr")

# 4. Interpolate at arbitrary points — O(log n) per query, no kd-tree.
import numpy as np
qlat = np.array([42.5, 13.0, -55.0])
qlon = np.array([3.5, 100.0, 280.0])
out = octogrid.interpolate(field, qlat, qlon, method="barycentric")
```

---

## Footprint calculated using real data

All numbers below are for **fields downsampled to float32**, with the
empreinte measured **in-memory** (the on-disk Zarr size is smaller still
thanks to Zarr's default codec). The reference is the same data stored
as a plain `float32` matrix; everything else is normalised to that.

### Stacked savings — CMEMS L4 SLA (720 × 1440, 42 % land NaN)

| Configuration                              | RAM      | % raw  | ratio |
| ------------------------------------------ | -------- | ------ | ----- |
| `float32` regular lat/lon (reference)      | 4.15 MB  | 100 %  | ×1.0  |
| regular + `bfloat16`                       | 2.07 MB  | 50 %   | ×2.0  |
| **octahedral** + `bfloat16`                | 1.04 MB  | **25 %**  | **×4.0** |
| regular + `zfp_adaptive` ε = 5 cm          | 0.50 MB  | 12 %   | ×8.3  |
| **octahedral** + `zfp_adaptive` ε = 5 cm   | 0.34 MB  | **8.2 %** | **×12.2** |

The topology layer alone delivers **×4 reduction** versus `float32`
lat/lon. Stacking with the adaptive ZFP codec at ε = 5 cm — well below
the noise floor of altimetry at 25 km resolution — yields **×12** with a
provable error bound.

### Adaptive ZFP on full lat/lon (no resampling) — daily CMEMS DUACS L4

Each field is compressed at three epsilon levels relative to its own
σ. The codec is NaN-aware (per-row RLE mask) so land does not weigh on
the compressed payload.

| Field | σ    | ε = σ/100 | ε = σ/30    | ε = σ/10 |
| ----- | ---- | --------- | ----------- | -------- |
| SLA   | 9.6 cm  | ×4.4 (23 %)  | ×4.5 (22 %)  | ×5.9 (17 %) |
| ADT   | 74 cm  | ×4.4 (23 %)  | ×6.2 (16 %)  | **×7.4 (14 %)** |
| ugos  | 17 cm/s | ×4.4 (23 %)  | ×4.6 (22 %)  | ×6.0 (17 %) |
| vgos  | 15 cm/s | ×4.1 (24 %)  | ×4.5 (22 %)  | ×4.8 (21 %) |

### Heavy NaN fraction — MDT 2024 CNES/CLS (1440 × 2880, 35 % NaN)

NaN-aware encoding is where the per-row RLE mask shines.

| ε         | RAM      | % raw  | ratio | max err on interp |
| --------- | -------- | ------ | ----- | ----------------- |
| σ/100 (7 mm)  | 3.0 MB | 18 %   | ×5.5 | 6.4 mm |
| σ/30 (24 mm)  | 2.5 MB | 15 %   | ×6.6 | 22 mm |
| σ/10 (71 mm)  | 2.2 MB | **13 %** | **×7.6** | 67 mm |

In every adaptive run, **100 % of interpolated query points fall inside
the configured ε**.

---

## Performance

Measured on a single core of an Apple M-class CPU:

| Operation                             | Throughput |
| ------------------------------------- | ---------- |
| Resample lat/lon → octahedral (1 M pt) | 0.01 s (≈ 35× faster than `scipy.RegularGridInterpolator`) |
| Barycentric interp, `raw`             | ≈ 20 M points/s |
| Barycentric interp, `zfp_adaptive`    | ≈ 4 M points/s |
| Field initialisation (encode)         | < 1 s for a 1 M-point field at any codec |

These are all well above the practical floor (≈ 10⁶ points/s, the
throughput of reading a Zarr chunk from the page cache).

---

## On-disk format (Zarr v3)

`to_zarr` writes a self-describing group whose grid topology is
inspectable with any standard Zarr client:

```
my_field.zarr/
├── (group attrs: format_version, codec_name, grid_kind,
│                  n_points, n_rows)
├── grid_latitudes_deg   float64  (n_rows,)
├── grid_n_lon           uint32   (n_rows,)
└── codec_blob           uint8    (n_bytes,)
```

Only `codec_blob` is octogrid-specific. Everything else — coordinates,
topology, metadata — is plain Zarr.

---

## Manipulating fields as arrays

Every `CompressedField` decodes back to a flat float32 NumPy array via
`field.to_numpy()`, and the grid exposes its per-point coordinates via
`grid.latitudes()` / `grid.longitudes()`. Everything downstream is then
just NumPy / xarray — for example, a **streaming mean and variance** over
a year of daily SLA fields without ever materialising the full stack in
RAM:

```python
import numpy as np, octogrid

daily = [octogrid.open(f"sla_{day}.zarr") for day in days_2024]
n = daily[0].n_points

# Pass 1 — mean
total = np.zeros(n, dtype=np.float64)
n_obs = np.zeros(n, dtype=np.int64)
for f in daily:
    v = f.to_numpy()
    ok = np.isfinite(v)
    total[ok] += v[ok]
    n_obs[ok] += 1
mean = (total / n_obs).astype(np.float32)

# Pass 2 — variance
ssq = np.zeros(n, dtype=np.float64)
for f in daily:
    v = f.to_numpy()
    ok = np.isfinite(v)
    ssq[ok] += (v[ok] - mean[ok]) ** 2
variance = (ssq / np.maximum(n_obs - 1, 1)).astype(np.float32)

# Optional: re-wrap the result on the same grid for further use
mean_field = octogrid.compress(daily[0].grid, "raw", mean)
```

Each iteration keeps exactly one daily field decompressed in memory, so
the compression gains survive the analysis pipeline.

## Visualizing a field

Two complementary renderings, both straight NumPy + matplotlib:

```python
import matplotlib.pyplot as plt
import numpy as np, octogrid

field = ...

# 1. Native scatter — shows the reduced topology (cells thin out polewards).
lats = field.grid.latitudes()
lons = field.grid.longitudes()
plt.scatter(lons, lats, c=field.to_numpy(), s=1.0, cmap="RdBu_r")

# 2. Rendered on a regular lat/lon viewport (barycentric interpolation).
nlat, nlon = 360, 720
lat_v = np.linspace(89.75, -89.75, nlat)
lon_v = np.linspace(0.25, 359.75, nlon)
lg, og = np.meshgrid(lat_v, lon_v, indexing="ij")
img = octogrid.interpolate(field, lg.ravel(), og.ravel()).reshape(nlat, nlon)
plt.imshow(img, extent=(0, 360, -90, 90), origin="upper", cmap="RdBu_r")
```

A turnkey script generating both panels is in
[`examples/visualize.py`](examples/visualize.py).

## API summary

```python
octogrid.ReducedGrid                       # topology only
octogrid.ReducedGrid.octahedral(n_lat, base=20)
octogrid.octahedral_matching_latlon(src_lons)

octogrid.resample_from_latlon(grid, lats, lons, src_arr)
octogrid.compress(grid, codec, values, **codec_kwargs)
octogrid.interpolate(field, lat_deg, lon_deg, method="barycentric")

grid.latitudes()                           # flat per-point coords
grid.longitudes()
field.to_numpy()                           # decode every value to a numpy float32 array

octogrid.from_xarray(da)                   # high-level: lat/lon → topology field
octogrid.to_zarr(field, store)             # persist (Zarr v3)
octogrid.open(store)                       # reload

# xarray accessor (auto-registered):
da.octogrid.compress(codec="raw", **kwargs)
da.octogrid.to_zarr(store, codec="raw", **kwargs)
```

`codec` is one of `"raw"`, `"bfloat16"`, `"uint16"`, `"zfp"` (with
`zfp_rate=…`), or `"zfp_adaptive"` (with `epsilon=…` and
`max_outlier_frac=…`). Compression is **fully optional**: the default
codec is `raw`, which keeps the float32 values as-is so that the win
comes entirely from the reduced grid.

---

## Development

```sh
cmake -S . -B build
cmake --build build -j
pytest tests/                       # synthetic-data suite (≈ 1 s)
pytest -m real_data                 # add tests against a local NetCDF
pre-commit install                  # enable formatting / lint hooks
```

Pre-commit chain mirrors `pangeo-pyinterp`: `ruff`, `flake8`,
`clang-format`, `cmake-format`/`cmake-lint`, `codespell`, `mypy`.

---

## Design pointers

- **Grid:** lats[] + n_lon[] per row + cumulative offsets. Generalised
  reduced grid; the octahedral factory is one of several possible
  topologies.
- **Interpolation:** barycentric on the implicit triangulation of two
  adjacent rows (3 codec fetches, ~ 10 FLOPs per query).
- **Codecs:** uniform `Codec` interface; built-in backends are raw,
  bfloat16, uint16-per-tile, ZFP fixed-rate, and adaptive ZFP with
  per-tile rate selection + outlier patches + RLE NaN mask.
- **Bindings:** `nanobind` for the Python module.

---

## Licence

BSD-3-Clause. See [`LICENSE`](LICENSE).

The library bundles [ZFP](https://github.com/LLNL/zfp) (BSD-3) and
[nanobind](https://github.com/wjakob/nanobind) (BSD-3) via CMake
`FetchContent`.

---

## Related work

- ECMWF Atlas — reference for octahedral grids and barycentric stencils
  on reduced grids. [Newsletter 152, MIR](https://www.ecmwf.int/en/newsletter/152/computing/new-ecmwf-interpolation-package-mir).
- `pangeo-pyinterp` — generic geospatial interpolation on Cartesian and
  spherical grids; complementary scope.
- `xdggs` — emerging pangeo convention for discrete global grids; we
  expect to align with it once it stabilises.
