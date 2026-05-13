"""``FieldStack`` — a collection of ``CompressedField`` sharing one grid.

A `FieldStack` is the natural place to put multi-variable / multi-channel
data (vector components, depth profiles, ensemble members, …). The core
`CompressedField` stays scalar-per-point; the stack is a thin Python
wrapper that:

* validates that every member shares the same grid topology,
* offers ergonomic access by name or by coordinate value,
* persists everything into a single Zarr v3 group so the on-disk story
  is identical to the single-field case but with extra subgroups,
* round-trips to xarray (``Dataset`` for named mode, ``DataArray`` with
  a leading axis for coord mode).

Two construction modes:

* **Named mode** — keys are arbitrary variable names::

      FieldStack(grid=g, fields={"u": fu, "v": fv, "T": fT})

* **Coord mode** — keys label positions along a named axis::

      FieldStack(
          grid=g,
          fields=[s0, s10, s50],
          coord_name="depth",
          coord_values=[0.0, 10.0, 50.0],
      )

Both modes co-exist in the same class; coord mode just attaches an
optional named axis. The on-disk format is the same.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from ._octogrid import (
    CompressedField,
    ReducedGrid,
    _field_from_blob,
    _serialize_codec,
)


if TYPE_CHECKING:
    import xarray as xr
    import zarr.storage
    from numpy.typing import ArrayLike, NDArray

    Store = Union[str, "zarr.storage.Store"]

#: On-disk format version for FieldStack groups. Bump on schema changes.
STACK_FORMAT_VERSION = 1


def _grid_signature(g: ReducedGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays that uniquely identify a grid topology (cheap)."""
    return (
        np.asarray([g.lat_deg(r) for r in range(g.n_rows)], dtype=np.float64),
        np.asarray([g.n_lon(r) for r in range(g.n_rows)], dtype=np.uint32),
    )


class FieldStack:
    """A grid-aligned collection of compressed fields."""

    def __init__(
        self,
        *,
        grid: ReducedGrid,
        fields: Mapping[str, CompressedField] | Sequence[CompressedField],
        coord_name: str | None = None,
        coord_values: ArrayLike | None = None,
    ) -> None:
        if not fields:
            raise ValueError("FieldStack must contain at least one field")

        # Normalise `fields` into an ordered dict[str, CompressedField].
        if isinstance(fields, Mapping):
            ordered = dict(fields)
        else:
            # Sequence path: derive keys from coord_values when available,
            # otherwise from positional indices.
            if coord_values is not None:
                cv_pre = np.asarray(coord_values)
                keys = [str(v) for v in cv_pre.tolist()]
            else:
                keys = [str(i) for i in range(len(fields))]
            if len(keys) != len(fields):
                raise ValueError(
                    "coord_values length must match the number of fields"
                )
            ordered = dict(zip(keys, fields, strict=True))

        # Validate that every field sits on the same grid.
        ref_lats, ref_n_lon = _grid_signature(grid)
        for name, field in ordered.items():
            f_lats, f_n_lon = _grid_signature(field.grid)
            if (
                field.grid.n_rows != grid.n_rows
                or field.grid.n_points != grid.n_points
                or not np.array_equal(f_lats, ref_lats)
                or not np.array_equal(f_n_lon, ref_n_lon)
            ):
                raise ValueError(
                    f"field {name!r} has a grid that differs from the stack"
                )

        # Validate the optional coord axis.
        cv: NDArray | None
        if coord_name is None and coord_values is None:
            cv = None
        else:
            if coord_name is None or coord_values is None:
                raise ValueError(
                    "coord_name and coord_values must be provided together"
                )
            cv = np.asarray(coord_values)
            if cv.ndim != 1 or cv.size != len(ordered):
                raise ValueError(
                    "coord_values must be 1-D with one entry per field"
                )

        self._grid = grid
        self._fields = ordered
        self._coord_name = coord_name
        self._coord_values = cv

    # ---- introspection ----------------------------------------------------

    @property
    def grid(self) -> ReducedGrid:
        return self._grid

    @property
    def names(self) -> list[str]:
        return list(self._fields)

    @property
    def coord_name(self) -> str | None:
        return self._coord_name

    @property
    def coord_values(self) -> NDArray | None:
        return (
            None if self._coord_values is None else self._coord_values.copy()
        )

    @property
    def n_fields(self) -> int:
        return len(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __contains__(self, name: object) -> bool:
        return name in self._fields

    def __getitem__(self, name: str) -> CompressedField:
        return self._fields[name]

    def __repr__(self) -> str:
        cv = self._coord_values
        if self._coord_name is None or cv is None:
            head = f"FieldStack({self.n_fields} fields: {self.names})"
        else:
            head = (
                f"FieldStack({self.n_fields} fields along "
                f"{self._coord_name}={cv.tolist()})"
            )
        return f"{head} on {self._grid.n_points:,} points"

    # ---- coord-mode access ----------------------------------------------

    def sel(self, **kwargs: float) -> CompressedField:
        """Pick a field by exact coordinate value.

        Only valid in coord mode (when ``coord_name`` was provided at
        construction).
        """
        if self._coord_name is None or self._coord_values is None:
            raise TypeError("sel() requires a coord axis on the stack")
        if list(kwargs) != [self._coord_name]:
            raise ValueError(
                f"sel() takes exactly one keyword: {self._coord_name!r}"
            )
        wanted = kwargs[self._coord_name]
        matches = np.flatnonzero(self._coord_values == wanted)
        if matches.size == 0:
            raise KeyError(
                f"{self._coord_name}={wanted!r} not in coord_values"
            )
        name = list(self._fields)[int(matches[0])]
        return self._fields[name]

    # ---- bulk decode ----------------------------------------------------

    def to_numpy(self) -> NDArray[np.float32]:
        """Decode every field into a (n_fields, n_points) float32 array."""
        return np.stack([f.to_numpy() for f in self._fields.values()], axis=0)

    # ---- xarray interop -------------------------------------------------

    def to_xarray(self) -> xr.DataArray | xr.Dataset:
        """Convert to xarray.

        Named mode → ``xarray.Dataset`` with one variable per field.
        Coord mode → ``xarray.DataArray`` with shape ``(n_coord, n_points)``.

        In both cases the per-point lat/lon coordinates are attached.
        Decoding is eager: large stacks will materialise their full
        content in RAM.
        """
        import xarray as xr  # noqa: PLC0415

        lats = self._grid.latitudes()
        lons = self._grid.longitudes()
        point_coords = {
            "latitude": ("point", lats),
            "longitude": ("point", lons),
        }
        if self._coord_name is None:
            return xr.Dataset(
                {
                    name: xr.DataArray(
                        field.to_numpy(),
                        dims=["point"],
                        coords=point_coords,
                    )
                    for name, field in self._fields.items()
                }
            )
        return xr.DataArray(
            self.to_numpy(),
            dims=[self._coord_name, "point"],
            coords={
                self._coord_name: self._coord_values,
                **point_coords,
            },
        )

    # ---- persistence ----------------------------------------------------

    def to_zarr(self, store: Store, *, mode: str = "w") -> None:
        """Persist the stack to a Zarr v3 store.

        Layout::

            <store>/
            ├── (attrs: format_version, kind="stack", field_names,
            │           n_points, n_rows, [coord_name, coord_values])
            ├── grid_latitudes_deg
            ├── grid_n_lon
            └── <field_name>/         (one subgroup per field)
                ├── (attrs: codec_name)
                └── codec_blob
        """
        import zarr  # noqa: PLC0415

        lats, n_lon = _grid_signature(self._grid)
        root = zarr.open_group(store, mode=mode, zarr_format=3)
        attrs: dict[str, Any] = {
            "format_version": STACK_FORMAT_VERSION,
            "kind": "stack",
            "field_names": list(self._fields),
            "n_points": int(self._grid.n_points),
            "n_rows": int(self._grid.n_rows),
        }
        cv = self._coord_values
        if self._coord_name is not None and cv is not None:
            attrs["coord_name"] = self._coord_name
            attrs["coord_values"] = cv.tolist()
        root.attrs.update(attrs)

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

        for name, field in self._fields.items():
            sub = root.create_group(name)
            sub.attrs["codec_name"] = field.codec_name
            blob = _serialize_codec(field)
            sub.create_array(
                "codec_blob",
                shape=blob.shape,
                dtype=blob.dtype,
            )[:] = blob


def open_stack(store: Store) -> FieldStack:
    """Reload a :class:`FieldStack` written by :meth:`FieldStack.to_zarr`."""
    import zarr  # noqa: PLC0415

    root = zarr.open_group(store, mode="r")
    attrs = dict(root.attrs)
    version = attrs.get("format_version")
    if version != STACK_FORMAT_VERSION:
        raise ValueError(f"unsupported FieldStack format version: {version!r}")
    if attrs.get("kind") != "stack":
        raise ValueError("zarr group is not a FieldStack")
    lats = np.asarray(root["grid_latitudes_deg"][:], dtype=np.float64)
    n_lon = np.asarray(root["grid_n_lon"][:], dtype=np.uint32)
    grid = ReducedGrid(
        latitudes_deg=lats.tolist(),
        n_lon=n_lon.tolist(),
    )

    names = list(attrs["field_names"])
    fields: dict[str, CompressedField] = {}
    for name in names:
        sub = root[name]
        codec_name = sub.attrs["codec_name"]
        # AdaptiveZfp reports a slightly different runtime name when it has
        # a NaN mask; the serialized form is identical, route both back.
        if codec_name.startswith("zfp-adaptive"):
            codec_name = "zfp-adaptive"
        blob = np.asarray(sub["codec_blob"][:], dtype=np.uint8)
        fields[name] = _field_from_blob(grid, codec_name, blob)

    return FieldStack(
        grid=grid,
        fields=fields,
        coord_name=attrs.get("coord_name"),
        coord_values=attrs.get("coord_values"),
    )
