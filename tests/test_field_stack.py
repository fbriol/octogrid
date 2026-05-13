"""FieldStack — multi-field collections sharing a grid."""

from __future__ import annotations

import numpy as np
import octogrid
import pytest


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_fields(small_octahedral):
    """A pair of independent scalar fields on the same grid."""
    rng = np.random.default_rng(0)
    n = small_octahedral.n_points
    u = rng.standard_normal(n).astype(np.float32)
    v = rng.standard_normal(n).astype(np.float32)
    return (
        octogrid.compress(small_octahedral, "raw", u),
        octogrid.compress(small_octahedral, "raw", v),
        u,
        v,
    )


@pytest.fixture
def depth_stack(small_octahedral):
    """Synthetic salinity profile: same field at 3 depths with a vertical
    decay factor — checks that coord-mode round-trips correctly."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal(small_octahedral.n_points).astype(np.float32)
    depths = [0.0, 50.0, 500.0]
    fields = [
        octogrid.compress(
            small_octahedral,
            "raw",
            (base * np.exp(-d / 200)).astype(np.float32),
        )
        for d in depths
    ]
    return fields, depths


# ---------------------------------------------------------------------------
# construction & validation
# ---------------------------------------------------------------------------


def test_named_stack_basic(small_octahedral, two_fields):
    fu, fv, u, v = two_fields
    stack = octogrid.FieldStack(
        grid=small_octahedral, fields={"u": fu, "v": fv}
    )
    assert stack.names == ["u", "v"]
    assert stack.n_fields == 2
    assert stack.coord_name is None
    assert stack.coord_values is None
    np.testing.assert_array_equal(stack["u"].to_numpy(), u)
    np.testing.assert_array_equal(stack["v"].to_numpy(), v)


def test_stack_rejects_empty(small_octahedral):
    with pytest.raises(ValueError, match="at least one field"):
        octogrid.FieldStack(grid=small_octahedral, fields={})


def test_stack_rejects_mismatched_grid(small_octahedral, two_fields):
    fu, _, _, _ = two_fields
    other_grid = octogrid.ReducedGrid.octahedral(n_lat=32, base=20)
    wrong = octogrid.compress(
        other_grid,
        "raw",
        np.zeros(other_grid.n_points, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="differs"):
        octogrid.FieldStack(
            grid=small_octahedral, fields={"a": fu, "b": wrong}
        )


def test_stack_coord_mode_construction(small_octahedral, depth_stack):
    fields, depths = depth_stack
    stack = octogrid.FieldStack(
        grid=small_octahedral,
        fields=fields,
        coord_name="depth",
        coord_values=depths,
    )
    assert stack.coord_name == "depth"
    np.testing.assert_array_equal(stack.coord_values, np.asarray(depths))
    # Keys derive from coord_values when fields is a sequence.
    assert stack.names == ["0.0", "50.0", "500.0"]


def test_stack_coord_requires_pair(small_octahedral, two_fields):
    fu, fv, _, _ = two_fields
    with pytest.raises(ValueError, match="together"):
        octogrid.FieldStack(
            grid=small_octahedral,
            fields={"a": fu, "b": fv},
            coord_name="depth",
        )


def test_stack_coord_values_length_mismatch(small_octahedral, depth_stack):
    fields, _ = depth_stack
    with pytest.raises(ValueError, match="length must match"):
        octogrid.FieldStack(
            grid=small_octahedral,
            fields=fields,
            coord_name="depth",
            coord_values=[0.0, 50.0],  # wrong length
        )


# ---------------------------------------------------------------------------
# access patterns
# ---------------------------------------------------------------------------


def test_stack_dict_like_access(small_octahedral, two_fields):
    fu, fv, _, _ = two_fields
    stack = octogrid.FieldStack(
        grid=small_octahedral, fields={"u": fu, "v": fv}
    )
    assert "u" in stack
    assert "missing" not in stack
    assert len(stack) == 2
    assert list(iter(stack)) == ["u", "v"]


def test_stack_sel_by_coord(small_octahedral, depth_stack):
    fields, depths = depth_stack
    stack = octogrid.FieldStack(
        grid=small_octahedral,
        fields=fields,
        coord_name="depth",
        coord_values=depths,
    )
    f = stack.sel(depth=50.0)
    np.testing.assert_array_equal(f.to_numpy(), fields[1].to_numpy())


def test_stack_sel_unknown_coord_raises(small_octahedral, depth_stack):
    fields, depths = depth_stack
    stack = octogrid.FieldStack(
        grid=small_octahedral,
        fields=fields,
        coord_name="depth",
        coord_values=depths,
    )
    with pytest.raises(KeyError, match="depth=999"):
        stack.sel(depth=999.0)


def test_stack_sel_requires_coord_axis(small_octahedral, two_fields):
    fu, fv, _, _ = two_fields
    stack = octogrid.FieldStack(
        grid=small_octahedral, fields={"u": fu, "v": fv}
    )
    with pytest.raises(TypeError, match="coord axis"):
        stack.sel(u=0.0)


def test_stack_to_numpy_shape(small_octahedral, two_fields):
    fu, fv, u, v = two_fields
    stack = octogrid.FieldStack(
        grid=small_octahedral, fields={"u": fu, "v": fv}
    )
    arr = stack.to_numpy()
    assert arr.shape == (2, small_octahedral.n_points)
    np.testing.assert_array_equal(arr[0], u)
    np.testing.assert_array_equal(arr[1], v)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def test_stack_zarr_round_trip_named(tmp_path, small_octahedral, two_fields):
    fu, fv, u, v = two_fields
    stack = octogrid.FieldStack(
        grid=small_octahedral, fields={"u": fu, "v": fv}
    )
    path = str(tmp_path / "stack_named.zarr")
    stack.to_zarr(path)
    reloaded = octogrid.open_stack(path)
    assert reloaded.names == ["u", "v"]
    assert reloaded.coord_name is None
    np.testing.assert_array_equal(reloaded["u"].to_numpy(), u)
    np.testing.assert_array_equal(reloaded["v"].to_numpy(), v)


def test_stack_zarr_round_trip_coord(tmp_path, small_octahedral, depth_stack):
    fields, depths = depth_stack
    stack = octogrid.FieldStack(
        grid=small_octahedral,
        fields=fields,
        coord_name="depth",
        coord_values=depths,
    )
    path = str(tmp_path / "stack_depth.zarr")
    stack.to_zarr(path)
    reloaded = octogrid.open_stack(path)
    assert reloaded.coord_name == "depth"
    np.testing.assert_array_equal(reloaded.coord_values, np.asarray(depths))
    # Per-field values bit-identical (raw codec).
    for original, name in zip(fields, reloaded.names, strict=True):
        np.testing.assert_array_equal(
            reloaded[name].to_numpy(), original.to_numpy()
        )


def test_stack_zarr_round_trip_with_compression(tmp_path, small_octahedral):
    """Cross-codec stack: each field can use a different codec."""
    rng = np.random.default_rng(2)
    n = small_octahedral.n_points
    fields = {
        "raw": octogrid.compress(
            small_octahedral,
            "raw",
            rng.standard_normal(n).astype(np.float32),
        ),
        "bf16": octogrid.compress(
            small_octahedral,
            "bfloat16",
            rng.standard_normal(n).astype(np.float32),
        ),
        "zfp_ad": octogrid.compress(
            small_octahedral,
            "zfp_adaptive",
            rng.standard_normal(n).astype(np.float32),
            epsilon=0.05,
            max_outlier_frac=0.01,
        ),
    }
    stack = octogrid.FieldStack(grid=small_octahedral, fields=fields)
    path = str(tmp_path / "mixed.zarr")
    stack.to_zarr(path)
    reloaded = octogrid.open_stack(path)
    for name, field in fields.items():
        np.testing.assert_array_equal(
            reloaded[name].to_numpy(), field.to_numpy()
        )


# ---------------------------------------------------------------------------
# xarray interop
# ---------------------------------------------------------------------------


def test_stack_to_xarray_named(small_octahedral, two_fields):
    pytest.importorskip("xarray")
    fu, fv, u, v = two_fields
    stack = octogrid.FieldStack(
        grid=small_octahedral, fields={"u": fu, "v": fv}
    )
    ds = stack.to_xarray()
    assert set(ds.data_vars) == {"u", "v"}
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    np.testing.assert_array_equal(ds["u"].values, u)
    np.testing.assert_array_equal(ds["v"].values, v)


def test_stack_to_xarray_coord_axis(small_octahedral, depth_stack):
    pytest.importorskip("xarray")
    fields, depths = depth_stack
    stack = octogrid.FieldStack(
        grid=small_octahedral,
        fields=fields,
        coord_name="depth",
        coord_values=depths,
    )
    da = stack.to_xarray()
    assert da.dims == ("depth", "point")
    np.testing.assert_array_equal(da.coords["depth"].values, depths)
    np.testing.assert_array_equal(da.shape, (3, small_octahedral.n_points))
