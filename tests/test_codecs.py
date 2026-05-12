"""Codec contract tests: encode → decode round-trip and footprint behaviour."""

from __future__ import annotations

import numpy as np
import octogrid
import pytest


@pytest.fixture
def values_5248():
    """A smooth analytic signal sized to the small octahedral grid."""
    rng = np.random.default_rng(1)
    return rng.uniform(-1.0, 1.0, 5248).astype(np.float32)


def _decode_all(field, n):
    """Helper: brute-force decode every linear index via barycentric trick.

    The C++ side does not expose a flat decode, so we rebuild the array
    by interpolating exactly at the grid nodes (weights collapse to 1, 0).
    """
    grid = field.grid
    lats, lons = [], []
    for r in range(grid.n_rows):
        n_lon = grid.n_lon(r)
        for i in range(n_lon):
            lats.append(grid.lat_deg(r))
            lons.append(i * 360.0 / n_lon)
    out = octogrid.interpolate(
        field, np.asarray(lats), np.asarray(lons), method="nearest"
    )
    assert out.size == n
    return out


def test_raw_codec_is_lossless(small_octahedral, values_5248):
    field = octogrid.compress(small_octahedral, "raw", values_5248)
    assert field.codec_name == "raw"
    decoded = _decode_all(field, values_5248.size)
    np.testing.assert_array_equal(decoded, values_5248)


def test_bfloat16_bounded_error(small_octahedral, values_5248):
    field = octogrid.compress(small_octahedral, "bfloat16", values_5248)
    decoded = _decode_all(field, values_5248.size)
    # bfloat16 has 8 bits of mantissa; relative error <= 2^-8.
    rel = np.abs(decoded - values_5248) / (np.abs(values_5248) + 1e-6)
    assert rel.max() < 1e-2


def test_uint16_quasi_lossless(small_octahedral, values_5248):
    field = octogrid.compress(small_octahedral, "uint16", values_5248)
    decoded = _decode_all(field, values_5248.size)
    # 16-bit uniform quantization on a [-1, 1] tile → ε ≤ 2/65535.
    assert np.abs(decoded - values_5248).max() < 5e-5


@pytest.mark.parametrize("rate", [4, 8, 16])
def test_zfp_decreasing_footprint(small_octahedral, values_5248, rate):
    field = octogrid.compress(
        small_octahedral, "zfp", values_5248, zfp_rate=rate
    )
    raw_bytes = values_5248.nbytes
    # ZFP footprint in fixed-rate mode is rate bits per value plus small
    # overhead. Allow generous slack.
    assert field.footprint_bytes <= raw_bytes * (rate / 32) * 1.1 + 4096


@pytest.mark.parametrize("epsilon", [1e-3, 1e-2, 1e-1])
def test_zfp_adaptive_respects_epsilon(small_octahedral, values_5248, epsilon):
    field = octogrid.compress(
        small_octahedral,
        "zfp_adaptive",
        values_5248,
        epsilon=epsilon,
        max_outlier_frac=0.01,
    )
    decoded = _decode_all(field, values_5248.size)
    # Outliers may exceed epsilon for at most 1 % of points per tile, but
    # because they are stored at float32 precision the *patched* values
    # round-trip exactly. So the global max error is bounded by epsilon.
    assert np.abs(decoded - values_5248).max() <= epsilon


def test_unknown_codec_raises(small_octahedral, values_5248):
    with pytest.raises(ValueError, match="unknown codec"):
        octogrid.compress(small_octahedral, "no-such-codec", values_5248)


def test_compress_rejects_wrong_length(small_octahedral):
    bad = np.zeros(small_octahedral.n_points + 1, dtype=np.float32)
    with pytest.raises(ValueError):
        octogrid.compress(small_octahedral, "raw", bad)
