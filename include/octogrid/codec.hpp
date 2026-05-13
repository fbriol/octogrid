#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace octogrid {

// Codec contract: a compression backend with O(1)-ish random access by linear
// index. Backends MUST support point-decode (single value) and SHOULD
// support gather-decode (small index set, e.g. 3 vertices of a triangle).
//
// Design notes:
// - encode() is called once at field construction; it returns the byte buffer
//   that the field will hold for its lifetime.
// - decode_one() is on the hot path. Keep it inlinable / branchless where
//   possible. The barycentric interpolator calls it 3 times per query.
// - The codec is stateless apart from per-tile metadata (which it stores in
//   the byte buffer itself or in side tables it allocates and owns).
class Codec {
 public:
  virtual ~Codec() = default;

  // Encode `values.size()` float32 samples into an internal buffer.
  // Returns the buffer (caller-owned by Codec for its lifetime).
  // Must be called exactly once per codec instance.
  virtual void encode(const float *values, std::size_t n) = 0;

  // Decode a single value at linear index `idx`.
  [[nodiscard]] virtual auto decode_one(std::size_t idx) const -> float = 0;

  // Decode `k` values at the given indices into `out`. Default impl loops.
  virtual void decode_gather(const std::size_t *indices, std::size_t k,
                             float *out) const {
    for (std::size_t i = 0; i < k; ++i) out[i] = decode_one(indices[i]);
  }

  // Decode all values into a flat float32 buffer of length `n`. Default
  // implementation loops decode_one; concrete codecs override when a
  // bulk path is faster (e.g. a memcpy for the raw codec).
  virtual void decode_all(float *out, std::size_t n) const {
    for (std::size_t i = 0; i < n; ++i) out[i] = decode_one(i);
  }

  // RAM footprint of the compressed representation (bytes).
  [[nodiscard]] virtual auto footprint_bytes() const -> std::size_t = 0;

  // Codec name (for debugging / logging).
  [[nodiscard]] virtual auto name() const -> const char * = 0;

  // Serialize the codec's internal state to a contiguous byte buffer.
  // Layout is codec-specific and not stable across major versions; pair
  // each serialized blob with the codec name and a format version stored
  // by the caller.
  [[nodiscard]] virtual auto serialize() const -> std::vector<std::uint8_t> = 0;
};

// Reconstruct a codec from a name + a previously-serialized blob.
// `grid` is needed by codecs whose state depends on row offsets.
auto deserialize_codec(const class ReducedGrid &grid, const std::string &name,
                       const std::uint8_t *data, std::size_t size)
    -> std::unique_ptr<Codec>;

// ---- Concrete codecs ------------------------------------------------------

// C0: Raw float32. No compression — used as the default "topology only" mode
// where the win comes from the reduced grid alone. Decode is a single load.
class RawFloat32Codec : public Codec {
 public:
  void encode(const float *values, std::size_t n) override;
  [[nodiscard]] auto decode_one(std::size_t idx) const -> float override {
    return data_[idx];
  }
  void decode_all(float *out, std::size_t n) const override;
  [[nodiscard]] auto footprint_bytes() const -> std::size_t override {
    return data_.size() * sizeof(float);
  }
  [[nodiscard]] auto name() const -> const char * override { return "raw"; }
  [[nodiscard]] auto serialize() const -> std::vector<std::uint8_t> override;
  static auto deserialize(const std::uint8_t *data, std::size_t size)
      -> std::unique_ptr<RawFloat32Codec>;

 private:
  std::vector<float> data_;
};

// C1: bfloat16. Plain uint16 storage of the high 16 bits of float32. Ratio
// fixed at ×2, decode = 1 shift. The "performance maximum" baseline.
class Bfloat16Codec : public Codec {
 public:
  void encode(const float *values, std::size_t n) override;
  [[nodiscard]] auto decode_one(std::size_t idx) const -> float override;
  [[nodiscard]] auto footprint_bytes() const -> std::size_t override {
    return data_.size() * 2;
  }
  [[nodiscard]] auto name() const -> const char * override {
    return "bfloat16";
  }
  [[nodiscard]] auto serialize() const -> std::vector<std::uint8_t> override;
  static auto deserialize(const std::uint8_t *data, std::size_t size)
      -> std::unique_ptr<Bfloat16Codec>;

 private:
  std::vector<std::uint16_t> data_;
};

// C2: uint16 quantization per tile. Tile size configurable (default = one
// full row of the grid, see field.cpp for wiring). Per-tile (min,max) stored
// as float32; values stored as uint16. Effective ratio ~×2 (uint16) — uint8
// variant left to a future codec. The dependency-free fallback.
class Uint16Codec : public Codec {
 public:
  // tile_size = number of consecutive samples per tile. The encoder is
  // told the tile boundaries externally (via tile_offsets) for the case
  // where tiles are row-aligned in a reduced grid.
  explicit Uint16Codec(std::vector<std::size_t> tile_offsets);

  void encode(const float *values, std::size_t n) override;
  [[nodiscard]] auto decode_one(std::size_t idx) const -> float override;
  [[nodiscard]] auto footprint_bytes() const -> std::size_t override;
  [[nodiscard]] auto name() const -> const char * override {
    return "uint16-tiled";
  }
  [[nodiscard]] auto serialize() const -> std::vector<std::uint8_t> override;
  static auto deserialize(std::vector<std::size_t> tile_offsets,
                          const std::uint8_t *data, std::size_t size)
      -> std::unique_ptr<Uint16Codec>;

 private:
  std::vector<std::size_t> tile_offsets_;  // size n_tiles + 1
  std::vector<float> tile_min_;
  std::vector<float> tile_scale_;  // (max - min) / 65535, or 0
  std::vector<std::uint16_t> data_;
};

// Factory helper: uint16 codec with one tile per grid row.
auto make_uint16_row_tiled(const class ReducedGrid &grid)
    -> std::unique_ptr<Codec>;

// Factory helper: bfloat16 codec (no tiling needed).
auto make_bfloat16() -> std::unique_ptr<Codec>;

// Factory helper: raw float32 codec (no compression).
auto make_raw() -> std::unique_ptr<Codec>;

#ifdef OCTOGRID_WITH_ZFP
// C3: ZFP fixed-rate, 1D blocks of 4 values. Random access by block.
// `rate` = bits per value (e.g. 8 → ratio ×4, 4 → ratio ×8, 16 → ratio ×2).
// Must be a positive integer; ZFP uses rate*4 bits per 4-value block.
auto make_zfp_fixed_rate(unsigned rate) -> std::unique_ptr<Codec>;

// C3+: Adaptive ZFP. Per-tile rate selection driven by an error budget,
// with an outlier patch layer for values that the base codec can't represent
// within `epsilon`. Compression-first behaviour: picks the smallest rate
// from a candidate set such that the fraction of outliers in the tile stays
// below `max_outlier_fraction_per_tile`. Outliers are stored at float32
// precision in a sorted (index, value) side table.
//
// Trade-off vs plain fixed-rate ZFP:
//   - Better ratio on heterogeneous fields (smooth tiles get 2–4 bits/value;
//     rough tiles get 8–12).
//   - Bounds the worst-case error to `epsilon` everywhere by construction.
//   - Decode cost: +1 binary search per query (in outlier table) and +1
//     load of per-tile rate.
auto make_zfp_adaptive(const class ReducedGrid &grid, double epsilon,
                       double max_outlier_fraction_per_tile = 0.01)
    -> std::unique_ptr<Codec>;
#endif

}  // namespace octogrid
