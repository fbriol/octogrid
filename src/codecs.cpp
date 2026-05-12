#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "octogrid/codec.hpp"
#include "octogrid/grid.hpp"

#ifdef OCTOGRID_WITH_ZFP
#include <zfp.h>
#endif

namespace octogrid {

// ---- Raw float32 ----------------------------------------------------------

void RawFloat32Codec::encode(const float *values, std::size_t n) {
  data_.assign(values, values + n);
}

std::vector<std::uint8_t> RawFloat32Codec::serialize() const {
  std::vector<std::uint8_t> out(data_.size() * sizeof(float));
  std::memcpy(out.data(), data_.data(), out.size());
  return out;
}

std::unique_ptr<RawFloat32Codec> RawFloat32Codec::deserialize(
    const std::uint8_t *data, std::size_t size) {
  if (size % sizeof(float) != 0)
    throw std::invalid_argument("raw codec: blob size not a multiple of 4");
  auto codec = std::make_unique<RawFloat32Codec>();
  codec->data_.resize(size / sizeof(float));
  std::memcpy(codec->data_.data(), data, size);
  return codec;
}

std::unique_ptr<Codec> make_raw() {
  return std::make_unique<RawFloat32Codec>();
}

// ---- Bfloat16 -------------------------------------------------------------

static inline std::uint16_t f32_to_bf16(float v) {
  std::uint32_t x;
  std::memcpy(&x, &v, sizeof(x));
  // Round-to-nearest-even with tie-breaking via bit injection.
  const std::uint32_t rounding_bias = 0x7FFF + ((x >> 16) & 1);
  return static_cast<std::uint16_t>((x + rounding_bias) >> 16);
}

static inline float bf16_to_f32(std::uint16_t v) {
  std::uint32_t x = static_cast<std::uint32_t>(v) << 16;
  float r;
  std::memcpy(&r, &x, sizeof(r));
  return r;
}

void Bfloat16Codec::encode(const float *values, std::size_t n) {
  data_.resize(n);
  for (std::size_t i = 0; i < n; ++i) data_[i] = f32_to_bf16(values[i]);
}

float Bfloat16Codec::decode_one(std::size_t idx) const {
  return bf16_to_f32(data_[idx]);
}

std::vector<std::uint8_t> Bfloat16Codec::serialize() const {
  std::vector<std::uint8_t> out(data_.size() * sizeof(std::uint16_t));
  std::memcpy(out.data(), data_.data(), out.size());
  return out;
}

std::unique_ptr<Bfloat16Codec> Bfloat16Codec::deserialize(
    const std::uint8_t *data, std::size_t size) {
  if (size % sizeof(std::uint16_t) != 0)
    throw std::invalid_argument("bfloat16 codec: bad blob size");
  auto codec = std::make_unique<Bfloat16Codec>();
  codec->data_.resize(size / sizeof(std::uint16_t));
  std::memcpy(codec->data_.data(), data, size);
  return codec;
}

// ---- Uint16 per-tile ------------------------------------------------------

Uint16Codec::Uint16Codec(std::vector<std::size_t> tile_offsets)
    : tile_offsets_(std::move(tile_offsets)) {
  if (tile_offsets_.size() < 2)
    throw std::invalid_argument("need at least one tile");
}

void Uint16Codec::encode(const float *values, std::size_t n) {
  if (tile_offsets_.back() != n)
    throw std::invalid_argument("tile_offsets last entry != n samples");
  const std::size_t n_tiles = tile_offsets_.size() - 1;
  tile_min_.resize(n_tiles);
  tile_scale_.resize(n_tiles);
  data_.resize(n);
  for (std::size_t t = 0; t < n_tiles; ++t) {
    const std::size_t lo = tile_offsets_[t];
    const std::size_t hi = tile_offsets_[t + 1];
    float vmin = values[lo], vmax = values[lo];
    for (std::size_t i = lo + 1; i < hi; ++i) {
      vmin = std::min(vmin, values[i]);
      vmax = std::max(vmax, values[i]);
    }
    tile_min_[t] = vmin;
    const float range = vmax - vmin;
    // Use 0 scale for constant tiles; decode short-circuits.
    tile_scale_[t] = (range > 0.0f) ? range / 65535.0f : 0.0f;
    if (range > 0.0f) {
      const float inv = 65535.0f / range;
      for (std::size_t i = lo; i < hi; ++i) {
        const float q = (values[i] - vmin) * inv;
        // round-to-nearest, clamp defensively against fp error.
        const float qr = std::round(q);
        const std::uint32_t qi = static_cast<std::uint32_t>(
            qr < 0.0f ? 0.0f : (qr > 65535.0f ? 65535.0f : qr));
        data_[i] = static_cast<std::uint16_t>(qi);
      }
    } else {
      std::fill(data_.begin() + lo, data_.begin() + hi,
                static_cast<std::uint16_t>(0));
    }
  }
}

float Uint16Codec::decode_one(std::size_t idx) const {
  // Locate tile via upper_bound on offsets. O(log n_tiles).
  auto it = std::upper_bound(tile_offsets_.begin(), tile_offsets_.end(), idx);
  const std::size_t t =
      static_cast<std::size_t>(it - tile_offsets_.begin()) - 1;
  return tile_min_[t] + tile_scale_[t] * static_cast<float>(data_[idx]);
}

std::size_t Uint16Codec::footprint_bytes() const {
  return data_.size() * sizeof(std::uint16_t) +
         tile_min_.size() * sizeof(float) + tile_scale_.size() * sizeof(float) +
         tile_offsets_.size() * sizeof(std::size_t);
}

// Serialized layout for Uint16Codec (versioned implicitly by codec name +
// outer attribute "format_version"):
//   [n_tiles : uint64]
//   [tile_min   : float32 × n_tiles]
//   [tile_scale : float32 × n_tiles]
//   [data       : uint16  × n_total]   (n_total = tile_offsets_.back())
// Tile boundaries are reconstructed from the grid on load.
std::vector<std::uint8_t> Uint16Codec::serialize() const {
  const std::size_t n_tiles = tile_min_.size();
  const std::size_t header = sizeof(std::uint64_t);
  const std::size_t mins = n_tiles * sizeof(float);
  const std::size_t scales = n_tiles * sizeof(float);
  const std::size_t data = data_.size() * sizeof(std::uint16_t);
  std::vector<std::uint8_t> out(header + mins + scales + data);
  std::uint8_t *p = out.data();
  const std::uint64_t n_tiles_u = n_tiles;
  std::memcpy(p, &n_tiles_u, header);
  p += header;
  std::memcpy(p, tile_min_.data(), mins);
  p += mins;
  std::memcpy(p, tile_scale_.data(), scales);
  p += scales;
  std::memcpy(p, data_.data(), data);
  return out;
}

std::unique_ptr<Uint16Codec> Uint16Codec::deserialize(
    std::vector<std::size_t> tile_offsets, const std::uint8_t *data,
    std::size_t size) {
  auto codec = std::make_unique<Uint16Codec>(std::move(tile_offsets));
  const std::size_t n_tiles = codec->tile_offsets_.size() - 1;
  const std::size_t header = sizeof(std::uint64_t);
  if (size < header)
    throw std::invalid_argument("uint16 codec: blob too small");
  std::uint64_t n_tiles_stored;
  std::memcpy(&n_tiles_stored, data, header);
  if (n_tiles_stored != n_tiles)
    throw std::invalid_argument("uint16 codec: tile count mismatch with grid");
  const std::size_t mins = n_tiles * sizeof(float);
  const std::size_t scales = n_tiles * sizeof(float);
  const std::size_t n_total = codec->tile_offsets_.back();
  const std::size_t data_bytes = n_total * sizeof(std::uint16_t);
  if (size != header + mins + scales + data_bytes)
    throw std::invalid_argument("uint16 codec: blob size mismatch");
  codec->tile_min_.resize(n_tiles);
  codec->tile_scale_.resize(n_tiles);
  codec->data_.resize(n_total);
  const std::uint8_t *p = data + header;
  std::memcpy(codec->tile_min_.data(), p, mins);
  p += mins;
  std::memcpy(codec->tile_scale_.data(), p, scales);
  p += scales;
  std::memcpy(codec->data_.data(), p, data_bytes);
  return codec;
}

// ---- Factories ------------------------------------------------------------

std::unique_ptr<Codec> make_bfloat16() {
  return std::make_unique<Bfloat16Codec>();
}

#ifdef OCTOGRID_WITH_ZFP

// ZFP fixed-rate, 1D. Each 4-value block uses exactly rate*4 bits, so we can
// seek to any block in O(1) via the bitstream and decode it independently.
// The 1D layout means blocks straddle row boundaries — acceptable for the
// prototype; per-row tiling would buy slightly better compression on
// piecewise-smooth fields at the cost of per-row stream headers.
class ZfpCodec : public Codec {
 public:
  explicit ZfpCodec(unsigned rate)
      : rate_(rate), zfp_(nullptr), stream_(nullptr) {
    if (rate == 0) throw std::invalid_argument("zfp rate must be > 0");
  }
  ~ZfpCodec() override {
    if (stream_) stream_close(stream_);
    if (zfp_) zfp_stream_close(zfp_);
  }

  void encode(const float *values, std::size_t n) override {
    n_ = n;
    // Round n up to a multiple of 4 by padding with the last value; ZFP's
    // header-based formats handle partial blocks but the low-level random
    // access we use needs full blocks.
    n_padded_ = (n + 3) & ~static_cast<std::size_t>(3);
    std::vector<float> padded(n_padded_);
    std::copy(values, values + n, padded.begin());
    for (std::size_t i = n; i < n_padded_; ++i) padded[i] = values[n - 1];

    zfp_field *field = zfp_field_1d(padded.data(), zfp_type_float,
                                    static_cast<uint>(n_padded_));
    zfp_ = zfp_stream_open(nullptr);
    // align=zfp_false → blocks bit-packed, no 64-bit word padding (which
    // would clamp the minimum rate to 16 bits/value in 1D). Random access
    // still works via bit-level stream_rseek.
    const double actual_rate = zfp_stream_set_rate(
        zfp_, static_cast<double>(rate_), zfp_type_float, 1, zfp_false);
    bits_per_block_ = static_cast<std::size_t>(actual_rate * 4.0 + 0.5);

    const std::size_t bufsize = zfp_stream_maximum_size(zfp_, field);
    buffer_.resize(bufsize);
    stream_ = stream_open(buffer_.data(), bufsize);
    zfp_stream_set_bit_stream(zfp_, stream_);
    zfp_stream_rewind(zfp_);
    const std::size_t zfpsize = zfp_compress(zfp_, field);
    if (zfpsize == 0) {
      zfp_field_free(field);
      throw std::runtime_error("zfp_compress failed");
    }
    buffer_.resize(zfpsize);
    // Stream may have been reallocated logically; reopen on the trimmed buf.
    stream_close(stream_);
    stream_ = stream_open(buffer_.data(), zfpsize);
    zfp_stream_set_bit_stream(zfp_, stream_);
    zfp_field_free(field);
  }

  float decode_one(std::size_t idx) const override {
    const std::size_t block = idx >> 2;
    const std::size_t lane = idx & 3;
    stream_rseek(stream_, block * bits_per_block_);
    float buf[4];
    zfp_decode_block_float_1(zfp_, buf);
    return buf[lane];
  }

  void decode_gather(const std::size_t *indices, std::size_t k,
                     float *out) const override {
    // When two indices fall in the same block, decode it once.
    float blk[4];
    std::size_t last_block = static_cast<std::size_t>(-1);
    for (std::size_t i = 0; i < k; ++i) {
      const std::size_t block = indices[i] >> 2;
      const std::size_t lane = indices[i] & 3;
      if (block != last_block) {
        stream_rseek(stream_, block * bits_per_block_);
        zfp_decode_block_float_1(zfp_, blk);
        last_block = block;
      }
      out[i] = blk[lane];
    }
  }

  std::size_t footprint_bytes() const override {
    return buffer_.size() + sizeof(*this);
  }
  const char *name() const override { return "zfp-fixed-rate"; }

  std::vector<std::uint8_t> serialize() const override {
    // Layout: [u32 rate][u64 n][u64 n_padded][u64 bits_per_block]
    //         [u64 buf_size][u8 × buf_size]
    const std::size_t header =
        sizeof(std::uint32_t) + 4 * sizeof(std::uint64_t);
    std::vector<std::uint8_t> out(header + buffer_.size());
    std::uint8_t *p = out.data();
    const std::uint32_t rate_u = rate_;
    std::memcpy(p, &rate_u, sizeof(rate_u));
    p += sizeof(rate_u);
    const std::uint64_t fields[4] = {n_, n_padded_, bits_per_block_,
                                     buffer_.size()};
    std::memcpy(p, fields, sizeof(fields));
    p += sizeof(fields);
    std::memcpy(p, buffer_.data(), buffer_.size());
    return out;
  }

  static std::unique_ptr<ZfpCodec> deserialize(const std::uint8_t *data,
                                               std::size_t size) {
    const std::size_t header =
        sizeof(std::uint32_t) + 4 * sizeof(std::uint64_t);
    if (size < header) throw std::invalid_argument("zfp codec: blob too small");
    std::uint32_t rate_u;
    std::memcpy(&rate_u, data, sizeof(rate_u));
    std::uint64_t fields[4];
    std::memcpy(fields, data + sizeof(rate_u), sizeof(fields));
    if (size != header + fields[3])
      throw std::invalid_argument("zfp codec: blob size mismatch");
    auto codec = std::make_unique<ZfpCodec>(rate_u);
    codec->n_ = fields[0];
    codec->n_padded_ = fields[1];
    codec->bits_per_block_ = fields[2];
    codec->buffer_.assign(data + header, data + header + fields[3]);
    codec->zfp_ = zfp_stream_open(nullptr);
    zfp_stream_set_rate(codec->zfp_, static_cast<double>(codec->rate_),
                        zfp_type_float, 1, zfp_false);
    codec->stream_ = stream_open(codec->buffer_.data(), codec->buffer_.size());
    zfp_stream_set_bit_stream(codec->zfp_, codec->stream_);
    return codec;
  }

 private:
  unsigned rate_;
  std::size_t n_ = 0;
  std::size_t n_padded_ = 0;
  std::size_t bits_per_block_ = 0;
  std::vector<unsigned char> buffer_;
  zfp_stream *zfp_;
  mutable bitstream *stream_;
};

std::unique_ptr<Codec> make_zfp_fixed_rate(unsigned rate) {
  return std::make_unique<ZfpCodec>(rate);
}

// ----- Adaptive ZFP (per-tile rate + outlier patches) ---------------------

namespace {

// Encode a tile to a freshly-allocated buffer at the given rate. Returns
// the raw byte buffer and exposes the bits-per-block via `bpb`. The encoded
// buffer is padded to byte boundary on the trailing edge.
std::vector<unsigned char> zfp_encode_tile(const float *src, std::size_t n,
                                           unsigned rate,
                                           std::size_t &bits_per_block_out) {
  const std::size_t n_padded = (n + 3) & ~static_cast<std::size_t>(3);
  std::vector<float> padded(n_padded);
  std::copy(src, src + n, padded.begin());
  for (std::size_t i = n; i < n_padded; ++i) padded[i] = src[n - 1];

  zfp_field *field =
      zfp_field_1d(padded.data(), zfp_type_float, static_cast<uint>(n_padded));
  zfp_stream *zfp = zfp_stream_open(nullptr);
  const double actual = zfp_stream_set_rate(zfp, static_cast<double>(rate),
                                            zfp_type_float, 1, zfp_false);
  bits_per_block_out = static_cast<std::size_t>(actual * 4.0 + 0.5);

  const std::size_t bufsize = zfp_stream_maximum_size(zfp, field);
  std::vector<unsigned char> buf(bufsize);
  bitstream *bs = stream_open(buf.data(), bufsize);
  zfp_stream_set_bit_stream(zfp, bs);
  zfp_stream_rewind(zfp);
  const std::size_t zfpsize = zfp_compress(zfp, field);
  stream_close(bs);
  zfp_stream_close(zfp);
  zfp_field_free(field);
  buf.resize(zfpsize);
  return buf;
}

// Decode a tile to scratch float array (used during encoder selection only).
void zfp_decode_tile(const unsigned char *buf, std::size_t bufsize,
                     unsigned rate, std::size_t n, float *out) {
  const std::size_t n_padded = (n + 3) & ~static_cast<std::size_t>(3);
  std::vector<float> padded(n_padded);
  zfp_field *field =
      zfp_field_1d(padded.data(), zfp_type_float, static_cast<uint>(n_padded));
  zfp_stream *zfp = zfp_stream_open(nullptr);
  zfp_stream_set_rate(zfp, static_cast<double>(rate), zfp_type_float, 1,
                      zfp_false);
  bitstream *bs = stream_open(const_cast<unsigned char *>(buf), bufsize);
  zfp_stream_set_bit_stream(zfp, bs);
  zfp_stream_rewind(zfp);
  zfp_decompress(zfp, field);
  std::copy(padded.begin(), padded.begin() + n, out);
  stream_close(bs);
  zfp_stream_close(zfp);
  zfp_field_free(field);
}

}  // namespace

class AdaptiveZfpCodec : public Codec {
 public:
  AdaptiveZfpCodec(std::vector<std::size_t> tile_offsets, double epsilon,
                   double max_outlier_frac)
      : tile_offsets_(std::move(tile_offsets)),
        epsilon_(epsilon),
        max_outlier_frac_(max_outlier_frac) {
    if (tile_offsets_.size() < 2)
      throw std::invalid_argument("need at least one tile");
    if (epsilon < 0) throw std::invalid_argument("epsilon must be >= 0");
    if (max_outlier_frac < 0 || max_outlier_frac > 1)
      throw std::invalid_argument("max_outlier_frac in [0,1]");
  }
  ~AdaptiveZfpCodec() override {
    if (stream_) stream_close(stream_);
    if (zfp_) zfp_stream_close(zfp_);
  }

  void encode(const float *values, std::size_t n) override {
    if (tile_offsets_.back() != n)
      throw std::invalid_argument("tile_offsets last entry != n samples");
    n_original_ = n;

    // NaN detection. Ocean fields routinely have continents → NaN. If any
    // are present we build, *per row*, a run-length-encoded mask: a list of
    // bit-transition positions plus the state at the row start. Continents
    // are spatially contiguous so rows typically have 2–8 transitions
    // instead of 1440 raw bits — ratio ~20–50× over the raw bitmap.
    has_nan_ = false;
    for (std::size_t i = 0; i < n; ++i) {
      if (!std::isfinite(values[i])) {
        has_nan_ = true;
        break;
      }
    }

    std::vector<float> finite_buffer;
    std::vector<std::size_t> enc_offsets;
    std::vector<std::vector<std::uint32_t>> local_to_orig;

    const std::size_t n_tiles = tile_offsets_.size() - 1;

    if (has_nan_) {
      // Build per-row RLE and finite buffer in a single pass.
      row_state0_.assign(n_tiles, 0);
      row_runs_offset_.assign(n_tiles + 1, 0);
      row_runs_.clear();
      row_finite_offset_.assign(n_tiles + 1, 0);
      finite_buffer.reserve(n);
      enc_offsets.assign(n_tiles + 1, 0);
      local_to_orig.resize(n_tiles);

      for (std::size_t t = 0; t < n_tiles; ++t) {
        const std::size_t lo = tile_offsets_[t];
        const std::size_t hi = tile_offsets_[t + 1];
        const std::uint8_t state0 =
            std::isfinite(values[lo]) ? std::uint8_t(1) : std::uint8_t(0);
        row_state0_[t] = state0;
        row_runs_offset_[t] = static_cast<std::uint32_t>(row_runs_.size());

        std::uint8_t state = state0;
        std::uint32_t finite_in_row = 0;
        local_to_orig[t].reserve(hi - lo);

        for (std::size_t i = lo; i < hi; ++i) {
          const std::uint8_t s =
              std::isfinite(values[i]) ? std::uint8_t(1) : std::uint8_t(0);
          if (s != state) {
            // Transition at position (i - lo) within the row.
            row_runs_.push_back(static_cast<std::uint32_t>(i - lo));
            state = s;
          }
          if (s) {
            finite_buffer.push_back(values[i]);
            local_to_orig[t].push_back(static_cast<std::uint32_t>(i - lo));
            finite_in_row++;
          }
        }
        enc_offsets[t + 1] = finite_buffer.size();
        row_finite_offset_[t + 1] = row_finite_offset_[t] + finite_in_row;
      }
      row_runs_offset_[n_tiles] = static_cast<std::uint32_t>(row_runs_.size());
      tile_finite_offset_ = enc_offsets;
    } else {
      enc_offsets = tile_offsets_;
    }

    const float *enc_values = has_nan_ ? finite_buffer.data() : values;

    // Inner ZFP loop, identical in dense/masked: per-tile rate selection
    // with outlier accounting. Tile sizes come from enc_offsets.
    static const unsigned kCandidates[] = {2, 3, 4, 6, 8, 12, 16};
    tile_rate_.assign(n_tiles, 0);
    tile_byte_offset_.assign(n_tiles + 1, 0);
    tile_bits_per_block_.assign(n_tiles, 0);

    std::vector<unsigned char> combined;
    combined.reserve(n * 2);
    std::vector<float> scratch;
    // Outliers indexed by ORIGINAL flat index (so decode can binary-search
    // before doing any rank conversion).
    std::vector<std::pair<std::uint32_t, float>> outliers;

    for (std::size_t t = 0; t < n_tiles; ++t) {
      const std::size_t lo = enc_offsets[t];
      const std::size_t hi = enc_offsets[t + 1];
      const std::size_t m = hi - lo;
      tile_byte_offset_[t] = combined.size();
      if (m == 0) {
        // All-NaN tile (typical for polar land bands). Skip ZFP entirely.
        tile_rate_[t] = 0;
        tile_bits_per_block_[t] = 0;
        continue;
      }
      const std::size_t max_outliers =
          static_cast<std::size_t>(std::ceil(m * max_outlier_frac_));
      scratch.assign(m, 0.0f);

      unsigned chosen_rate = kCandidates[0];
      std::vector<unsigned char> chosen_buf;
      std::size_t chosen_bpb = 0;
      std::vector<std::size_t> chosen_outliers_local;

      for (unsigned rate : kCandidates) {
        std::size_t bpb = 0;
        auto buf = zfp_encode_tile(enc_values + lo, m, rate, bpb);
        zfp_decode_tile(buf.data(), buf.size(), rate, m, scratch.data());
        std::vector<std::size_t> bad;
        for (std::size_t i = 0; i < m; ++i) {
          if (std::fabs(scratch[i] - enc_values[lo + i]) > epsilon_)
            bad.push_back(i);
        }
        if (bad.size() <= max_outliers ||
            rate ==
                kCandidates[(sizeof(kCandidates) / sizeof(*kCandidates)) - 1]) {
          chosen_rate = rate;
          chosen_buf = std::move(buf);
          chosen_bpb = bpb;
          chosen_outliers_local = std::move(bad);
          break;
        }
      }

      tile_rate_[t] = static_cast<std::uint8_t>(chosen_rate);
      tile_bits_per_block_[t] = chosen_bpb;
      combined.insert(combined.end(), chosen_buf.begin(), chosen_buf.end());
      for (std::size_t idx_local : chosen_outliers_local) {
        std::uint32_t orig_in_tile;
        if (has_nan_)
          orig_in_tile = local_to_orig[t][idx_local];
        else
          orig_in_tile = static_cast<std::uint32_t>(idx_local);
        outliers.emplace_back(
            static_cast<std::uint32_t>(tile_offsets_[t] + orig_in_tile),
            enc_values[lo + idx_local]);
      }
    }
    tile_byte_offset_.back() = combined.size();
    buffer_ = std::move(combined);

    std::sort(outliers.begin(), outliers.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    outlier_idx_.reserve(outliers.size());
    outlier_val_.reserve(outliers.size());
    for (auto &p : outliers) {
      outlier_idx_.push_back(p.first);
      outlier_val_.push_back(p.second);
    }

    zfp_ = zfp_stream_open(nullptr);
    stream_ = stream_open(buffer_.data(), buffer_.size());
    zfp_stream_set_bit_stream(zfp_, stream_);
  }

  // Walk the row's RLE runs for position `p_in_row` (relative to the row
  // start). Returns (state_at_p, rank_in_row), where rank_in_row is the
  // number of finite bits in positions [0, p_in_row) within this row.
  // Typical row has 1-8 transitions; the loop is O(k) and faster than a
  // binary search for small k.
  inline void row_mask_lookup(std::size_t t, std::uint32_t p_in_row,
                              std::uint8_t &state_out,
                              std::uint32_t &rank_in_row) const {
    const std::uint32_t off = row_runs_offset_[t];
    const std::uint32_t end = row_runs_offset_[t + 1];
    std::uint8_t state = row_state0_[t];
    std::uint32_t run_start = 0;
    std::uint32_t finite = 0;
    for (std::uint32_t k = off; k < end; ++k) {
      const std::uint32_t ti = row_runs_[k];
      if (p_in_row < ti) {
        state_out = state;
        rank_in_row = finite + (state ? (p_in_row - run_start) : 0);
        return;
      }
      if (state) finite += (ti - run_start);
      run_start = ti;
      state = !state;
    }
    state_out = state;
    rank_in_row = finite + (state ? (p_in_row - run_start) : 0);
  }

  float decode_one(std::size_t idx) const override {
    auto it = std::upper_bound(tile_offsets_.begin(), tile_offsets_.end(), idx);
    const std::size_t t =
        static_cast<std::size_t>(it - tile_offsets_.begin()) - 1;

    std::size_t local;
    if (has_nan_) {
      std::uint8_t state;
      std::uint32_t rank_in_row;
      const std::uint32_t p =
          static_cast<std::uint32_t>(idx - tile_offsets_[t]);
      row_mask_lookup(t, p, state, rank_in_row);
      if (!state) return std::numeric_limits<float>::quiet_NaN();
      local = static_cast<std::size_t>(rank_in_row);
    } else {
      local = idx - tile_offsets_[t];
    }
    if (tile_rate_[t] == 0)  // entire tile was NaN — shouldn't happen here
      return std::numeric_limits<float>::quiet_NaN();

    const std::size_t block = local >> 2;
    const std::size_t lane = local & 3;
    const std::size_t bpb = tile_bits_per_block_[t];
    const std::size_t bit_offset = tile_byte_offset_[t] * 8u + block * bpb;
    zfp_stream_set_rate(zfp_, static_cast<double>(tile_rate_[t]),
                        zfp_type_float, 1, zfp_false);
    stream_rseek(stream_, bit_offset);
    float buf[4];
    zfp_decode_block_float_1(zfp_, buf);
    float v = buf[lane];

    if (!outlier_idx_.empty()) {
      auto oit = std::lower_bound(outlier_idx_.begin(), outlier_idx_.end(),
                                  static_cast<std::uint32_t>(idx));
      if (oit != outlier_idx_.end() && *oit == idx)
        v = outlier_val_[oit - outlier_idx_.begin()];
    }
    return v;
  }

  std::size_t footprint_bytes() const override {
    std::size_t total = buffer_.size() +
                        tile_rate_.size() * sizeof(std::uint8_t) +
                        tile_byte_offset_.size() * sizeof(std::size_t) +
                        tile_bits_per_block_.size() * sizeof(std::size_t) +
                        outlier_idx_.size() * sizeof(std::uint32_t) +
                        outlier_val_.size() * sizeof(float) +
                        tile_offsets_.size() * sizeof(std::size_t);
    if (has_nan_) {
      total += row_state0_.size() * sizeof(std::uint8_t) +
               row_runs_.size() * sizeof(std::uint32_t) +
               row_runs_offset_.size() * sizeof(std::uint32_t) +
               row_finite_offset_.size() * sizeof(std::uint32_t) +
               tile_finite_offset_.size() * sizeof(std::size_t);
    }
    return total;
  }
  const char *name() const override {
    return has_nan_ ? "zfp-adaptive+mask" : "zfp-adaptive";
  }

  // Serialized layout (see header comments above for semantics). Lengths of
  // arrays whose size is determined by the grid (n_tiles, tile_offsets_)
  // are reconstructed at deserialize time; only sizes specific to this
  // instance (buffer, outliers, runs) are stored.
  std::vector<std::uint8_t> serialize() const override;
  static std::unique_ptr<AdaptiveZfpCodec> deserialize(
      const class ReducedGrid &grid, const std::uint8_t *data,
      std::size_t size);

 private:
  std::vector<std::size_t> tile_offsets_;  // ORIGINAL grid offsets (per row)
  double epsilon_;
  double max_outlier_frac_;

  std::vector<std::uint8_t> tile_rate_;
  std::vector<std::size_t> tile_byte_offset_;  // n_tiles + 1
  std::vector<std::size_t> tile_bits_per_block_;
  std::vector<unsigned char> buffer_;

  std::vector<std::uint32_t> outlier_idx_;
  std::vector<float> outlier_val_;

  // NaN-aware fields. Empty when has_nan_==false.
  bool has_nan_ = false;
  std::size_t n_original_ = 0;
  // Per-row RLE: row r's transition positions are in row_runs_ at offsets
  // [row_runs_offset_[r], row_runs_offset_[r+1]). State at row start is in
  // row_state0_[r] (0 = NaN, 1 = finite); state flips at each transition.
  std::vector<std::uint8_t> row_state0_;
  std::vector<std::uint32_t> row_runs_;           // transition positions
  std::vector<std::uint32_t> row_runs_offset_;    // n_tiles + 1
  std::vector<std::uint32_t> row_finite_offset_;  // n_tiles + 1, cumul finite
  std::vector<std::size_t>
      tile_finite_offset_;  // n_tiles + 1 (encoder offsets)

  zfp_stream *zfp_ = nullptr;
  mutable bitstream *stream_ = nullptr;
};

std::unique_ptr<Codec> make_zfp_adaptive(const ReducedGrid &grid,
                                         double epsilon,
                                         double max_outlier_fraction_per_tile) {
  std::vector<std::size_t> offsets(grid.n_rows() + 1);
  offsets[0] = 0;
  for (std::size_t r = 0; r < grid.n_rows(); ++r)
    offsets[r + 1] = offsets[r] + grid.n_lon(r);
  return std::make_unique<AdaptiveZfpCodec>(std::move(offsets), epsilon,
                                            max_outlier_fraction_per_tile);
}

// ---- AdaptiveZfp serialize / deserialize ---------------------------------

namespace {
template <typename T>
void append_pod(std::vector<std::uint8_t> &out, const T &v) {
  const std::size_t s = sizeof(T);
  const std::size_t off = out.size();
  out.resize(off + s);
  std::memcpy(out.data() + off, &v, s);
}
template <typename T>
void append_array(std::vector<std::uint8_t> &out, const std::vector<T> &v) {
  const std::size_t s = v.size() * sizeof(T);
  const std::size_t off = out.size();
  out.resize(off + s);
  if (s) std::memcpy(out.data() + off, v.data(), s);
}
template <typename T>
const std::uint8_t *read_array(const std::uint8_t *p, std::vector<T> &out,
                               std::size_t n) {
  out.resize(n);
  if (n) std::memcpy(out.data(), p, n * sizeof(T));
  return p + n * sizeof(T);
}
template <typename T>
const std::uint8_t *read_pod(const std::uint8_t *p, T &out) {
  std::memcpy(&out, p, sizeof(T));
  return p + sizeof(T);
}
}  // namespace

std::vector<std::uint8_t> AdaptiveZfpCodec::serialize() const {
  const std::size_t n_tiles = tile_rate_.size();
  std::vector<std::uint8_t> out;
  out.reserve(buffer_.size() + 256);
  // Header: version, has_nan, n_tiles, n_outliers, buffer_size
  append_pod<std::uint8_t>(out, /*version=*/1);
  append_pod<std::uint8_t>(out, has_nan_ ? 1 : 0);
  append_pod<std::uint64_t>(out, n_tiles);
  append_pod<std::uint64_t>(out, outlier_idx_.size());
  append_pod<std::uint64_t>(out, buffer_.size());
  append_array(out, tile_rate_);
  append_array(out, tile_byte_offset_);  // n_tiles + 1
  append_array(out, tile_bits_per_block_);
  append_array(out, buffer_);
  append_array(out, outlier_idx_);
  append_array(out, outlier_val_);
  if (has_nan_) {
    append_pod<std::uint64_t>(out, row_runs_.size());
    append_array(out, row_state0_);
    append_array(out, row_runs_);
    append_array(out, row_runs_offset_);
    append_array(out, row_finite_offset_);
    append_array(out, tile_finite_offset_);
  }
  return out;
}

std::unique_ptr<AdaptiveZfpCodec> AdaptiveZfpCodec::deserialize(
    const ReducedGrid &grid, const std::uint8_t *data, std::size_t size) {
  // Reconstruct tile_offsets from grid (matches the make_zfp_adaptive path).
  std::vector<std::size_t> offsets(grid.n_rows() + 1);
  offsets[0] = 0;
  for (std::size_t r = 0; r < grid.n_rows(); ++r)
    offsets[r + 1] = offsets[r] + grid.n_lon(r);

  auto codec = std::make_unique<AdaptiveZfpCodec>(std::move(offsets),
                                                  /*epsilon=*/0.0,
                                                  /*max_outlier_frac=*/0.0);
  const std::uint8_t *p = data;
  const std::uint8_t *end = data + size;
  std::uint8_t version, has_nan_u;
  p = read_pod(p, version);
  if (version != 1)
    throw std::invalid_argument("zfp-adaptive: unsupported version");
  p = read_pod(p, has_nan_u);
  codec->has_nan_ = has_nan_u != 0;
  std::uint64_t n_tiles, n_outliers, buf_size;
  p = read_pod(p, n_tiles);
  p = read_pod(p, n_outliers);
  p = read_pod(p, buf_size);
  if (n_tiles != grid.n_rows())
    throw std::invalid_argument("zfp-adaptive: tile count != grid rows");

  p = read_array(p, codec->tile_rate_, n_tiles);
  p = read_array(p, codec->tile_byte_offset_, n_tiles + 1);
  p = read_array(p, codec->tile_bits_per_block_, n_tiles);
  p = read_array(p, codec->buffer_, buf_size);
  p = read_array(p, codec->outlier_idx_, n_outliers);
  p = read_array(p, codec->outlier_val_, n_outliers);
  if (codec->has_nan_) {
    std::uint64_t n_runs;
    p = read_pod(p, n_runs);
    p = read_array(p, codec->row_state0_, n_tiles);
    p = read_array(p, codec->row_runs_, n_runs);
    p = read_array(p, codec->row_runs_offset_, n_tiles + 1);
    p = read_array(p, codec->row_finite_offset_, n_tiles + 1);
    p = read_array(p, codec->tile_finite_offset_, n_tiles + 1);
  }
  if (p != end)
    throw std::invalid_argument("zfp-adaptive: trailing bytes after blob");

  codec->zfp_ = zfp_stream_open(nullptr);
  codec->stream_ = stream_open(codec->buffer_.data(), codec->buffer_.size());
  zfp_stream_set_bit_stream(codec->zfp_, codec->stream_);
  return codec;
}

#endif  // OCTOGRID_WITH_ZFP

std::unique_ptr<Codec> make_uint16_row_tiled(const ReducedGrid &grid) {
  std::vector<std::size_t> offsets(grid.n_rows() + 1);
  offsets[0] = 0;
  for (std::size_t r = 0; r < grid.n_rows(); ++r)
    offsets[r + 1] = offsets[r] + grid.n_lon(r);
  return std::make_unique<Uint16Codec>(std::move(offsets));
}

std::unique_ptr<Codec> deserialize_codec(const ReducedGrid &grid,
                                         const std::string &name,
                                         const std::uint8_t *data,
                                         std::size_t size) {
  if (name == "raw") return RawFloat32Codec::deserialize(data, size);
  if (name == "bfloat16") return Bfloat16Codec::deserialize(data, size);
  if (name == "uint16-tiled") {
    std::vector<std::size_t> offsets(grid.n_rows() + 1);
    offsets[0] = 0;
    for (std::size_t r = 0; r < grid.n_rows(); ++r)
      offsets[r + 1] = offsets[r] + grid.n_lon(r);
    return Uint16Codec::deserialize(std::move(offsets), data, size);
  }
#ifdef OCTOGRID_WITH_ZFP
  if (name == "zfp-fixed-rate") return ZfpCodec::deserialize(data, size);
  if (name == "zfp-adaptive" || name == "zfp-adaptive+mask")
    return AdaptiveZfpCodec::deserialize(grid, data, size);
#endif
  throw std::invalid_argument("unknown codec name: " + name);
}

}  // namespace octogrid
