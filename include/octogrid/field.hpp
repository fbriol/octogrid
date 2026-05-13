#pragma once
#include <memory>

#include "codec.hpp"
#include "grid.hpp"

namespace octogrid {

// A compressed scalar field on a ReducedGrid.
// Stores its grid by value so that ownership semantics are clean across the
// C++/Python boundary; grids are cheap (two vectors sized to n_rows).
class CompressedField {
 public:
  // Build from raw float32 values: calls codec->encode(values, ...).
  CompressedField(ReducedGrid grid, std::unique_ptr<Codec> codec,
                  const float *values);

  // Build from an already-encoded codec (e.g. after deserialize).
  CompressedField(ReducedGrid grid, std::unique_ptr<Codec> codec);

  [[nodiscard]] auto grid() const -> const ReducedGrid & { return grid_; }
  [[nodiscard]] auto codec() const -> const Codec & { return *codec_; }

  [[nodiscard]] auto at(std::size_t linear_idx) const -> float {
    return codec_->decode_one(linear_idx);
  }

  [[nodiscard]] auto footprint_bytes() const -> std::size_t {
    return codec_->footprint_bytes();
  }

 private:
  ReducedGrid grid_;
  std::unique_ptr<Codec> codec_;
};

}  // namespace octogrid
