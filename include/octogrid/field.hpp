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

  const ReducedGrid &grid() const { return grid_; }
  const Codec &codec() const { return *codec_; }

  float at(std::size_t linear_idx) const {
    return codec_->decode_one(linear_idx);
  }

  std::size_t footprint_bytes() const { return codec_->footprint_bytes(); }

 private:
  ReducedGrid grid_;
  std::unique_ptr<Codec> codec_;
};

}  // namespace octogrid
