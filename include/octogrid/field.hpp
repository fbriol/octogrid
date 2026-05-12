#pragma once
#include <memory>

#include "codec.hpp"
#include "grid.hpp"

namespace octogrid {

// A compressed scalar field on a ReducedGrid.
// Owns the codec (and its buffers). Holds a non-owning ref to the grid —
// grids are cheap and typically shared across many fields.
class CompressedField {
 public:
  CompressedField(const ReducedGrid &grid, std::unique_ptr<Codec> codec,
                  const float *values);

  const ReducedGrid &grid() const { return grid_; }
  const Codec &codec() const { return *codec_; }

  float at(std::size_t linear_idx) const {
    return codec_->decode_one(linear_idx);
  }

  std::size_t footprint_bytes() const { return codec_->footprint_bytes(); }

 private:
  const ReducedGrid &grid_;
  std::unique_ptr<Codec> codec_;
};

}  // namespace octogrid
