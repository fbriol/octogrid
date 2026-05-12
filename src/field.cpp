#include "rgrid/field.hpp"

namespace rgrid {

CompressedField::CompressedField(const ReducedGrid &grid,
                                 std::unique_ptr<Codec> codec,
                                 const float *values)
    : grid_(grid), codec_(std::move(codec)) {
  codec_->encode(values, grid_.n_points());
}

}  // namespace rgrid
