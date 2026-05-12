#include "octogrid/field.hpp"

namespace octogrid {

CompressedField::CompressedField(ReducedGrid grid, std::unique_ptr<Codec> codec,
                                 const float *values)
    : grid_(std::move(grid)), codec_(std::move(codec)) {
  codec_->encode(values, grid_.n_points());
}

CompressedField::CompressedField(ReducedGrid grid, std::unique_ptr<Codec> codec)
    : grid_(std::move(grid)), codec_(std::move(codec)) {}

}  // namespace octogrid
