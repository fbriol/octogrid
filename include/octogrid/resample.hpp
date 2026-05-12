#pragma once
#include <cstddef>

#include "grid.hpp"

namespace octogrid {

// Bilinear resampling from a regular lat/lon source grid onto an arbitrary
// ReducedGrid target. Longitude is periodic (wraps at 360°). NaN propagates
// through the bilinear stencil: any NaN among the 4 source corners yields
// NaN at the target.
//
// Preconditions on source axes (caller responsibility):
//   - src_lats strictly DECREASING from index 0 (north pole side) to
//     n_src_lats - 1 (south pole side). Same orientation as ReducedGrid.
//   - src_lons strictly INCREASING in [0, 360). Period 360 assumed.
//   - src_data points to n_src_lats * n_src_lons float32 values, row-major
//     ((lat, lon), i.e. row stride = n_src_lons).
//
// `out` must have at least target.n_points() entries; it is overwritten.
void resample_from_latlon(const ReducedGrid &target, const double *src_lats,
                          std::size_t n_src_lats, const double *src_lons,
                          std::size_t n_src_lons, const float *src_data,
                          float *out);

}  // namespace octogrid
