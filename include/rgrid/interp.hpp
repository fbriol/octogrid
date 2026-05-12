#pragma once
#include <cstddef>

#include "field.hpp"

namespace rgrid {

// Nearest neighbour (B1): iso-latitude row search + 1D longitude search.
// No global kd-tree. O(log n_rows + log n_lon).
float interp_nearest(const CompressedField &f, double lat_deg, double lon_deg);

// Barycentric interpolation (B3): on the implicit triangulation formed by
// pairs of adjacent latitude rows. For each query, locates the surrounding
// quadrilateral (2 points per row), splits into two triangles along its
// diagonal, identifies the containing triangle by sign of barycentric
// coordinates, and returns the weighted sum of 3 vertex values.
//
// 3 codec fetches + ~10 FLOPs. The hot path under the compression-first
// axiom (see plan §3.4).
float interp_barycentric(const CompressedField &f, double lat_deg,
                         double lon_deg);

// Batched API. Outputs values for `n` query points. Returns NaN for queries
// outside the grid's latitude span.
void interp_barycentric_batch(const CompressedField &f, const double *lat_deg,
                              const double *lon_deg, std::size_t n, float *out);

void interp_nearest_batch(const CompressedField &f, const double *lat_deg,
                          const double *lon_deg, std::size_t n, float *out);

}  // namespace rgrid
