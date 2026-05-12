#include "octogrid/resample.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace octogrid {

namespace {

// Find the largest index `i` such that arr[i] >= value (arr decreasing).
// Returns 0 if value > arr[0], n-1 if value < arr[n-1]. Always returns a
// valid bracket start: the pair (i, i+1) covers value for i < n-1.
inline std::size_t bracket_decreasing(const double *arr, std::size_t n,
                                      double value) {
  // Binary search for the first element strictly less than `value`. Since
  // arr is decreasing, std::lower_bound with greater-comparator finds the
  // first index whose entry is < value.
  auto it = std::lower_bound(arr, arr + n, value,
                             [](double a, double b) { return a > b; });
  if (it == arr) return 0;
  if (it == arr + n) return n - 2;  // clamp so that i+1 < n
  return static_cast<std::size_t>(it - arr) - 1;
}

// Find the bracket (c0, c1) in src_lons for query lon_q, with wrap.
// Returns (c0, c1) and a fractional weight wfrac in [0, 1) such that
// lon_q ≈ (1-wfrac) * lon[c0] + wfrac * lon[c1] (longitudes unwrapped
// onto a common period). `lon_lo` is the unwrapped longitude of c0 with
// the periodic correction baked in, useful for computing wfrac when the
// query straddles the seam.
inline void bracket_lon(const double *src_lons, std::size_t n, double lon_q,
                        std::size_t &c0, std::size_t &c1, double &wfrac) {
  // Normalize lon_q to [0, 360).
  double l = std::fmod(lon_q, 360.0);
  if (l < 0) l += 360.0;

  if (l < src_lons[0]) {
    // Query is in the wrap region [src_lons[n-1], src_lons[0] + 360).
    c0 = n - 1;
    c1 = 0;
    const double lon_hi = src_lons[0] + 360.0;
    wfrac = (l + 360.0 - src_lons[c0]) / (lon_hi - src_lons[c0]);
    return;
  }
  if (l >= src_lons[n - 1]) {
    // Either >= last and < first+360 → wrap segment.
    c0 = n - 1;
    c1 = 0;
    const double lon_hi = src_lons[0] + 360.0;
    wfrac = (l - src_lons[c0]) / (lon_hi - src_lons[c0]);
    return;
  }
  // Standard case: src_lons[c0] <= l < src_lons[c0+1].
  auto it = std::upper_bound(src_lons, src_lons + n, l);
  c0 = static_cast<std::size_t>(it - src_lons) - 1;
  c1 = c0 + 1;
  wfrac = (l - src_lons[c0]) / (src_lons[c1] - src_lons[c0]);
}

}  // namespace

void resample_from_latlon(const ReducedGrid &target, const double *src_lats,
                          std::size_t n_src_lats, const double *src_lons,
                          std::size_t n_src_lons, const float *src_data,
                          float *out) {
  // Iterate target rows / longitudes. For each, locate bracket and bilinear.
  for (std::size_t r = 0; r < target.n_rows(); ++r) {
    const double lat_q = target.lat_deg(r);
    // Latitude bracket in the source.
    std::size_t latN, latS;
    // Clamp out-of-range queries by returning NaN.
    bool lat_in_range = true;
    if (lat_q > src_lats[0] || lat_q < src_lats[n_src_lats - 1])
      lat_in_range = false;
    if (lat_in_range) {
      const std::size_t b = bracket_decreasing(src_lats, n_src_lats, lat_q);
      latN = b;
      latS = b + 1;
    } else {
      latN = latS = 0;
    }
    const double t = lat_in_range ? (src_lats[latN] - lat_q) /
                                        (src_lats[latN] - src_lats[latS])
                                  : 0.0;

    const std::size_t row_off = target.row_offset(r);
    const std::uint32_t n_lon = target.n_lon(r);
    const double lon_step = 360.0 / static_cast<double>(n_lon);

    for (std::uint32_t i = 0; i < n_lon; ++i) {
      const double lon_q = i * lon_step;
      if (!lat_in_range) {
        out[row_off + i] = std::numeric_limits<float>::quiet_NaN();
        continue;
      }
      std::size_t cW, cE;
      double u;
      bracket_lon(src_lons, n_src_lons, lon_q, cW, cE, u);

      const float vNW = src_data[latN * n_src_lons + cW];
      const float vNE = src_data[latN * n_src_lons + cE];
      const float vSW = src_data[latS * n_src_lons + cW];
      const float vSE = src_data[latS * n_src_lons + cE];

      const float vN = static_cast<float>((1.0 - u) * vNW + u * vNE);
      const float vS = static_cast<float>((1.0 - u) * vSW + u * vSE);
      out[row_off + i] = static_cast<float>((1.0 - t) * vN + t * vS);
    }
  }
}

}  // namespace octogrid
