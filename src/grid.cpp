#include "rgrid/grid.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace rgrid {

ReducedGrid::ReducedGrid(std::vector<double> latitudes_deg,
                         std::vector<std::uint32_t> n_lon)
    : lat_(std::move(latitudes_deg)), n_lon_(std::move(n_lon)) {
  if (lat_.size() != n_lon_.size())
    throw std::invalid_argument("latitudes and n_lon must have same length");
  if (lat_.empty())
    throw std::invalid_argument("grid must have at least one row");
  // Enforce decreasing latitude order (N -> S). Detect strict monotonic.
  for (std::size_t i = 1; i < lat_.size(); ++i) {
    if (!(lat_[i] < lat_[i - 1]))
      throw std::invalid_argument("latitudes must be strictly decreasing");
  }
  offsets_.resize(lat_.size() + 1);
  offsets_[0] = 0;
  for (std::size_t i = 0; i < lat_.size(); ++i) {
    if (n_lon_[i] == 0)
      throw std::invalid_argument("n_lon must be > 0 for every row");
    offsets_[i + 1] = offsets_[i] + n_lon_[i];
  }
  total_points_ = offsets_.back();
}

double ReducedGrid::lon_deg(std::size_t row, std::uint32_t i) const {
  const double step = 360.0 / static_cast<double>(n_lon_[row]);
  return i * step;
}

void ReducedGrid::bracket_rows(double lat_deg, std::size_t &north,
                               std::size_t &south) const {
  // lat_ is strictly decreasing. Use binary search for the first index whose
  // lat is <= lat_deg.
  auto it = std::lower_bound(lat_.begin(), lat_.end(), lat_deg,
                             [](double a, double b) { return a > b; });
  if (it == lat_.begin()) {
    north = south = 0;  // query is north of grid
    return;
  }
  if (it == lat_.end()) {
    north = south = lat_.size() - 1;  // query is south of grid
    return;
  }
  south = static_cast<std::size_t>(it - lat_.begin());
  north = south - 1;
}

void ReducedGrid::bracket_lon(std::size_t row, double lon_deg,
                              std::uint32_t &west, std::uint32_t &east,
                              double &wfrac) const {
  const std::uint32_t n = n_lon_[row];
  const double step = 360.0 / static_cast<double>(n);
  // Wrap lon to [0, 360).
  double l = std::fmod(lon_deg, 360.0);
  if (l < 0) l += 360.0;
  const double f = l / step;
  const std::uint32_t w = static_cast<std::uint32_t>(std::floor(f)) % n;
  const std::uint32_t e = (w + 1) % n;
  west = w;
  east = e;
  wfrac = f - std::floor(f);
}

ReducedGrid ReducedGrid::octahedral(std::size_t n_lat, std::uint32_t base) {
  if (n_lat < 2 || (n_lat % 2) != 0)
    throw std::invalid_argument("n_lat must be even and >= 2");
  // Equally spaced colatitudes: this is the prototype-quality approximation
  // of Gaussian roots (good enough for correctness of bracketing /
  // interpolation; the true Gaussian variant is a drop-in replacement).
  std::vector<double> lats(n_lat);
  for (std::size_t i = 0; i < n_lat; ++i) {
    const double colat = (i + 0.5) * 180.0 / static_cast<double>(n_lat);
    lats[i] = 90.0 - colat;  // decreasing from ~90 to ~-90
  }
  // Octahedral growth law (ECMWF Ox-grid): n_lon(row) = base + 4 * d_pole,
  // where d_pole is the row's distance to the nearest pole.
  std::vector<std::uint32_t> n_lon(n_lat);
  const std::size_t half = n_lat / 2;
  for (std::size_t i = 0; i < n_lat; ++i) {
    const std::size_t d_pole = (i < half) ? i : (n_lat - 1 - i);
    n_lon[i] = base + 4 * static_cast<std::uint32_t>(d_pole);
  }
  return ReducedGrid(std::move(lats), std::move(n_lon));
}

ReducedGrid ReducedGrid::regular(std::size_t n_lat, std::uint32_t n_lon_each) {
  if (n_lat < 2) throw std::invalid_argument("n_lat must be >= 2");
  std::vector<double> lats(n_lat);
  for (std::size_t i = 0; i < n_lat; ++i) {
    const double colat = (i + 0.5) * 180.0 / static_cast<double>(n_lat);
    lats[i] = 90.0 - colat;
  }
  std::vector<std::uint32_t> n_lon(n_lat, n_lon_each);
  return ReducedGrid(std::move(lats), std::move(n_lon));
}

}  // namespace rgrid
