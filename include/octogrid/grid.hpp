#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace octogrid {

// Generalized reduced grid: latitudes (decreasing from north to south) +
// number of equally-spaced longitude points per row. Covers classical
// reduced Gaussian (RGG), octahedral RGG, and any user-provided layout.
//
// Memory layout: flat 1D, row-major (row 0 = northernmost).
// Indexing within a row: lon[i] = lon0 + i * (360 / n_lon), i in [0, n_lon).
//
// This class stores ONLY topology (lat[], n_lon[], row offsets). Values live
// in a CompressedField that pairs a Grid with a Codec.
class ReducedGrid {
 public:
  ReducedGrid(std::vector<double> latitudes_deg,
              std::vector<std::uint32_t> n_lon);

  std::size_t n_rows() const { return lat_.size(); }
  std::size_t n_points() const { return total_points_; }
  std::uint32_t n_lon(std::size_t row) const { return n_lon_[row]; }
  double lat_deg(std::size_t row) const { return lat_[row]; }

  // Offset of the first point of `row` in the flat 1D layout.
  std::size_t row_offset(std::size_t row) const { return offsets_[row]; }

  // Longitude (degrees, in [0, 360)) of point i within row.
  double lon_deg(std::size_t row, std::uint32_t i) const;

  // Linear index <-> (row, col_in_row).
  std::size_t linear_index(std::size_t row, std::uint32_t i) const {
    return offsets_[row] + i;
  }

  // Find the two rows bracketing a query latitude (in degrees).
  // Returns (north_row, south_row). If lat is outside the grid in latitude,
  // both indices are clamped (the caller can detect by equality).
  void bracket_rows(double lat_deg, std::size_t &north,
                    std::size_t &south) const;

  // Find the two longitude indices in `row` bracketing query lon (degrees).
  // Wraps around 360°. Returns (west_idx, east_idx) and fractional weight
  // wfrac in [0,1) such that lon = (1-wfrac)*lon[west] + wfrac*lon[east].
  void bracket_lon(std::size_t row, double lon_deg, std::uint32_t &west,
                   std::uint32_t &east, double &wfrac) const;

  // Factory: octahedral-style reduced grid.
  // n_lat must be even. Latitudes are equally spaced in colatitude (a fair
  // approximation of true Gaussian roots, sufficient for the prototype).
  // n_lon(row) = base + 4 * dist_from_pole_row, mirrored about equator —
  // the same growth law as ECMWF's Ox grid (base typically 20).
  static ReducedGrid octahedral(std::size_t n_lat, std::uint32_t base = 20);

  // Factory: regular lat/lon grid (uniform n_lon across all rows).
  // Useful as baseline for benchmarks and tests.
  static ReducedGrid regular(std::size_t n_lat, std::uint32_t n_lon);

 private:
  std::vector<double> lat_;           // monotonic, decreasing (N -> S)
  std::vector<std::uint32_t> n_lon_;  // per-row longitude count
  std::vector<std::size_t> offsets_;  // n_rows + 1, cumulative
  std::size_t total_points_;
};

}  // namespace octogrid
