// Minimal smoke test for the octogrid core: build a small octahedral grid,
// encode a synthetic analytic field with each codec, interpolate on known
// points, and check error stays within codec-specific tolerance.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "octogrid/codec.hpp"
#include "octogrid/field.hpp"
#include "octogrid/grid.hpp"
#include "octogrid/interp.hpp"

namespace {

// Smooth analytic field on the sphere: f(lat, lon) = sin(lat) * cos(lon).
float analytic(double lat_deg, double lon_deg) {
  const double d2r = M_PI / 180.0;
  return static_cast<float>(std::sin(lat_deg * d2r) * std::cos(lon_deg * d2r));
}

int check(bool cond, const char *msg) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  using namespace octogrid;
  int failures = 0;

  // 1. Build an octahedral grid with ~modest resolution.
  auto grid = ReducedGrid::octahedral(/*n_lat=*/64, /*base=*/20);

  // 2. Fill it with the analytic field, decoded value-by-value.
  std::vector<float> values(grid.n_points());
  for (std::size_t r = 0; r < grid.n_rows(); ++r) {
    const double lat = grid.lat_deg(r);
    for (std::uint32_t i = 0; i < grid.n_lon(r); ++i) {
      values[grid.linear_index(r, i)] = analytic(lat, grid.lon_deg(r, i));
    }
  }

  std::printf(
      "octahedral grid: %zu rows, %zu total points, raw float32 = "
      "%.1f KB\n",
      grid.n_rows(), grid.n_points(), grid.n_points() * sizeof(float) / 1024.0);

  // 3. Test each codec.
  struct CodecCase {
    const char *name;
    std::unique_ptr<Codec> (*make)(const ReducedGrid &);
  };
  auto make_bf16 = [](const ReducedGrid &) { return make_bfloat16(); };
  auto make_u16 = [](const ReducedGrid &g) { return make_uint16_row_tiled(g); };

  struct Case {
    const char *name;
    std::unique_ptr<Codec> codec;
    double tol_nearest;
    double tol_bary;
  };
  std::vector<Case> cases;
  cases.push_back({"bfloat16", make_bf16(grid), 5e-2, 5e-3});
  cases.push_back({"uint16-tiled", make_u16(grid), 5e-2, 5e-3});

  for (auto &c : cases) {
    CompressedField field(grid, std::move(c.codec), values.data());
    std::printf("\n[%s] footprint = %.1f KB (ratio = %.2fx)\n", c.name,
                field.footprint_bytes() / 1024.0,
                static_cast<double>(grid.n_points() * sizeof(float)) /
                    field.footprint_bytes());

    // Round-trip: decode every stored point and compare to encoded.
    double max_err = 0.0;
    for (std::size_t i = 0; i < grid.n_points(); ++i) {
      const double err = std::fabs(field.at(i) - values[i]);
      if (err > max_err) max_err = err;
    }
    std::printf("  round-trip max err = %.3e\n", max_err);

    // Off-grid interpolation: pick a query that is NOT a grid point.
    const double qlat = 17.3;
    const double qlon = 42.7;
    const float expected = analytic(qlat, qlon);
    const float vN = interp_nearest(field, qlat, qlon);
    const float vB = interp_barycentric(field, qlat, qlon);
    std::printf("  query (%.1f, %.1f) — true=%.6f  nearest=%.6f  bary=%.6f\n",
                qlat, qlon, expected, vN, vB);

    failures += check(std::fabs(vN - expected) < c.tol_nearest,
                      "nearest error within tolerance");
    failures += check(std::fabs(vB - expected) < c.tol_bary,
                      "barycentric error within tolerance");

    // Barycentric must beat nearest on a smooth field.
    failures +=
        check(std::fabs(vB - expected) <= std::fabs(vN - expected) + 1e-3,
              "barycentric should be at least as good as nearest");
  }

  if (failures == 0) {
    std::printf("\nALL OK\n");
    return 0;
  }
  std::fprintf(stderr, "\n%d failure(s)\n", failures);
  return 1;
}
