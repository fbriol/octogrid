#include "octogrid/interp.hpp"

#include <cmath>

namespace octogrid {

namespace {

// Normalize an angular value (degrees) to be the closest representative
// of `lon` modulo 360 to the reference `ref` (i.e., lies in (ref-180,
// ref+180]).
constexpr auto unwrap_to(double lon, double ref) -> double {
  double d = lon - ref;
  while (d > 180.0) {
    lon -= 360.0;
    d = lon - ref;
  }
  while (d <= -180.0) {
    lon += 360.0;
    d = lon - ref;
  }
  return lon;
}

struct Vertex {
  double lon;  // in a local unwrapped frame centered on the query
  double lat;
  std::size_t idx;
};

// Compute the signed area of triangle (A, B, C) in (lon, lat) plane.
// Sign convention: positive when (A, B, C) is counter-clockwise.
constexpr auto signed_area(const Vertex &A, const Vertex &B, const Vertex &C)
    -> double {
  return (B.lon - A.lon) * (C.lat - A.lat) - (C.lon - A.lon) * (B.lat - A.lat);
}

}  // namespace

auto interp_nearest(const CompressedField &f, double lat_deg, double lon_deg)
    -> float {
  const auto &g = f.grid();
  std::size_t rN, rS;
  g.bracket_rows(lat_deg, rN, rS);
  const double dN = std::fabs(g.lat_deg(rN) - lat_deg);
  const double dS = std::fabs(g.lat_deg(rS) - lat_deg);
  const std::size_t r = (dN <= dS) ? rN : rS;
  std::uint32_t w, e;
  double wfrac;
  g.bracket_lon(r, lon_deg, w, e, wfrac);
  const std::uint32_t i = (wfrac < 0.5) ? w : e;
  return f.at(g.linear_index(r, i));
}

auto interp_barycentric(const CompressedField &f, double lat_deg,
                        double lon_deg) -> float {
  const auto &g = f.grid();
  std::size_t rN, rS;
  g.bracket_rows(lat_deg, rN, rS);

  if (rN == rS) {
    // Query outside the grid's latitude span: 1D fall-back along the
    // edge row. Two fetches, linear in longitude.
    std::uint32_t w, e;
    double wfrac;
    g.bracket_lon(rN, lon_deg, w, e, wfrac);
    const float vw = f.at(g.linear_index(rN, w));
    const float ve = f.at(g.linear_index(rN, e));
    return static_cast<float>((1.0 - wfrac) * vw + wfrac * ve);
  }

  // Find the 4 surrounding quad vertices in (lon, lat) — but DON'T decode
  // values yet. We first determine which of the two triangles (split along
  // the NW-SE diagonal) contains the query, then issue exactly 3 codec
  // fetches. This is the B3 contract from the plan.
  std::uint32_t wN, eN, wS, eS;
  double fN, fS;  // unused here; we use absolute longitudes
  g.bracket_lon(rN, lon_deg, wN, eN, fN);
  g.bracket_lon(rS, lon_deg, wS, eS, fS);

  const double latN = g.lat_deg(rN);
  const double latS = g.lat_deg(rS);

  // Bring all longitudes into a common unwrapped frame around the query.
  Vertex NW{.lon = unwrap_to(g.lon_deg(rN, wN), lon_deg),
            .lat = latN,
            .idx = g.linear_index(rN, wN)};
  Vertex NE{.lon = unwrap_to(g.lon_deg(rN, eN), lon_deg),
            .lat = latN,
            .idx = g.linear_index(rN, eN)};
  Vertex SW{.lon = unwrap_to(g.lon_deg(rS, wS), lon_deg),
            .lat = latS,
            .idx = g.linear_index(rS, wS)};
  Vertex SE{.lon = unwrap_to(g.lon_deg(rS, eS), lon_deg),
            .lat = latS,
            .idx = g.linear_index(rS, eS)};

  // NE/SE wrap correction: when the row's n_lon is small, the "east"
  // neighbour of the western-most bracket index might wrap back near lon=0;
  // unwrap_to handles that, but it might still place NE west of NW. Force
  // the east vertex to lie east (numerically larger lon) than the west one
  // by adding 360° if needed — preserves grid topology.
  if (NE.lon < NW.lon) NE.lon += 360.0;
  if (SE.lon < SW.lon) SE.lon += 360.0;

  // Query point in the same frame.
  Vertex Q{.lon = lon_deg, .lat = lat_deg, .idx = 0};

  // Diagonal NW-SE splits the quad. Pick the triangle containing Q via
  // signed area test on the diagonal.
  // T0 = (NW, NE, SE) — upper-right of diagonal (NE side).
  // T1 = (NW, SE, SW) — lower-left of diagonal (SW side).
  // Compute the diagonal sign for NE and SW: they straddle the diagonal,
  // so they have opposite signs. The query is on whichever side matches.
  const double s_q = signed_area(NW, SE, Q);
  const double s_ne = signed_area(NW, SE, NE);

  const Vertex *A;
  const Vertex *B;
  const Vertex *C;
  if (s_q * s_ne >= 0.0) {
    // Same side as NE → triangle T0.
    A = &NW;
    B = &NE;
    C = &SE;
  } else {
    // Opposite → triangle T1.
    A = &NW;
    B = &SE;
    C = &SW;
  }

  // Barycentric weights of Q in (A, B, C).
  const double denom = (B->lat - C->lat) * (A->lon - C->lon) +
                       (C->lon - B->lon) * (A->lat - C->lat);
  // denom == 0 only if the triangle is degenerate (collinear); shouldn't
  // happen with valid grid rows. If it does, fall back to vertex A.
  if (std::fabs(denom) < 1e-30) return f.at(A->idx);
  const double inv = 1.0 / denom;
  const double wA = ((B->lat - C->lat) * (Q.lon - C->lon) +
                     (C->lon - B->lon) * (Q.lat - C->lat)) *
                    inv;
  const double wB = ((C->lat - A->lat) * (Q.lon - C->lon) +
                     (A->lon - C->lon) * (Q.lat - C->lat)) *
                    inv;
  const double wC = 1.0 - wA - wB;

  // 3 codec fetches.
  const float vA = f.at(A->idx);
  const float vB = f.at(B->idx);
  const float vC = f.at(C->idx);
  return static_cast<float>(wA * vA + wB * vB + wC * vC);
}

void interp_barycentric_batch(const CompressedField &f, const double *lat,
                              const double *lon, std::size_t n, float *out) {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = interp_barycentric(f, lat[i], lon[i]);
  }
}

void interp_nearest_batch(const CompressedField &f, const double *lat,
                          const double *lon, std::size_t n, float *out) {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = interp_nearest(f, lat[i], lon[i]);
  }
}

}  // namespace octogrid
