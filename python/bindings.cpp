#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <memory>
#include <stdexcept>

#include "rgrid/codec.hpp"
#include "rgrid/field.hpp"
#include "rgrid/grid.hpp"
#include "rgrid/interp.hpp"

namespace nb = nanobind;
using namespace rgrid;

// Aliases for 1D float32 / float64 contiguous CPU arrays.
using FloatArray =
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using DoubleArray =
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
// Return type: a numpy float32 array owned by a capsule that releases the
// buffer when garbage-collected.
using FloatOut = nb::ndarray<float, nb::numpy, nb::ndim<1>>;

namespace {

CompressedField make_field(const ReducedGrid &grid,
                           const std::string &codec_name, FloatArray values,
                           unsigned zfp_rate, double epsilon,
                           double max_outlier_frac) {
  if (values.size() != grid.n_points())
    throw std::invalid_argument("values length != grid.n_points()");
  std::unique_ptr<Codec> codec;
  if (codec_name == "bfloat16") {
    codec = make_bfloat16();
  } else if (codec_name == "uint16") {
    codec = make_uint16_row_tiled(grid);
  } else if (codec_name == "zfp") {
#ifdef RGRID_WITH_ZFP
    codec = make_zfp_fixed_rate(zfp_rate);
#else
    throw std::runtime_error("zfp codec not compiled in");
#endif
  } else if (codec_name == "zfp_adaptive") {
#ifdef RGRID_WITH_ZFP
    codec = make_zfp_adaptive(grid, epsilon, max_outlier_frac);
#else
    throw std::runtime_error("zfp codec not compiled in");
#endif
  } else {
    throw std::invalid_argument("unknown codec: " + codec_name);
  }
  return CompressedField(grid, std::move(codec), values.data());
}

FloatOut batch_interp(const CompressedField &f, DoubleArray lat,
                      DoubleArray lon, const std::string &method) {
  if (lat.size() != lon.size())
    throw std::invalid_argument("lat and lon must have same length");
  const std::size_t n = lat.size();

  // Allocate the output and bind its lifetime to a Python-owned capsule so
  // that nanobind hands NumPy a buffer it can free independently.
  float *data = new float[n];
  nb::capsule owner(data,
                    [](void *p) noexcept { delete[] static_cast<float *>(p); });

  if (method == "barycentric") {
    interp_barycentric_batch(f, lat.data(), lon.data(), n, data);
  } else if (method == "nearest") {
    interp_nearest_batch(f, lat.data(), lon.data(), n, data);
  } else {
    delete[] data;
    throw std::invalid_argument("unknown method: " + method);
  }

  size_t shape[1] = {n};
  return FloatOut(data, /*ndim=*/1, shape, owner);
}

}  // namespace

NB_MODULE(_rgrid, m) {
  m.doc() = "rgrid — compact in-memory reduced grids and interpolation";

  nb::class_<ReducedGrid>(m, "ReducedGrid")
      .def(nb::init<std::vector<double>, std::vector<std::uint32_t>>(),
           nb::arg("latitudes_deg"), nb::arg("n_lon"))
      .def_static("octahedral", &ReducedGrid::octahedral, nb::arg("n_lat"),
                  nb::arg("base") = 20u)
      .def_static("regular", &ReducedGrid::regular, nb::arg("n_lat"),
                  nb::arg("n_lon"))
      .def_prop_ro("n_rows", &ReducedGrid::n_rows)
      .def_prop_ro("n_points", &ReducedGrid::n_points)
      .def("n_lon", &ReducedGrid::n_lon, nb::arg("row"))
      .def("lat_deg", &ReducedGrid::lat_deg, nb::arg("row"));

  nb::class_<CompressedField>(m, "CompressedField")
      .def_prop_ro("footprint_bytes", &CompressedField::footprint_bytes)
      .def_prop_ro("codec_name",
                   [](const CompressedField &f) { return f.codec().name(); })
      .def_prop_ro("n_points", [](const CompressedField &f) {
        return f.grid().n_points();
      });

  m.def("compress", &make_field, nb::arg("grid"), nb::arg("codec"),
        nb::arg("values"), nb::arg("zfp_rate") = 8u, nb::arg("epsilon") = 1e-3,
        nb::arg("max_outlier_frac") = 0.01,
        "Build a CompressedField. codec ∈ "
        "{'bfloat16','uint16','zfp','zfp_adaptive'}. "
        "zfp_rate (bits/value) only used for 'zfp'. "
        "epsilon, max_outlier_frac only used for 'zfp_adaptive'.");

  m.def("interpolate", &batch_interp, nb::arg("field"), nb::arg("lat_deg"),
        nb::arg("lon_deg"), nb::arg("method") = "barycentric",
        "Interpolate field at query points. "
        "method ∈ {'barycentric','nearest'}.");
}
