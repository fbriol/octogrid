#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <memory>
#include <stdexcept>

#include "octogrid/codec.hpp"
#include "octogrid/field.hpp"
#include "octogrid/grid.hpp"
#include "octogrid/interp.hpp"
#include "octogrid/resample.hpp"

namespace nb = nanobind;

// Aliases for 1D float32 / float64 contiguous CPU arrays.
using FloatArray =
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

using DoubleArray =
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Return type: a numpy float32 array owned by a capsule that releases the
// buffer when garbage-collected.
using FloatOut = nb::ndarray<float, nb::numpy, nb::ndim<1>>;
using DoubleOut = nb::ndarray<double, nb::numpy, nb::ndim<1>>;

namespace {

auto make_field(const octogrid::ReducedGrid &grid,
                const std::string &codec_name, FloatArray values,
                unsigned zfp_rate, double epsilon, double max_outlier_frac)
    -> octogrid::CompressedField {
  if (values.size() != grid.n_points()) {
    throw std::invalid_argument("values length != grid.n_points()");
  }
  std::unique_ptr<octogrid::Codec> codec;
  if (codec_name == "raw") {
    codec = octogrid::make_raw();
  } else if (codec_name == "bfloat16") {
    codec = octogrid::make_bfloat16();
  } else if (codec_name == "uint16") {
    codec = octogrid::make_uint16_row_tiled(grid);
  } else if (codec_name == "zfp") {
#ifdef OCTOGRID_WITH_ZFP
    codec = octogrid::make_zfp_fixed_rate(zfp_rate);
#else
    throw std::runtime_error("zfp codec not compiled in");
#endif
  } else if (codec_name == "zfp_adaptive") {
#ifdef OCTOGRID_WITH_ZFP
    codec = octogrid::make_zfp_adaptive(grid, epsilon, max_outlier_frac);
#else
    throw std::runtime_error("zfp codec not compiled in");
#endif
  } else {
    throw std::invalid_argument("unknown codec: " + codec_name);
  }
  return {grid, std::move(codec), values.data()};
}

auto batch_interp(const octogrid::CompressedField &f, DoubleArray lat,
                  DoubleArray lon, const std::string &method) -> FloatOut {
  if (lat.size() != lon.size()) {
    throw std::invalid_argument("lat and lon must have same length");
  }
  const std::size_t n = lat.size();

  // Use a unique_ptr so that any exception thrown below cleans the buffer
  // without colliding with the capsule deleter we attach on success.

  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  std::unique_ptr<float[]> buf(new float[n]);

  if (method == "barycentric") {
    interp_barycentric_batch(f, lat.data(), lon.data(), n, buf.get());
  } else if (method == "nearest") {
    interp_nearest_batch(f, lat.data(), lon.data(), n, buf.get());
  } else {
    throw std::invalid_argument("unknown method: " + method);
  }
  float *data = buf.release();
  nb::capsule owner(data,
                    [](void *p) noexcept { delete[] static_cast<float *>(p); });
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  size_t shape[1] = {n};
  return {data, /*ndim=*/1, shape, owner};
}

// Bilinear resampling from a regular lat/lon source onto a ReducedGrid.
// The Python facade is responsible for delivering axes in the canonical
// orientation (lats decreasing, lons increasing in [0, 360)) and a
// C-contiguous (n_lat, n_lon) float32 source array.
using Float2D =
    nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

auto resample(const octogrid::ReducedGrid &grid, DoubleArray src_lats,
              DoubleArray src_lons, Float2D src_arr) -> FloatOut {
  const std::size_t n_lat = src_lats.size();
  const std::size_t n_lon = src_lons.size();
  if (src_arr.shape(0) != n_lat || src_arr.shape(1) != n_lon) {
    throw std::invalid_argument("src_arr shape does not match (n_lat, n_lon)");
  }

  const std::size_t n = grid.n_points();
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  std::unique_ptr<float[]> buf(new float[n]);
  resample_from_latlon(grid, src_lats.data(), n_lat, src_lons.data(), n_lon,
                       src_arr.data(), buf.get());
  float *data = buf.release();
  nb::capsule owner(data,
                    [](void *p) noexcept { delete[] static_cast<float *>(p); });
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  size_t shape[1] = {n};
  return {data, /*ndim=*/1, shape, owner};
}

}  // namespace

NB_MODULE(_octogrid, m) {
  m.doc() = "octogrid — compact in-memory reduced grids and interpolation";

  nb::class_<octogrid::ReducedGrid>(m, "ReducedGrid")
      .def(nb::init<std::vector<double>, std::vector<std::uint32_t>>(),
           nb::arg("latitudes_deg"), nb::arg("n_lon"))
      .def_static("octahedral", &octogrid::ReducedGrid::octahedral,
                  nb::arg("n_lat"), nb::arg("base") = 20u)
      .def_static("regular", &octogrid::ReducedGrid::regular, nb::arg("n_lat"),
                  nb::arg("n_lon"))
      .def_prop_ro("n_rows", &octogrid::ReducedGrid::n_rows)
      .def_prop_ro("n_points", &octogrid::ReducedGrid::n_points)
      .def("n_lon", &octogrid::ReducedGrid::n_lon, nb::arg("row"))
      .def("lat_deg", &octogrid::ReducedGrid::lat_deg, nb::arg("row"))
      .def(
          "latitudes",
          [](const octogrid::ReducedGrid &g) {
            const std::size_t n = g.n_points();
            // NOLINTNEXTLINE(modernize-avoid-c-arrays)
            std::unique_ptr<double[]> buf(new double[n]);
            g.fill_latitudes(buf.get());
            double *data = buf.release();
            nb::capsule owner(data, [](void *p) noexcept {
              delete[] static_cast<double *>(p);
            });
            // NOLINTNEXTLINE(modernize-avoid-c-arrays)
            size_t shape[1] = {n};
            return DoubleOut(data, /*ndim=*/1, shape, owner);
          },
          "Latitude (deg) of every grid point, flat — shape (n_points,).")
      .def(
          "longitudes",
          [](const octogrid::ReducedGrid &g) {
            const std::size_t n = g.n_points();
            // NOLINTNEXTLINE(modernize-avoid-c-arrays)
            std::unique_ptr<double[]> buf(new double[n]);
            g.fill_longitudes(buf.get());
            double *data = buf.release();
            nb::capsule owner(data, [](void *p) noexcept {
              delete[] static_cast<double *>(p);
            });
            // NOLINTNEXTLINE(modernize-avoid-c-arrays)
            size_t shape[1] = {n};
            return DoubleOut(data, /*ndim=*/1, shape, owner);
          },
          "Longitude (deg, in [0, 360)) of every grid point, flat.");

  nb::class_<octogrid::CompressedField>(m, "CompressedField")
      .def_prop_ro("footprint_bytes",
                   &octogrid::CompressedField::footprint_bytes)
      .def_prop_ro(
          "codec_name",
          [](const octogrid::CompressedField &f) { return f.codec().name(); })
      .def_prop_ro(
          "grid",
          [](const octogrid::CompressedField &f)
              -> const octogrid::ReducedGrid & { return f.grid(); },
          nb::rv_policy::reference_internal)
      .def_prop_ro("n_points",
                   [](const octogrid::CompressedField &f) {
                     return f.grid().n_points();
                   })
      .def(
          "to_numpy",
          [](const octogrid::CompressedField &f) {
            const std::size_t n = f.grid().n_points();
            // NOLINTNEXTLINE(modernize-avoid-c-arrays)
            std::unique_ptr<float[]> buf(new float[n]);
            f.codec().decode_all(buf.get(), n);
            float *data = buf.release();
            nb::capsule owner(data, [](void *p) noexcept {
              delete[] static_cast<float *>(p);
            });
            // NOLINTNEXTLINE(modernize-avoid-c-arrays)
            size_t shape[1] = {n};
            return FloatOut(data, /*ndim=*/1, shape, owner);
          },
          "Decode every value into a fresh numpy float32 array of length "
          "n_points. NaN is preserved.");

  m.def("compress", &make_field, nb::arg("grid"), nb::arg("codec"),
        nb::arg("values"), nb::arg("zfp_rate") = 8u, nb::arg("epsilon") = 1e-3,
        nb::arg("max_outlier_frac") = 0.01,
        "Build a CompressedField. codec ∈ "
        "{'raw','bfloat16','uint16','zfp','zfp_adaptive'}. "
        "zfp_rate (bits/value) only used for 'zfp'. "
        "epsilon, max_outlier_frac only used for 'zfp_adaptive'.");

  m.def("interpolate", &batch_interp, nb::arg("field"), nb::arg("lat_deg"),
        nb::arg("lon_deg"), nb::arg("method") = "barycentric",
        "Interpolate field at query points. "
        "method ∈ {'barycentric','nearest'}.");

  m.def("resample_from_latlon", &resample, nb::arg("grid"), nb::arg("src_lats"),
        nb::arg("src_lons"), nb::arg("src_arr"),
        "Bilinearly resample a regular lat/lon source onto a ReducedGrid. "
        "src_lats must be strictly decreasing; src_lons strictly increasing "
        "in [0, 360); src_arr shape (n_lat, n_lon). NaN propagates.");

  // ---- Serialization --------------------------------------------------

  m.def(
      "_serialize_codec",
      [](const octogrid::CompressedField &f) {
        const auto blob = f.codec().serialize();
        // Copy out via capsule-owned uint8 ndarray.
        auto *data = new std::uint8_t[blob.size()];
        std::memcpy(data, blob.data(), blob.size());
        nb::capsule owner(data, [](void *p) noexcept {
          delete[] static_cast<std::uint8_t *>(p);
        });
        // NOLINTNEXTLINE(modernize-avoid-c-arrays)
        size_t shape[1] = {blob.size()};
        return nb::ndarray<std::uint8_t, nb::numpy, nb::ndim<1>>(
            data, /*ndim=*/1, shape, owner);
      },
      nb::arg("field"),
      "Return the codec's serialized state as a numpy uint8 array.");

  m.def(
      "_field_from_blob",
      [](const octogrid::ReducedGrid &grid, const std::string &codec_name,
         nb::ndarray<const std::uint8_t, nb::ndim<1>, nb::c_contig,
                     nb::device::cpu>
             blob) {
        auto codec =
            deserialize_codec(grid, codec_name, blob.data(), blob.size());
        return octogrid::CompressedField(grid, std::move(codec));
      },
      nb::arg("grid"), nb::arg("codec_name"), nb::arg("blob"),
      "Rebuild a CompressedField from a grid + codec name + serialized blob.");
}
