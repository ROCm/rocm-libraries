// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// nanobind bindings for the standalone mosaic kernel-recommender library.
// Self-contained: depends ONLY on mosaic's public headers (no HIP / GEMM
// framework headers).

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <vector>

#include "mosaic/model.hpp"
#include "mosaic/types.hpp"

namespace nb = nanobind;
using namespace nanobind::literals;

NB_MODULE(mosaic, m) {
  m.doc() = "Standalone framework-agnostic GEMM kernel recommender (mosaic).";

  // ── enums ─────────────────────────────────────────────────────────────────
  nb::enum_<mosaic::DataType>(m, "DataType")
      .value("Float", mosaic::DataType::Float)
      .value("Double", mosaic::DataType::Double)
      .value("ComplexFloat", mosaic::DataType::ComplexFloat)
      .value("ComplexDouble", mosaic::DataType::ComplexDouble)
      .value("Half", mosaic::DataType::Half)
      .value("Int8x4", mosaic::DataType::Int8x4)
      .value("Int32", mosaic::DataType::Int32)
      .value("BFloat16", mosaic::DataType::BFloat16)
      .value("Int8", mosaic::DataType::Int8)
      .value("Int4", mosaic::DataType::Int4)
      .value("Int64", mosaic::DataType::Int64)
      .value("XFloat32", mosaic::DataType::XFloat32)
      .value("Float8_fnuz", mosaic::DataType::Float8_fnuz)
      .value("BFloat8_fnuz", mosaic::DataType::BFloat8_fnuz)
      .value("Float8BFloat8_fnuz", mosaic::DataType::Float8BFloat8_fnuz)
      .value("BFloat8Float8_fnuz", mosaic::DataType::BFloat8Float8_fnuz)
      .value("Float8", mosaic::DataType::Float8)
      .value("BFloat8", mosaic::DataType::BFloat8)
      .value("Float8BFloat8", mosaic::DataType::Float8BFloat8)
      .value("BFloat8Float8", mosaic::DataType::BFloat8Float8)
      .value("Float6", mosaic::DataType::Float6)
      .value("BFloat6", mosaic::DataType::BFloat6)
      .value("Float4", mosaic::DataType::Float4)
      .value("Count", mosaic::DataType::Count)
      .value("None", mosaic::DataType::None)
      .export_values();

  nb::enum_<mosaic::Transpose>(m, "Transpose")
      .value("T", mosaic::Transpose::T)
      .value("N", mosaic::Transpose::N)
      .value("Count", mosaic::Transpose::Count)
      .export_values();

  // ── structs ───────────────────────────────────────────────────────────────
  nb::class_<mosaic::Dim3>(m, "Dim3")
      .def(nb::init<>())
      .def_rw("m", &mosaic::Dim3::m)
      .def_rw("n", &mosaic::Dim3::n)
      .def_rw("k", &mosaic::Dim3::k)
      .def("mn", &mosaic::Dim3::mn)
      .def("mk", &mosaic::Dim3::mk)
      .def("nk", &mosaic::Dim3::nk);

  nb::class_<mosaic::Problem>(m, "Problem")
      .def(nb::init<>())
      .def_rw("size", &mosaic::Problem::size)
      .def_rw("batch", &mosaic::Problem::batch)
      .def_rw("a_transpose", &mosaic::Problem::a_transpose)
      .def_rw("b_transpose", &mosaic::Problem::b_transpose)
      .def_rw("a_dtype", &mosaic::Problem::a_dtype)
      .def_rw("b_dtype", &mosaic::Problem::b_dtype)
      .def_rw("c_dtype", &mosaic::Problem::c_dtype)
      .def_rw("d_dtype", &mosaic::Problem::d_dtype)
      .def_rw("mi_dtype", &mosaic::Problem::mi_dtype);

  nb::class_<mosaic::Config>(m, "Config")
      .def(nb::init<>())
      .def_rw("mt", &mosaic::Config::mt)
      .def_rw("mi", &mosaic::Config::mi)
      .def_rw("occupancy", &mosaic::Config::occupancy)
      .def_rw("cache_hints_a", &mosaic::Config::cache_hints_a)
      .def_rw("cache_hints_b", &mosaic::Config::cache_hints_b)
      .def_rw("grvw_a", &mosaic::Config::grvw_a)
      .def_rw("grvw_b", &mosaic::Config::grvw_b)
      .def_rw("gwvw_d", &mosaic::Config::gwvw_d)
      .def_rw("vector_width_a", &mosaic::Config::vector_width_a)
      .def_rw("vector_width_b", &mosaic::Config::vector_width_b)
      .def_rw("depth_u", &mosaic::Config::depth_u)
      .def_rw("global_split_u", &mosaic::Config::global_split_u)
      .def_rw("index", &mosaic::Config::index)
      // Extended ML features (consumed by the item/inter towers).
      .def_rw("cache_hints_c", &mosaic::Config::cache_hints_c)
      .def_rw("cache_hints_d", &mosaic::Config::cache_hints_d)
      .def_rw("cache_hints_e", &mosaic::Config::cache_hints_e)
      .def_rw("prefetch_global_read", &mosaic::Config::prefetch_global_read)
      .def_rw("prefetch_local_read", &mosaic::Config::prefetch_local_read)
      .def_rw("lds_read_vector_width", &mosaic::Config::lds_read_vector_width)
      .def_rw("local_split_u", &mosaic::Config::local_split_u)
      .def_rw("lds_pad_a", &mosaic::Config::lds_pad_a)
      .def_rw("lds_pad_b", &mosaic::Config::lds_pad_b)
      .def_rw("lds_buffer_pad_a", &mosaic::Config::lds_buffer_pad_a)
      .def_rw("lds_buffer_pad_b", &mosaic::Config::lds_buffer_pad_b);

  nb::class_<mosaic::Hardware>(m, "Hardware")
      .def(nb::init<>())
      .def_rw("N_CU", &mosaic::Hardware::N_CU)
      .def_rw("lds_capacity", &mosaic::Hardware::lds_capacity)
      .def_rw("L2_capacity", &mosaic::Hardware::L2_capacity)
      .def_rw("parallel_mi_cu", &mosaic::Hardware::parallel_mi_cu)
      // std::tuple<double,double,double> via nanobind/stl/tuple.h caster.
      .def_rw("mem_bw_per_wg_coefficients",
              &mosaic::Hardware::mem_bw_per_wg_coefficients);

  nb::class_<mosaic::Result>(m, "Result")
      .def(nb::init<>())
      .def_rw("config_index", &mosaic::Result::config_index)
      .def_rw("score", &mosaic::Result::score)
      .def_rw("scored", &mosaic::Result::scored);

  // ── free functions ──────────────────────────────────────────────────────--
  m.def("load_weights", &mosaic::load_weights, nb::arg("bin_path"),
        "Explicitly load MLREC_v6 weights from a .bin path. Returns false on "
        "any I/O or format error.");

  m.def("weights_loaded", &mosaic::weights_loaded,
        "True once a model has been successfully loaded.");

  m.def("route", &mosaic::route, nb::arg("problem"),
        "Route a problem to its leaf model-cell index (or -1).");

  m.def("rank_configs", &mosaic::rank_configs,
        nb::arg("problem"), nb::arg("hardware"), nb::arg("configs"),
        "Rank candidate configs (each carrying its own ML features) for a "
        "problem. Returns a Result per input config: survivors first "
        "(scored=True, descending score), then filtered-out configs "
        "(scored=False).");
}
