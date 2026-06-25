// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// nanobind bindings for the standalone tilewright kernel-recommender library.
// Self-contained: depends ONLY on tilewright's public headers (no HIP / GEMM
// framework headers).

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <vector>

#include "tilewright/model.hpp"
#include "tilewright/types.hpp"

namespace nb = nanobind;
using namespace nanobind::literals;

NB_MODULE(tilewright, m) {
  m.doc() = "Standalone framework-agnostic GEMM kernel recommender (tilewright).";

  // ── enums ─────────────────────────────────────────────────────────────────
  nb::enum_<tilewright::DataType>(m, "DataType")
      .value("Float", tilewright::DataType::Float)
      .value("Double", tilewright::DataType::Double)
      .value("ComplexFloat", tilewright::DataType::ComplexFloat)
      .value("ComplexDouble", tilewright::DataType::ComplexDouble)
      .value("Half", tilewright::DataType::Half)
      .value("Int8x4", tilewright::DataType::Int8x4)
      .value("Int32", tilewright::DataType::Int32)
      .value("BFloat16", tilewright::DataType::BFloat16)
      .value("Int8", tilewright::DataType::Int8)
      .value("Int4", tilewright::DataType::Int4)
      .value("Int64", tilewright::DataType::Int64)
      .value("XFloat32", tilewright::DataType::XFloat32)
      .value("Float8_fnuz", tilewright::DataType::Float8_fnuz)
      .value("BFloat8_fnuz", tilewright::DataType::BFloat8_fnuz)
      .value("Float8BFloat8_fnuz", tilewright::DataType::Float8BFloat8_fnuz)
      .value("BFloat8Float8_fnuz", tilewright::DataType::BFloat8Float8_fnuz)
      .value("Float8", tilewright::DataType::Float8)
      .value("BFloat8", tilewright::DataType::BFloat8)
      .value("Float8BFloat8", tilewright::DataType::Float8BFloat8)
      .value("BFloat8Float8", tilewright::DataType::BFloat8Float8)
      .value("Float6", tilewright::DataType::Float6)
      .value("BFloat6", tilewright::DataType::BFloat6)
      .value("Float4", tilewright::DataType::Float4)
      .value("Count", tilewright::DataType::Count)
      .value("None", tilewright::DataType::None)
      .export_values();

  nb::enum_<tilewright::Transpose>(m, "Transpose")
      .value("T", tilewright::Transpose::T)
      .value("N", tilewright::Transpose::N)
      .value("Count", tilewright::Transpose::Count)
      .export_values();

  // ── structs ───────────────────────────────────────────────────────────────
  nb::class_<tilewright::Dim3>(m, "Dim3")
      .def(nb::init<>())
      .def_rw("m", &tilewright::Dim3::m)
      .def_rw("n", &tilewright::Dim3::n)
      .def_rw("k", &tilewright::Dim3::k)
      .def("mn", &tilewright::Dim3::mn)
      .def("mk", &tilewright::Dim3::mk)
      .def("nk", &tilewright::Dim3::nk);

  nb::class_<tilewright::Problem>(m, "Problem")
      .def(nb::init<>())
      .def_rw("size", &tilewright::Problem::size)
      .def_rw("batch", &tilewright::Problem::batch)
      .def_rw("a_transpose", &tilewright::Problem::a_transpose)
      .def_rw("b_transpose", &tilewright::Problem::b_transpose)
      .def_rw("a_dtype", &tilewright::Problem::a_dtype)
      .def_rw("b_dtype", &tilewright::Problem::b_dtype)
      .def_rw("c_dtype", &tilewright::Problem::c_dtype)
      .def_rw("d_dtype", &tilewright::Problem::d_dtype)
      .def_rw("mi_dtype", &tilewright::Problem::mi_dtype);

  nb::class_<tilewright::Config>(m, "Config")
      .def(nb::init<>())
      .def_rw("mt", &tilewright::Config::mt)
      .def_rw("mi", &tilewright::Config::mi)
      .def_rw("occupancy", &tilewright::Config::occupancy)
      .def_rw("cache_hints_a", &tilewright::Config::cache_hints_a)
      .def_rw("cache_hints_b", &tilewright::Config::cache_hints_b)
      .def_rw("grvw_a", &tilewright::Config::grvw_a)
      .def_rw("grvw_b", &tilewright::Config::grvw_b)
      .def_rw("gwvw_d", &tilewright::Config::gwvw_d)
      .def_rw("index", &tilewright::Config::index);

  nb::class_<tilewright::Hardware>(m, "Hardware")
      .def(nb::init<>())
      .def_rw("N_CU", &tilewright::Hardware::N_CU)
      .def_rw("lds_capacity", &tilewright::Hardware::lds_capacity)
      .def_rw("L2_capacity", &tilewright::Hardware::L2_capacity)
      .def_rw("parallel_mi_cu", &tilewright::Hardware::parallel_mi_cu)
      // std::tuple<double,double,double> via nanobind/stl/tuple.h caster.
      .def_rw("mem_bw_per_wg_coefficients", &tilewright::Hardware::mem_bw_per_wg_coefficients);

  nb::class_<tilewright::Result>(m, "Result")
      .def(nb::init<>())
      .def_rw("config_index", &tilewright::Result::config_index)
      .def_rw("score", &tilewright::Result::score)
      .def_rw("scored", &tilewright::Result::scored);

  // ── free functions ──────────────────────────────────────────────────────--
  m.def("load_weights",
        &tilewright::load_weights,
        nb::arg("bin_path"),
        "Explicitly load MLREC_v1 weights from a .bin path. Returns false on "
        "any I/O or format error.");

  m.def("weights_loaded",
        &tilewright::weights_loaded,
        "True once a model has been successfully loaded.");

  m.def("route",
        static_cast<int (*)(const tilewright::Problem&)>(&tilewright::route),
        nb::arg("problem"),
        "Route a problem to its leaf model-cell index (or -1).");

  m.def("rank_configs",
        static_cast<std::vector<tilewright::Result> (*)(const tilewright::Problem&,
                                                        const tilewright::Hardware&,
                                                        const std::vector<tilewright::Config>&,
                                                        std::size_t)>(&tilewright::rank_configs),
        nb::arg("problem"),
        nb::arg("hardware"),
        nb::arg("configs"),
        nb::arg("min_scored") = 0,
        "Rank candidate configs (each carrying its own ML features) for a "
        "problem. Returns a Result per input config: survivors first "
        "(scored=True, descending score), then filtered-out configs "
        "(scored=False). `min_scored` requests ranking depth beyond the "
        "per-cell smart_K whitelist (0 = whitelist only; see model.hpp).");
}
