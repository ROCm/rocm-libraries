// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/origami.hpp"
#include "origami/streamk.hpp"
#include "origami/types.hpp"

using hardware_t = origami::hardware_t;

NB_MODULE(origami, m) {
  nanobind::enum_<hardware_t::architecture_t>(m, "architecture_t")
      .value("gfx942", hardware_t::architecture_t::gfx942)
      .value("gfx950", hardware_t::architecture_t::gfx950)
      .export_values();

  nanobind::enum_<origami::data_type_t>(m, "data_type_t")
      .value("Float", origami::data_type_t::Float)
      .value("ComplexFloat", origami::data_type_t::ComplexFloat)
      .value("ComplexDouble", origami::data_type_t::ComplexDouble)
      .value("Double", origami::data_type_t::Double)
      .value("Half", origami::data_type_t::Half)
      .value("Int8x4", origami::data_type_t::Int8x4)
      .value("Int32", origami::data_type_t::Int32)
      .value("BFloat16", origami::data_type_t::BFloat16)
      .value("Int8", origami::data_type_t::Int8)
      .value("Int64", origami::data_type_t::Int64)
      .value("XFloat32", origami::data_type_t::XFloat32)
      .value("Float8_fnuz", origami::data_type_t::Float8_fnuz)
      .value("BFloat8_fnuz", origami::data_type_t::BFloat8_fnuz)
      .value("Float8BFloat8_fnuz", origami::data_type_t::Float8BFloat8_fnuz)
      .value("BFloat8Float8_fnuz", origami::data_type_t::BFloat8Float8_fnuz)
      .value("Float8", origami::data_type_t::Float8)
      .value("BFloat8", origami::data_type_t::BFloat8)
      .value("Float8BFloat8", origami::data_type_t::Float8BFloat8)
      .value("BFloat8Float8", origami::data_type_t::BFloat8Float8)
      .value("Float6", origami::data_type_t::Float6)
      .value("BFloat6", origami::data_type_t::BFloat6)
      .value("Float4", origami::data_type_t::Float4)
      .export_values();

  // After your other nanobind::enum_ blocks
  nanobind::enum_<origami::transpose_t>(m, "transpose_t")
      .value("T", origami::transpose_t::T)
      .value("N", origami::transpose_t::N)
      // Optional: usually you don't expose Count, but you can if you want
      // .value("Count", origami::transpose_t::Count)
      .export_values();

  m.def("int_to_data_type", &origami::int_to_data_type, "Convert int to data_type_t.");

  nanobind::enum_<origami::grid_selection_t>(m, "grid_selection_t")
      .value("number_of_cus", origami::grid_selection_t::number_of_cus)
      .value("min_resources", origami::grid_selection_t::min_resources)
      .value("energy_aware", origami::grid_selection_t::energy_aware)
      .value("reduction_cost_aware", origami::grid_selection_t::reduction_cost_aware)
      .value("data_parallel", origami::grid_selection_t::data_parallel)
      .value("analytical", origami::grid_selection_t::analytical)
      .value("k_split_aware", origami::grid_selection_t::k_split_aware)
      .export_values();

  nanobind::enum_<origami::reduction_t>(m, "reduction_t")
      .value("Tree", origami::reduction_t::tree)
      .value("Parallel", origami::reduction_t::parallel)
      .export_values();

  m.def("int_to_reduction_t", &origami::int_to_reduction_t, "Convert int to reduction_t.");

  // Add new struct bindings
  nanobind::class_<origami::dim3_t>(m, "dim3_t")
      .def(nanobind::init<std::size_t, std::size_t, std::size_t>())
      .def_rw("m", &origami::dim3_t::m)
      .def_rw("n", &origami::dim3_t::n)
      .def_rw("k", &origami::dim3_t::k)
      .def("mn", &origami::dim3_t::mn)
      .def("mk", &origami::dim3_t::mk)
      .def("nk", &origami::dim3_t::nk)
      .def("mnk", &origami::dim3_t::mnk);

  nanobind::class_<origami::config_t>(m, "config_t")
      .def(nanobind::init<>())
      .def_rw("mt", &origami::config_t::mt)
      .def_rw("mi", &origami::config_t::mi)
      .def_rw("occupancy", &origami::config_t::occupancy)
      .def_rw("workgroup_mapping", &origami::config_t::workgroup_mapping)
      .def_rw("non_temporal_a", &origami::config_t::non_temporal_a)
      .def_rw("non_temporal_b", &origami::config_t::non_temporal_b);

  nanobind::class_<origami::prediction_result_t>(m, "prediction_result_t")
      .def(nanobind::init<>())
      .def_rw("latency", &origami::prediction_result_t::latency)
      .def_rw("config", &origami::prediction_result_t::config);

  nanobind::class_<origami::problem_t>(m, "problem_t")
      .def(nanobind::init<>())
      .def_rw("size", &origami::problem_t::size)
      .def_rw("batch", &origami::problem_t::batch)
      .def_rw("transpose_a", &origami::problem_t::a_transpose)
      .def_rw("transpose_b", &origami::problem_t::b_transpose)
      .def_rw("a_dtype", &origami::problem_t::a_dtype)
      .def_rw("b_dtype", &origami::problem_t::b_dtype)
      .def_rw("c_dtype", &origami::problem_t::c_dtype)
      .def_rw("d_dtype", &origami::problem_t::d_dtype)
      .def_rw("mi_dtype", &origami::problem_t::mi_dtype)
      .def_rw("a_mx_block_size", &origami::problem_t::a_mx_block_size)
      .def_rw("b_mx_block_size", &origami::problem_t::b_mx_block_size);

  nanobind::class_<hardware_t>(m, "hardware_t")
      .def(nanobind::init<hardware_t::architecture_t,
                          size_t,
                          size_t,
                          size_t,
                          double,
                          double,
                          double,
                          size_t,
                          double,
                          size_t,
                          std::tuple<double, double, double>>())
      .def("print", &hardware_t::print)
      .def("print_debug_info", &hardware_t::print_debug_info)
      .def("extract_analytical_metrics_csv", &hardware_t::extract_analytical_metrics_csv)
      .def("get_analytical_metrics", &hardware_t::get_analytical_metrics)
      .def("set_metrics_collection_mode", &hardware_t::set_metrics_collection_mode)
      .def_rw("N_CU", &hardware_t::N_CU)
      .def_rw("lds_capacity", &hardware_t::lds_capacity)
      .def_rw("mem1_perf_ratio", &hardware_t::mem1_perf_ratio)
      .def_rw("mem2_perf_ratio", &hardware_t::mem2_perf_ratio)
      .def_rw("mem3_perf_ratio", &hardware_t::mem3_perf_ratio)
      .def_rw("L2_capacity", &hardware_t::L2_capacity)
      .def_rw("CU_per_L2", &hardware_t::CU_per_L2)
      .def_rw("compute_clock_ghz", &hardware_t::compute_clock_ghz)
      .def_rw("parallel_mi_cu", &hardware_t::parallel_mi_cu)
      .def_rw("mem_bw_per_wg_coefficients", &hardware_t::mem_bw_per_wg_coefficients)
      .def_rw("NUM_XCD", &hardware_t::NUM_XCD)
      .def_rw("debug_info", &hardware_t::debug_info);

  m.def("get_hardware_for_device",
        &hardware_t::get_hardware_for_device,
        "This gets a hardware object for a device.");

  m.def("datatype_to_bits", &origami::data_type_to_bits, "Return the number of bits in a datatype");
  m.def("string_to_datatype",
        &origami::string_to_data_type,
        "Convert a string representation of a datatype into data_type_t enum");

  m.def("select_config",
        &origami::select_config,
        "Select best configuration based on problem and hardware");
  m.def("select_grid_size",
        &origami::streamk::select_grid_size,
        "Select best grid size for the given configuration");
  m.def("select_workgroup_mapping",
        &origami::select_workgroup_mapping,
        "Select best workgroup mapping");
  m.def("rank_configs", &origami::rank_configs, "Rank configurations by performance");
  m.def("select_config_mnk",
        &origami::select_config_mnk,
        "Select best configuration for M,N,K dimensions");
  m.def("select_topk_configs", &origami::select_topk_configs, "Select top K configurations");
  m.def("compute_perf_gflops", &origami::compute_perf_gflops, "Compute performance in GFLOPS");

  // StreamK functions
  m.def("select_reduction",
        &origami::streamk::select_reduction,
        "Select best StreamK reduction strategy");

  // GEMM functions
  m.def("compute_total_latency",
        static_cast<double (*)(
            const origami::problem_t&, const origami::hardware_t&, const origami::config_t&, size_t max_cus)>(
            &origami::compute_total_latency),
        "Compute total latency");

  // Analytical metrics extraction functions
  m.def("extract_analytical_metrics",
        &origami::extract_analytical_metrics,
        "Extract analytical metrics from GEMM computation");
  m.def("extract_analytical_metrics_csv",
        &origami::extract_analytical_metrics_csv,
        "Extract analytical metrics and export to CSV");
}