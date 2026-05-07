// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
using namespace nanobind::literals;

NB_MODULE(origami, m) {
  nanobind::enum_<hardware_t::architecture_t>(m, "architecture_t")
      .value("gfx90a", hardware_t::architecture_t::gfx90a)
      .value("gfx942", hardware_t::architecture_t::gfx942)
      .value("gfx950", hardware_t::architecture_t::gfx950)
      .value("gfx1201", hardware_t::architecture_t::gfx1201)
      .value("gfx1100", hardware_t::architecture_t::gfx1100)
      .value("gfx1150", hardware_t::architecture_t::gfx1150)
      .value("gfx1151", hardware_t::architecture_t::gfx1151)
      .value("gfx1152", hardware_t::architecture_t::gfx1152)
      .value("gfx1153", hardware_t::architecture_t::gfx1153)
      .value("gfx1250", hardware_t::architecture_t::gfx1250)
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
      .value("Int4", origami::data_type_t::Int4)
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
      .value("Spinlock", origami::reduction_t::spinlock)
      .value("Tree", origami::reduction_t::tree)
      .value("Parallel", origami::reduction_t::parallel)
      .value("Atomic", origami::reduction_t::atomic)
      .export_values();

  m.def("int_to_reduction_t", &origami::int_to_reduction_t, "Convert int to reduction_t.");

  nanobind::enum_<origami::prediction_modes_t>(m, "prediction_modes_t")
      .value("estimation", origami::prediction_modes_t::estimation)
      .value("simulation", origami::prediction_modes_t::simulation)
      .export_values();

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

  nanobind::class_<origami::dim4_t>(m, "dim4_t")
      .def(nanobind::init<std::size_t, std::size_t, std::size_t, std::size_t>())
      .def_rw("k", &origami::dim4_t::k)
      .def_rw("m", &origami::dim4_t::m)
      .def_rw("n", &origami::dim4_t::n)
      .def_rw("b", &origami::dim4_t::b)
      .def("mn", &origami::dim4_t::mn)
      .def("mnk", &origami::dim4_t::mnk)
      .def("total", &origami::dim4_t::total);

  // Tensile-specific parameters (used when prediction_mode == simulation)
  nanobind::class_<origami::tensile_params_t>(m, "tensile_params_t")
      .def(nanobind::init<>())
      .def_rw("depth_u", &origami::tensile_params_t::depth_u)
      .def_rw("global_split_u", &origami::tensile_params_t::global_split_u)
      .def_rw("global_accumulation", &origami::tensile_params_t::global_accumulation)
      .def_rw("local_split_u", &origami::tensile_params_t::local_split_u)
      .def_rw("direct_to_vgpr_a", &origami::tensile_params_t::direct_to_vgpr_a)
      .def_rw("direct_to_vgpr_b", &origami::tensile_params_t::direct_to_vgpr_b)
      .def_rw("direct_to_lds_a", &origami::tensile_params_t::direct_to_lds_a)
      .def_rw("direct_to_lds_b", &origami::tensile_params_t::direct_to_lds_b)
      .def_rw("num_loads_coalesced_a", &origami::tensile_params_t::num_loads_coalesced_a)
      .def_rw("num_loads_coalesced_b", &origami::tensile_params_t::num_loads_coalesced_b)
      .def_rw("wave_num", &origami::tensile_params_t::wave_num)
      .def_rw("wave_group_m", &origami::tensile_params_t::wave_group_m)
      .def_rw("wave_group_n", &origami::tensile_params_t::wave_group_n)
      .def_rw("prefetch_global_read", &origami::tensile_params_t::prefetch_global_read)
      .def_rw("math_clocks_unrolled_loop", &origami::tensile_params_t::math_clocks_unrolled_loop)
      .def_rw("swizzle_a", &origami::tensile_params_t::swizzle_a)
      .def_rw("swizzle_b", &origami::tensile_params_t::swizzle_b)
      .def_rw("workgroup_mapping_xcc", &origami::tensile_params_t::workgroup_mapping_xcc)
      .def_rw("workgroup_mapping_xcc_group",
              &origami::tensile_params_t::workgroup_mapping_xcc_group)
      .def_rw("global_split_u_coalesced", &origami::tensile_params_t::global_split_u_coalesced)
      .def_rw("global_split_u_wgm_round_robin",
              &origami::tensile_params_t::global_split_u_wgm_round_robin);

  nanobind::class_<origami::config_t>(m, "config_t")
      .def(nanobind::init<>())
      .def_rw("mt", &origami::config_t::mt)
      .def_rw("mi", &origami::config_t::mi)
      .def_rw("hand_optimized_main_loop", &origami::config_t::hand_optimized_main_loop)
      .def_rw("occupancy", &origami::config_t::occupancy)
      .def_rw("workgroup_mapping", &origami::config_t::workgroup_mapping)
      .def_rw("cache_hints_a", &origami::config_t::cache_hints_a)
      .def_rw("cache_hints_b", &origami::config_t::cache_hints_b)
      .def_rw("workspace_size", &origami::config_t::workspace_size)
      .def_rw("workspace_size_per_elem_c", &origami::config_t::workspace_size_per_elem_c)
      .def_rw("reduction_strategy", &origami::config_t::reduction_strategy)
      .def_rw("grid_selection", &origami::config_t::grid_selection)
      .def_rw("prediction_mode", &origami::config_t::prediction_mode)
      .def_rw("grvw_a", &origami::config_t::grvw_a)
      .def_rw("grvw_b", &origami::config_t::grvw_b)
      .def_rw("gwvw_d", &origami::config_t::gwvw_d)
      .def_rw("vector_width_a", &origami::config_t::vector_width_a)
      .def_rw("vector_width_b", &origami::config_t::vector_width_b)
      // Tensile-specific parameters accessed via variant backend
      .def("tensile",
           static_cast<origami::tensile_params_t& (origami::config_t::*)()>(
               &origami::config_t::tensile),
           nanobind::rv_policy::reference_internal,
           "Get mutable reference to Tensile params (initializes if not set)")
      .def("has_tensile_params",
           &origami::config_t::has_tensile_params,
           "Check if Tensile params are currently set")
      .def(
          "set_tensile_params",
          [](origami::config_t& c, const origami::tensile_params_t& p) { c.backend = p; },
          "Set Tensile params from a tensile_params_t object");

  nanobind::class_<origami::workgroup_mapping_t>(m, "workgroup_mapping_t")
      .def(nanobind::init<>())
      .def_rw("wgmxccchunk", &origami::workgroup_mapping_t::wgmxccchunk)
      .def_rw("wgmxcc", &origami::workgroup_mapping_t::wgmxcc)
      .def_rw("wgm", &origami::workgroup_mapping_t::wgm);

  nanobind::class_<origami::prediction_result_t>(m, "prediction_result_t")
      .def(nanobind::init<>())
      .def_rw("latency", &origami::prediction_result_t::latency)
      .def_rw("config", &origami::prediction_result_t::config);

  nanobind::class_<origami::context_t>(m, "context_t")
      .def(nanobind::init<>())
      .def(nanobind::init<const origami::problem_t&,
                          const origami::hardware_t&,
                          const origami::config_t&>())
      .def_rw("grid_m", &origami::context_t::grid_m)
      .def_rw("grid_n", &origami::context_t::grid_n)
      .def_rw("num_output_tiles", &origami::context_t::num_output_tiles)
      .def_rw("reduction_strategy", &origami::context_t::reduction_strategy)
      .def_rw("splitting_factor", &origami::context_t::splitting_factor)
      .def_rw("num_wgs", &origami::context_t::num_wgs)
      .def_rw("num_timesteps", &origami::context_t::num_timesteps)
      .def_rw("active_cus", &origami::context_t::active_cus)
      .def_rw("mem_bw_limited", &origami::context_t::mem_bw_limited)
      .def_rw("write_mem_bw_limited", &origami::context_t::write_mem_bw_limited)
      .def_rw("tile_elements", &origami::context_t::tile_elements)
      .def_rw("output_tile_bytes", &origami::context_t::output_tile_bytes)
      .def_rw("wgm", &origami::context_t::wgm);

  nanobind::class_<origami::problem_t>(m, "problem_t")
      .def(nanobind::init<>())
      .def_rw("size", &origami::problem_t::size)
      .def_rw("batch", &origami::problem_t::batch)
      .def_rw("a_transpose", &origami::problem_t::a_transpose)
      .def_rw("b_transpose", &origami::problem_t::b_transpose)
      .def_rw("a_dtype", &origami::problem_t::a_dtype)
      .def_rw("b_dtype", &origami::problem_t::b_dtype)
      .def_rw("c_dtype", &origami::problem_t::c_dtype)
      .def_rw("d_dtype", &origami::problem_t::d_dtype)
      .def_rw("mi_dtype", &origami::problem_t::mi_dtype)
      .def_rw("a_mx_block_size", &origami::problem_t::a_mx_block_size)
      .def_rw("b_mx_block_size", &origami::problem_t::b_mx_block_size);

  nanobind::class_<origami::staggerU_t>(m, "staggerU_t")
      .def(nanobind::init<>())
      .def_rw("staggerUMapping", &origami::staggerU_t::staggerUMapping)
      .def_rw("staggerU", &origami::staggerU_t::staggerU)
      .def_rw("staggerUStrideShift", &origami::staggerU_t::staggerUStrideShift);

  nanobind::class_<hardware_t>(m, "hardware_t")
      .def(nanobind::init<hardware_t::architecture_t,
                          size_t,                                 // N_CU
                          size_t,                                 // lds_capacity
                          size_t,                                 // NUM_XCD
                          double,                                 // mem1_perf_ratio
                          double,                                 // mem2_perf_ratio
                          double,                                 // mem3_perf_ratio
                          size_t,                                 // L2_capacity
                          double,                                 // compute_clock_ghz
                          size_t,                                 // parallel_mi_cu
                          std::tuple<double, double, double>>())  // mem_bw_per_wg_coefficients
      .def("print", &hardware_t::print)
      .def("get_valid_matrix_instructions",
           &hardware_t::get_valid_matrix_instructions,
           "Get valid matrix instruction dimensions for a given datatype")
      .def("get_recommended_matrix_instruction",
           &hardware_t::get_recommended_matrix_instruction,
           "Get recommended matrix instruction dimension (highest throughput) for a given datatype")
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
      .def_rw("NUM_XCD", &hardware_t::NUM_XCD);

  m.def("get_hardware_for_device",
        static_cast<hardware_t (*)(int)>(&hardware_t::get_hardware_for_device),
        "This gets a hardware object for a device.");

  // Needs named arguments
  m.def("get_hardware_for_arch",
        &hardware_t::get_hardware_for_arch,
        nanobind::arg("arch"),
        nanobind::arg("N_CU"),
        nanobind::arg("lds_capacity"),
        nanobind::arg("L2_capacity"),
        nanobind::arg("compute_clock_khz"),
        "Create hardware object for a specific architecture with specified parameters.");
  m.def("datatype_to_bits", &origami::datatype_to_bits, "Return the number of bits in a datatype");
  m.def("string_to_datatype",
        &origami::string_to_datatype,
        "Convert a string representation of a datatype into data_type_t enum");
  m.def("datatype_to_string",
        &origami::datatype_to_string,
        "Convert data_type_t enum to string representation");

  // Origami functions [origami.cpp]
  m.def("select_config",
        &origami::select_config,
        "Select best configuration based on problem and hardware");
  m.def("select_workgroup_mapping",
        &origami::select_workgroup_mapping,
        "Select best workgroup mapping");
  m.def("select_staggerU", &origami::select_staggerU, "Select best staggerU parameters");
  m.def("rank_configs", &origami::rank_configs, "Rank configurations by performance");
  m.def("select_config_mnk",
        &origami::select_config_mnk,
        "Select best configuration for M,N,K dimensions");
  m.def("select_topk_configs", &origami::select_topk_configs, "Select topk configurations");
  m.def("compute_perf_gflops", &origami::compute_perf_gflops, "Compute performance in GFLOPS");

  // StreamK functions [streamk.cpp]
  m.def("compute_number_of_output_tiles",
        &origami::streamk::compute_number_of_output_tiles,
        "Compute number of output tiles");
  m.def("select_reduction",
        &origami::streamk::select_reduction,
        "Select best StreamK reduction strategy");
  m.def("select_grid_size",
        &origami::streamk::select_grid_size,
        "Select best grid size for the given configuration");

  // GEMM functions [gemm.cpp] — ordered to match gemm.cpp implementation
  m.def("calculate_work_utilization",
        &origami::calculate_work_utilization,
        "Calculate the work utilization ratio");
  m.def("calculate_output_utilization",
        &origami::calculate_output_utilization,
        "Calculate the output utilization ratio");
  m.def("round_elements_to_128B",
        &origami::round_elements_to_128B,
        "Round elements to 128B alignment");
  m.def("predict_workgroup_mapping",
        &origami::predict_workgroup_mapping,
        "Fast WGM prediction based on last-XCD L2 cost minimization");
  m.def("compute_launch_parameters",
        &origami::compute_launch_parameters,
        "Compute launch parameters for the kernel");
  m.def("check_lds_capacity", &origami::check_lds_capacity, "Check if MT fits in LDS");
  m.def("compute_mem_bw_from_occupancy",
        &origami::compute_mem_bw_from_occupancy,
        "Compute limited achievable memory bandwidth based on active CUs");
  m.def("compute_mall_tiles", &origami::compute_mall_tiles, "Compute MALL tile dimensions");
  m.def("compute_l2_tiles", &origami::compute_l2_tiles, "Compute L2 tile dimensions");
  m.def("wgm_to_grid",
        &origami::wgm_to_grid,
        "Map a linear WG ID to 4D tile coordinates (k, m, n, b)");
  m.def("count_unique_tiles",
        &origami::count_unique_tiles,
        "Count unique tiles for a specific XCD during a specific timestep");
  m.def("count_unique_tiles_timestep",
        &origami::count_unique_tiles_timestep,
        "Count unique tiles for an entire timestep (all XCDs combined)");
  m.def("estimate_cache_hit_rates",
        &origami::estimate_cache_hit_rates,
        "Estimate MALL and L2 hit rates using two-timestep analytical model");
  m.def("compute_number_matrix_instructions",
        &origami::compute_number_matrix_instructions,
        "Compute the number of matrix instructions required");
  m.def("arithmetic_intensity", &origami::arithmetic_intensity, "Compute arithmetic intensity");
  m.def("emulated_tf32_arithmetic_intensity",
        &origami::emulated_tf32_arithmetic_intensity,
        "Compute emulated TF32 arithmetic intensity");
  m.def("compute_cvt_overhead_x1",
        &origami::compute_cvt_overhead_x1,
        "Compute TF32 X1 conversion overhead");
  m.def("compute_cvt_overhead",
        &origami::compute_cvt_overhead,
        "Compute TF32 X3 conversion overhead");
  m.def("compute_mt_compute_latency",
        &origami::compute_mt_compute_latency,
        "Compute the latency to process a single macro-tile");
  m.def("estimate_l2_hit", &origami::estimate_l2_hit, "Estimate L2 hit rate");
  m.def("estimate_mall_hit", &origami::estimate_mall_hit, "Estimate MALL hit rate");
  m.def("compute_l2_hit_rate_global",
        &origami::compute_l2_hit_rate_global,
        "Compute L2 hit rate from a global perspective");
  m.def("compute_memory_latency",
        &origami::compute_memory_latency,
        "Compute memory latency per macro tile");
  m.def("compute_tile_latency",
        &origami::compute_tile_latency,
        "Compute latency to compute a K-complete tile");
  m.def("compute_timestep_latency",
        &origami::compute_timestep_latency,
        "Compute latency per K-complete MT wave");
  m.def("compute_total_latency", &origami::compute_total_latency, "Compute total latency");
  m.def("compute_total_latency",
        static_cast<double (*)(const origami::problem_t&,
                               const origami::hardware_t&,
                               const origami::config_t&,
                               size_t max_cus)>(&origami::compute_total_latency),
        "Compute total latency (uses Formocast when config.prediction_mode == simulation)");

  // Lambda wrappers (auto-create context_t from problem/hardware/config)
  m.def(
      "estimate_l2_hit",
      [](const origami::problem_t& problem,
         const origami::hardware_t& hardware,
         const origami::config_t& config) {
        origami::context_t context(problem, hardware, config);
        return origami::estimate_l2_hit(problem, hardware, config, context);
      },
      "Estimate L2 hit rate (auto-creates context)");
  m.def(
      "estimate_mall_hit",
      [](const origami::problem_t& problem,
         const origami::hardware_t& hardware,
         const origami::config_t& config) {
        origami::context_t context(problem, hardware, config);
        return origami::estimate_mall_hit(problem, hardware, config, context);
      },
      "Estimate MALL hit rate (auto-creates context)");
  m.def(
      "estimate_cache_hit_rates",
      [](const origami::problem_t& problem,
         const origami::hardware_t& hardware,
         const origami::config_t& config) {
        origami::context_t context(problem, hardware, config);
        return origami::estimate_cache_hit_rates(problem, hardware, config, context);
      },
      "Estimate per-operand cache hit rates as "
      "(H_mem_l1_A, H_mem_l1_B, H_mem_l2_A, H_mem_l2_B, H_mem_mall_A, H_mem_mall_B) "
      "using the analytical model (auto-creates context)");
  m.def(
      "compute_memory_latency",
      [](const origami::problem_t& problem,
         const origami::hardware_t& hardware,
         const origami::config_t& config) {
        origami::context_t context(problem, hardware, config);
        return origami::compute_memory_latency(problem, hardware, config, context);
      },
      "Compute memory latency per macro tile (auto-creates context)");
  m.def(
      "compute_tile_latency",
      [](const origami::problem_t& problem,
         const origami::hardware_t& hardware,
         const origami::config_t& config) {
        origami::context_t context(problem, hardware, config);
        return origami::compute_tile_latency(problem, hardware, config, context);
      },
      "Compute latency to compute a K-complete tile (auto-creates context)");
  m.def(
      "compute_timestep_latency",
      [](const origami::problem_t& problem,
         const origami::hardware_t& hardware,
         const origami::config_t& config) {
        origami::context_t context(problem, hardware, config);
        return origami::compute_timestep_latency(problem, hardware, config, context);
      },
      "Compute latency per K-complete MT wave (auto-creates context)");
}
