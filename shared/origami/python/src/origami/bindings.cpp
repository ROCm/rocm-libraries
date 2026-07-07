// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include "origami/attention.hpp"
#include "origami/comm/hardware_device.hpp"
#include "origami/comm/origami_comm.hpp"
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

  nanobind::enum_<origami::model_t>(m, "model_t")
      .value("gemm", origami::model_t::gemm)
      .value("attention", origami::model_t::attention)
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
      .def_rw("subtile", &origami::config_t::subtile)
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

  nanobind::class_<origami::gemm::context_t>(m, "context_t")
      .def(nanobind::init<>())
      .def(nanobind::init<const origami::problem_t&,
                          const origami::hardware_t&,
                          const origami::config_t&>())
      .def_rw("grid_m", &origami::gemm::context_t::grid_m)
      .def_rw("grid_n", &origami::gemm::context_t::grid_n)
      .def_rw("num_output_tiles", &origami::gemm::context_t::num_output_tiles)
      .def_rw("reduction_strategy", &origami::gemm::context_t::reduction_strategy)
      .def_rw("splitting_factor", &origami::gemm::context_t::splitting_factor)
      .def_rw("num_wgs", &origami::gemm::context_t::num_wgs)
      .def_rw("num_timesteps", &origami::gemm::context_t::num_timesteps)
      .def_rw("active_cus", &origami::gemm::context_t::active_cus)
      .def_rw("mem_bw_limited", &origami::gemm::context_t::mem_bw_limited)
      .def_rw("write_mem_bw_limited", &origami::gemm::context_t::write_mem_bw_limited)
      .def_rw("tile_elements", &origami::gemm::context_t::tile_elements)
      .def_rw("output_tile_bytes", &origami::gemm::context_t::output_tile_bytes)
      .def_rw("wgm", &origami::gemm::context_t::wgm);

  nanobind::class_<origami::problem_t>(m, "problem_t")
      .def(nanobind::init<>())
      .def_rw("size", &origami::problem_t::size)
      .def_rw("batch", &origami::problem_t::batch)
      .def_rw("q_heads", &origami::problem_t::q_heads)
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
                          size_t,                                 // rf_capacity
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
      .def_rw("rf_capacity", &hardware_t::rf_capacity)
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
        nanobind::arg("rf_capacity"),
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
        &origami::gemm::calculate_work_utilization,
        "Calculate the work utilization ratio");
  m.def("calculate_output_utilization",
        &origami::gemm::calculate_output_utilization,
        "Calculate the output utilization ratio");
  m.def("round_elements_to_128B",
        &origami::gemm::round_elements_to_128B,
        "Round elements to 128B alignment");
  m.def("predict_workgroup_mapping",
        &origami::gemm::predict_workgroup_mapping,
        "Fast WGM prediction based on last-XCD L2 cost minimization");
  m.def("compute_launch_parameters",
        &origami::gemm::compute_launch_parameters,
        "Compute launch parameters for the kernel");
  m.def("check_lds_capacity", &origami::gemm::check_lds_capacity, "Check if MT fits in LDS");
  m.def("compute_mem_bw_from_occupancy",
        &origami::gemm::compute_mem_bw_from_occupancy,
        "Compute limited achievable memory bandwidth based on active CUs");
  m.def("compute_mall_tiles", &origami::gemm::compute_mall_tiles, "Compute MALL tile dimensions");
  m.def("compute_l2_tiles", &origami::gemm::compute_l2_tiles, "Compute L2 tile dimensions");
  m.def("wgm_to_grid",
        &origami::gemm::wgm_to_grid,
        "Map a linear WG ID to 4D tile coordinates (k, m, n, b)");
  m.def("count_unique_tiles",
        &origami::gemm::count_unique_tiles,
        "Count unique tiles for a specific XCD during a specific timestep");
  m.def("count_unique_tiles_timestep",
        &origami::gemm::count_unique_tiles_timestep,
        "Count unique tiles for an entire timestep (all XCDs combined)");
  m.def("estimate_cache_hit_rates",
        &origami::gemm::estimate_cache_hit_rates,
        "Estimate MALL and L2 hit rates using two-timestep analytical model");
  m.def("compute_number_matrix_instructions",
        &origami::gemm::compute_number_matrix_instructions,
        "Compute the number of matrix instructions required");
  m.def("arithmetic_intensity", &origami::gemm::arithmetic_intensity, "Compute arithmetic intensity");
  m.def("emulated_tf32_arithmetic_intensity",
        &origami::gemm::emulated_tf32_arithmetic_intensity,
        "Compute emulated TF32 arithmetic intensity");
  m.def("compute_cvt_overhead_x1",
        &origami::gemm::compute_cvt_overhead_x1,
        "Compute TF32 X1 conversion overhead");
  m.def("compute_cvt_overhead",
        &origami::gemm::compute_cvt_overhead,
        "Compute TF32 X3 conversion overhead");
  m.def("compute_mt_compute_latency",
        &origami::gemm::compute_mt_compute_latency,
        "Compute the latency to process a single macro-tile");
  m.def("estimate_l2_hit", &origami::gemm::estimate_l2_hit, "Estimate L2 hit rate");
  m.def("estimate_mall_hit", &origami::gemm::estimate_mall_hit, "Estimate MALL hit rate");
  m.def("compute_l2_hit_rate_global",
        &origami::gemm::compute_l2_hit_rate_global,
        "Compute L2 hit rate from a global perspective");
  m.def("compute_memory_latency",
        &origami::gemm::compute_memory_latency,
        "Compute memory latency per macro tile");
  m.def("compute_tile_latency",
        &origami::gemm::compute_tile_latency,
        "Compute latency to compute a K-complete tile");
  m.def("compute_timestep_latency",
        &origami::gemm::compute_timestep_latency,
        "Compute latency per K-complete MT wave");
  m.def("compute_total_latency", &origami::gemm::compute_total_latency, "Compute total latency");
  m.def("compute_total_latency",
        static_cast<double (*)(const origami::problem_t&,
                               const origami::hardware_t&,
                               const origami::config_t&,
                               size_t max_cus)>(&origami::gemm::compute_total_latency),
        "Compute total latency (uses Formocast when config.prediction_mode == simulation)");

  // Attention functions
  m.def("att_compute_total_latency",
        static_cast<double (*)(const origami::problem_t&,
                               const origami::hardware_t&,
                               const origami::config_t&,
                               size_t max_cus)>(&origami::attention::compute_total_latency),
        "Compute total latency for Flash Attention");
  m.def("att_compute_number_matrix_instructions",
        &origami::attention::compute_number_matrix_instructions,
        "Compute the number of matrix instructions required for attention");
  m.def("att_compute_mt_compute_latency",
        &origami::attention::compute_mt_compute_latency,
        "Compute the latency to process a single macro-tile for attention");
  m.def("att_check_lds_capacity",
        &origami::attention::check_lds_capacity,
        "Check if attention MT fits in LDS");
  m.def("att_estimate_l2_hit",
        &origami::attention::estimate_l2_hit,
        "Estimate L2 hit rate for attention");
  m.def("att_estimate_mall_hit",
        &origami::attention::estimate_mall_hit,
        "Estimate MALL hit rate for attention");
  m.def("att_compute_memory_latency",
        &origami::attention::compute_memory_latency,
        "Compute memory latency per macro tile for attention");
  m.def("att_compute_tile_latency",
        &origami::attention::compute_tile_latency,
        "Compute latency to compute a K-complete tile for attention");
  m.def("att_compute_timestep_latency",
        &origami::attention::compute_timestep_latency,
        "Compute latency per K-complete MT wave for attention");
  m.def("att_calculate_work_utilization",
        &origami::attention::calculate_work_utilization,
        "Calculate work utilization for attention");
  m.def("att_calculate_output_utilization",
        &origami::attention::calculate_output_utilization,
        "Calculate output utilization for attention");
  m.def("att_compute_cu_occupancy",
        &origami::attention::compute_cu_occupancy,
        "Compute CU occupancy for attention");
  m.def("att_arithmetic_intensity",
        &origami::attention::arithmetic_intensity,
        "Compute arithmetic intensity for attention");
  m.def("att_emulated_tf32_arithmetic_intensity",
        &origami::attention::emulated_tf32_arithmetic_intensity,
        "Compute emulated TF32 arithmetic intensity for attention");
  m.def("att_round_elements_to_128B",
        &origami::attention::round_elements_to_128B,
        "Round elements to 128B boundary for attention");
  m.def("att_compute_mem_bw_from_occupancy",
        &origami::attention::compute_mem_bw_from_occupancy,
        "Compute memory bandwidth from occupancy for attention");
  m.def("att_compute_l2_hit_rate_global",
        &origami::attention::compute_l2_hit_rate_global,
        "Compute global L2 hit rate for attention");

  // Lambda wrappers (auto-create context_t from problem/hardware/config)
  m.def(
      "estimate_l2_hit",
      [](const origami::problem_t& problem,
         const origami::hardware_t& hardware,
         const origami::config_t& config) {
        origami::gemm::context_t context(problem, hardware, config);
        return origami::gemm::estimate_l2_hit(problem, hardware, config, context);
      },
      "Estimate L2 hit rate (auto-creates context)");
  m.def(
      "estimate_mall_hit",
      [](const origami::problem_t& problem,
         const origami::hardware_t& hardware,
         const origami::config_t& config) {
        origami::gemm::context_t context(problem, hardware, config);
        return origami::gemm::estimate_mall_hit(problem, hardware, config, context);
      },
      "Estimate MALL hit rate (auto-creates context)");
  m.def(
      "estimate_cache_hit_rates",
      [](const origami::problem_t& problem,
         const origami::hardware_t& hardware,
         const origami::config_t& config) {
        origami::gemm::context_t context(problem, hardware, config);
        return origami::gemm::estimate_cache_hit_rates(problem, hardware, config, context);
      },
      "Estimate per-operand cache hit rates as "
      "(H_mem_l1_A, H_mem_l1_B, H_mem_l2_A, H_mem_l2_B, H_mem_mall_A, H_mem_mall_B) "
      "using the analytical model (auto-creates context)");
  m.def(
      "compute_memory_latency",
      [](const origami::problem_t& problem,
         const origami::hardware_t& hardware,
         const origami::config_t& config) {
        origami::gemm::context_t context(problem, hardware, config);
        return origami::gemm::compute_memory_latency(problem, hardware, config, context);
      },
      "Compute memory latency per macro tile (auto-creates context)");
  m.def(
      "compute_tile_latency",
      [](const origami::problem_t& problem,
         const origami::hardware_t& hardware,
         const origami::config_t& config) {
        origami::gemm::context_t context(problem, hardware, config);
        return origami::gemm::compute_tile_latency(problem, hardware, config, context);
      },
      "Compute latency to compute a K-complete tile (auto-creates context)");
  m.def(
      "compute_timestep_latency",
      [](const origami::problem_t& problem,
         const origami::hardware_t& hardware,
         const origami::config_t& config) {
        origami::gemm::context_t context(problem, hardware, config);
        return origami::gemm::compute_timestep_latency(problem, hardware, config, context);
      },
      "Compute latency per K-complete MT wave (auto-creates context)");

  // ───────────────────────────────────────────────────────────────────────
  // origami.comm — analytical communication (collective) cost model.
  //
  // Mirrors origami::comm's public API: the byte-level entry point
  // (predict_row), the tensor-level entry point (predict_tensor_collective),
  // and the lower-level compute_collective_latency for codesign studies that
  // sweep algorithms. Hardware/heuristics are exposed as the MI300X defaults so
  // callers can predict without constructing a system, yet tweak them when
  // they need to. The string-typed predict overloads are bound (op/dtype/
  // framework as plain Python strings) because that is the ergonomic edge.
  // ───────────────────────────────────────────────────────────────────────
  namespace oc = origami::comm;
  auto comm    = m.def_submodule("comm", "Analytical communication (collective) cost model.");

  nanobind::enum_<oc::primitive_t>(
      comm, "primitive_t", "The collective OPERATION (a problem property).")
      .value("all_gather", oc::primitive_t::all_gather)
      .value("reduce_scatter", oc::primitive_t::reduce_scatter)
      .value("broadcast", oc::primitive_t::broadcast)
      .value("all_reduce", oc::primitive_t::all_reduce)
      .value("all_to_all", oc::primitive_t::all_to_all)
      .export_values();

  nanobind::enum_<oc::algorithm_t>(
      comm, "algorithm_t", "The collective IMPLEMENTATION (a config/perf choice).")
      .value("automatic", oc::algorithm_t::automatic)
      .value("ring", oc::algorithm_t::ring)
      .value("one_shot", oc::algorithm_t::one_shot)
      .value("two_shot", oc::algorithm_t::two_shot)
      .value("direct", oc::algorithm_t::direct)
      .export_values();

  nanobind::enum_<oc::framework_t>(
      comm, "framework_t", "Caller software stack (sets the host-overhead floor).")
      .value("raw", oc::framework_t::raw)
      .value("rccl", oc::framework_t::rccl)
      .value("nccl", oc::framework_t::nccl)
      .value("torch", oc::framework_t::torch)
      .value("jax", oc::framework_t::jax)
      .value("mpi", oc::framework_t::mpi)
      .export_values();

  nanobind::enum_<oc::load_width_t>(
      comm, "load_width_t", "Bytes moved per VMEM instruction (the enum value is the width).")
      .value("DWORD", oc::load_width_t::DWORD)
      .value("DWORDX4", oc::load_width_t::DWORDX4)
      .value("DWORDX16", oc::load_width_t::DWORDX16)
      .export_values();

  // ── hardware_t: per-GPU compute/memory ceilings ──────────────────
  nanobind::class_<oc::hardware_t>(comm, "hardware_t", "Per-GPU compute and memory ceilings.")
      .def(nanobind::init<>())
      .def_prop_ro(
          "arch",
          [](const oc::hardware_t& h) { return std::string{origami::arch_enum_to_name(h.arch)}; })
      .def_rw("num_cu", &oc::hardware_t::num_cu)
      .def_rw("num_xcd", &oc::hardware_t::num_xcd)
      .def_rw("cu_per_xcd", &oc::hardware_t::cu_per_xcd)
      .def_rw("clock_ghz", &oc::hardware_t::clock_ghz)
      .def_rw("vmem_issue_rate", &oc::hardware_t::vmem_issue_rate)
      .def_rw("valu_rate", &oc::hardware_t::valu_rate)
      .def_rw("tcp_bw", &oc::hardware_t::tcp_bw)
      .def_rw("mshr_depth_per_wave", &oc::hardware_t::mshr_depth_per_wave)
      .def_rw("waves_per_wg", &oc::hardware_t::waves_per_wg)
      .def_rw("xgmi_latency_cycles", &oc::hardware_t::xgmi_latency_cycles)
      .def_rw("l2_bw_per_cu", &oc::hardware_t::l2_bw_per_cu)
      .def_rw("mall_bw", &oc::hardware_t::mall_bw)
      .def_rw("hbm_read_bw", &oc::hardware_t::hbm_read_bw)
      .def_rw("hbm_write_bw", &oc::hardware_t::hbm_write_bw)
      .def("cycles_to_us", &oc::hardware_t::cycles_to_us, "cycles"_a)
      .def("cycles_to_ns", &oc::hardware_t::cycles_to_ns, "cycles"_a);

  // ── comm_hardware_t: the inter-GPU fabric ────────────────────────
  nanobind::class_<oc::comm_hardware_t>(comm, "comm_hardware_t", "Inter-GPU xGMI fabric ceilings.")
      .def(nanobind::init<>())
      .def_rw("link_bw", &oc::comm_hardware_t::link_bw)
      .def_rw("num_peer_links", &oc::comm_hardware_t::num_peer_links)
      .def_rw("num_sdma_engines", &oc::comm_hardware_t::num_sdma_engines)
      .def_rw("sdma_read_bw", &oc::comm_hardware_t::sdma_read_bw)
      .def_rw("sdma_write_bw", &oc::comm_hardware_t::sdma_write_bw)
      .def_rw("atomic_latency_cycles", &oc::comm_hardware_t::atomic_latency_cycles)
      .def_rw("launch_overhead_cycles", &oc::comm_hardware_t::launch_overhead_cycles)
      .def_rw("clock_ghz", &oc::comm_hardware_t::clock_ghz);

  // ── system_t: a GPU plus its fabric ──────────────────────────────
  nanobind::class_<oc::system_t>(
      comm, "system_t", "A GPU plus the fabric that joins it to its peers.")
      .def(nanobind::init<>())
      .def_rw("gpu", &oc::system_t::gpu)
      .def_rw("fabric", &oc::system_t::fabric);

  // ── heuristics_t: empirical fudge factors ────────────────────────
  // Only the scalar knobs are exposed (the *_ns / *_cycles fit tables stay in
  // C++); construct the default and flip these to run sensitivity studies.
  nanobind::class_<oc::heuristics_t>(comm, "heuristics_t", "Empirical calibration knobs.")
      .def(nanobind::init<>())
      .def_rw("min_bytes_per_wg", &oc::heuristics_t::min_bytes_per_wg)
      .def_rw("assume_rank_symmetry", &oc::heuristics_t::assume_rank_symmetry);

  // ── gpu_topology_t: live per-device shape ────────────────────────
  nanobind::class_<oc::gpu_topology_t>(
      comm, "gpu_topology_t", "Per-device GPU topology (CU/XCD counts, L2 capacity).")
      .def(
          "__init__",
          [](oc::gpu_topology_t* self,
             hardware_t::architecture_t arch,
             std::size_t num_cu,
             std::size_t num_xcd,
             std::size_t cu_per_xcd,
             std::size_t l2_capacity_bytes) {
            new (self) oc::gpu_topology_t{arch, num_cu, num_xcd, cu_per_xcd, l2_capacity_bytes};
          },
          "arch"_a,
          "num_cu"_a,
          "num_xcd"_a,
          "cu_per_xcd"_a,
          "l2_capacity_bytes"_a)
      .def_rw("arch", &oc::gpu_topology_t::arch)
      .def_rw("num_cu", &oc::gpu_topology_t::num_cu)
      .def_rw("num_xcd", &oc::gpu_topology_t::num_xcd)
      .def_rw("cu_per_xcd", &oc::gpu_topology_t::cu_per_xcd)
      .def_rw("l2_capacity_bytes", &oc::gpu_topology_t::l2_capacity_bytes);

  // ── arch_ceilings_t: calibrated per-architecture ceilings ────────
  // Opaque handle produced by get_arch_ceilings and consumed by make_system; the
  // native-unit fields are an implementation detail callers do not edit.
  nanobind::class_<oc::arch_ceilings_t>(
      comm, "arch_ceilings_t", "Calibrated per-architecture comm ceilings (native units).");

  comm.def("get_arch_ceilings",
           &oc::get_arch_ceilings,
           "arch"_a,
           "Calibrated communication ceilings for an architecture (raises if uncalibrated).");

  comm.def("make_system",
           &oc::make_system,
           "ceilings"_a,
           "topology"_a,
           "clock_ghz"_a,
           "Fuse calibrated ceilings, a topology, and a clock into a system_t.");

  // ── Live-device system factories (HIP-dependent) ─────────────────
  comm.def("system_from_hardware",
           &oc::system_from_hardware,
           "hardware"_a,
           "Build a comm system_t from an origami.hardware_t (same device, one topology).");

  comm.def("system_from_device",
           &oc::system_from_device,
           "device_id"_a,
           "Build a comm system_t by querying a live HIP device (picks up CPX partitioning).");

  // Default heuristics, bound as a module attribute (copy): pass straight
  // into the predict functions, or mutate a copy for what-if studies. There is
  // deliberately no hardcoded MI300X system attribute — build a system_t for the
  // device you are about to run on via system_from_device / system_from_hardware,
  // or from make_system with an explicit topology.
  comm.attr("DEFAULT_HEURISTICS") = oc::DEFAULT_HEURISTICS;

  // ── tile_shape_t: a 2D tile with a contiguity bit ────────────────
  nanobind::class_<oc::tile_shape_t>(
      comm, "tile_shape_t", "A 2D (m x n x dtype) tile with a contiguity bit.")
      .def(nanobind::init<>())
      .def_rw("m", &oc::tile_shape_t::m)
      .def_rw("n", &oc::tile_shape_t::n)
      .def_rw("dtype", &oc::tile_shape_t::dtype)
      .def_rw("split_dim", &oc::tile_shape_t::split_dim)
      .def_rw("contiguous", &oc::tile_shape_t::contiguous)
      .def("bytes", &oc::tile_shape_t::bytes)
      .def("elements", &oc::tile_shape_t::elements)
      .def("cachelines", &oc::tile_shape_t::cachelines);

  // ── comm_problem_t: the correctness inputs ───────────────────────
  nanobind::class_<oc::comm_problem_t>(
      comm, "comm_problem_t", "[M,N] tensor over num_gpus; the collective is a problem property.")
      .def(
          "__init__",
          [](oc::comm_problem_t* self,
             std::size_t M,
             std::size_t N,
             int num_gpus,
             origami::data_type_t dtype,
             int split_dim,
             oc::primitive_t collective) {
            new (self) oc::comm_problem_t{M, N, num_gpus, dtype, split_dim, collective};
          },
          "M"_a,
          "N"_a,
          "num_gpus"_a,
          "dtype"_a      = origami::data_type_t::BFloat16,
          "split_dim"_a  = 0,
          "collective"_a = oc::primitive_t::all_reduce)
      .def_rw("M", &oc::comm_problem_t::M)
      .def_rw("N", &oc::comm_problem_t::N)
      .def_rw("num_gpus", &oc::comm_problem_t::num_gpus)
      .def_rw("dtype", &oc::comm_problem_t::dtype)
      .def_rw("split_dim", &oc::comm_problem_t::split_dim)
      .def_rw("collective", &oc::comm_problem_t::collective)
      .def("message_bytes", &oc::comm_problem_t::message_bytes)
      .def("gpu_tile_bytes", &oc::comm_problem_t::gpu_tile_bytes);

  // ── comm_config_t: the performance inputs ────────────────────────
  nanobind::class_<oc::comm_config_t>(
      comm, "comm_config_t", "Workgroup-level execution config; carries the algorithm choice.")
      .def(
          "__init__",
          [](oc::comm_config_t* self,
             int num_wgs,
             oc::load_width_t load_width,
             int vgprs_for_data,
             int min_bytes_per_wg,
             oc::algorithm_t algorithm) {
            oc::comm_config_t cfg{};
            cfg.num_wgs          = num_wgs;
            cfg.load_width       = load_width;
            cfg.vgprs_for_data   = vgprs_for_data;
            cfg.min_bytes_per_wg = min_bytes_per_wg;
            cfg.algorithm        = algorithm;
            new (self) oc::comm_config_t{cfg};
          },
          "num_wgs"_a,
          "load_width"_a       = oc::load_width_t::DWORDX16,
          "vgprs_for_data"_a   = 128,
          "min_bytes_per_wg"_a = 16384,
          "algorithm"_a        = oc::algorithm_t::automatic)
      .def_rw("num_wgs", &oc::comm_config_t::num_wgs)
      .def_rw("load_width", &oc::comm_config_t::load_width)
      .def_rw("vgprs_for_data", &oc::comm_config_t::vgprs_for_data)
      .def_rw("min_bytes_per_wg", &oc::comm_config_t::min_bytes_per_wg)
      .def_rw("algorithm", &oc::comm_config_t::algorithm);

  // ── tensor_collective_prediction_t: the rich tensor-level result ─
  nanobind::class_<oc::tensor_collective_prediction_t>(
      comm, "tensor_collective_prediction_t", "Result of predict_tensor_collective.")
      .def_ro("predicted_us", &oc::tensor_collective_prediction_t::predicted_us)
      .def_ro("op", &oc::tensor_collective_prediction_t::op)
      .def_ro("input_shape", &oc::tensor_collective_prediction_t::input_shape)
      .def_ro("dim", &oc::tensor_collective_prediction_t::dim)
      .def_ro("world_size", &oc::tensor_collective_prediction_t::world_size)
      .def_ro("nchannels", &oc::tensor_collective_prediction_t::nchannels)
      .def_ro("dtype", &oc::tensor_collective_prediction_t::dtype)
      .def_ro("per_rank_bytes", &oc::tensor_collective_prediction_t::per_rank_bytes)
      .def_ro("wire_bytes_per_rank", &oc::tensor_collective_prediction_t::wire_bytes_per_rank)
      .def_ro("msg_bytes", &oc::tensor_collective_prediction_t::msg_bytes)
      .def_ro("gpu_tile", &oc::tensor_collective_prediction_t::gpu_tile)
      .def_ro("framework", &oc::tensor_collective_prediction_t::framework)
      .def_ro("framework_overhead_us", &oc::tensor_collective_prediction_t::framework_overhead_us)
      .def("backend_us", &oc::tensor_collective_prediction_t::backend_us);

  // ── predict_row: byte-level latency in microseconds ──────────────
  comm.def(
      "predict_row",
      [](const std::string& primitive,
         std::size_t msg_bytes,
         int world_size,
         int nchannels,
         const oc::system_t& system,
         std::size_t M,
         std::size_t N,
         int split_dim,
         const oc::heuristics_t& heur) {
        return oc::predict_row(
            primitive, msg_bytes, world_size, nchannels, system, M, N, split_dim, heur);
      },
      "primitive"_a,
      "msg_bytes"_a,
      "world_size"_a,
      "nchannels"_a,
      "system"_a,
      "M"_a         = 0,
      "N"_a         = 0,
      "split_dim"_a = 0,
      "heur"_a      = oc::DEFAULT_HEURISTICS,
      "Predict one collective call's latency in microseconds from a benchmark row.");

  // ── predict_tensor_collective: shape/dtype-level prediction ──────
  comm.def(
      "predict_tensor_collective",
      [](const std::string& op,
         const std::vector<std::size_t>& input_shape,
         const std::string& dtype,
         int world_size,
         const oc::system_t& system,
         int dim,
         int nchannels,
         const std::string& framework,
         const oc::heuristics_t& heur) {
        return oc::predict_tensor_collective(
            op, input_shape, dtype, world_size, system, dim, nchannels, framework, heur);
      },
      "op"_a,
      "input_shape"_a,
      "dtype"_a,
      "world_size"_a,
      "system"_a,
      "dim"_a       = 0,
      "nchannels"_a = 32,
      "framework"_a = "raw",
      "heur"_a      = oc::DEFAULT_HEURISTICS,
      "Predict a collective's latency (microseconds) from a per-rank tensor shape.");

  // ── compute_collective_latency: GPU cycles, for algorithm sweeps ─
  comm.def(
      "compute_collective_latency",
      [](const oc::comm_problem_t& problem,
         const oc::comm_config_t& config,
         const oc::system_t& system,
         const oc::heuristics_t& heur) {
        return oc::compute_collective_latency(problem, config, system, heur);
      },
      "problem"_a,
      "config"_a,
      "system"_a,
      "heur"_a = oc::DEFAULT_HEURISTICS,
      "Predicted GPU cycles for the whole collective (max over ranks).");
}
