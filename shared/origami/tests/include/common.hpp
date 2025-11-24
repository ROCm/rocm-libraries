// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once
#include <vector>
#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/origami.hpp"
#include "origami/streamk.hpp"

// List of GPU architectures to test
inline const std::vector<int> test_architectures = {942, 950};

// Helper function to construct problem_t
inline origami::problem_t make_problem(size_t m,
                                       size_t n,
                                       size_t k,
                                       origami::transpose_t a_trans = origami::transpose_t::T,
                                       origami::transpose_t b_trans = origami::transpose_t::N,
                                       size_t batch                 = 1,
                                       int mx_block_size            = 0) {
  origami::problem_t problem;
  problem.size.m          = m;
  problem.size.n          = n;
  problem.size.k          = k;
  problem.batch           = batch;
  problem.a_transpose     = a_trans;
  problem.b_transpose     = b_trans;
  problem.a_dtype         = origami::data_type_t::BFloat16;
  problem.b_dtype         = origami::data_type_t::BFloat16;
  problem.c_dtype         = origami::data_type_t::BFloat16;
  problem.d_dtype         = origami::data_type_t::BFloat16;
  problem.mi_dtype        = origami::data_type_t::BFloat16;
  problem.a_mx_block_size = mx_block_size;
  problem.b_mx_block_size = mx_block_size;
  return problem;
}

// Helper function to construct config_t
inline origami::config_t make_config(size_t mt_m,
                                     size_t mt_n,
                                     size_t mt_k,
                                     size_t mi_m   = 16,
                                     size_t mi_n   = 16,
                                     size_t mi_k   = 16,
                                     int wgm       = 1,
                                     int occupancy = 1,
                                     int non_temporal_a = 0,
                                     int non_temporal_b = 0) {
  origami::config_t config;
  config.mt.m              = mt_m;
  config.mt.n              = mt_n;
  config.mt.k              = mt_k;
  config.mi.m              = mi_m;
  config.mi.n              = mi_n;
  config.mi.k              = mi_k;
  config.occupancy         = occupancy;
  config.workgroup_mapping = wgm;
  config.cache_hints_a    = non_temporal_a;
  config.cache_hints_b    = non_temporal_b;
  return config;
}

// Helper function to construct hardware_t with all parameters
inline origami::hardware_t make_hardware(
    int gpu_arch,
    size_t n_cu                                                   = 304,
    size_t lds_capacity                                           = 65536,
    size_t num_xcd                                                = 8,
    double mem1_perf_ratio                                        = 1.0,
    double mem2_perf_ratio                                        = 1.0,
    double mem3_perf_ratio                                        = 1.0,
    size_t l2_capacity                                            = 4000000,
    double compute_clock_ghz                                      = 1.0,
    size_t parallel_mi_cu                                         = 1,
    std::tuple<double, double, double> mem_bw_per_wg_coefficients = std::make_tuple(0, 0.015, 0)) {
  const std::string gpu_arch_str = "gfx" + std::to_string(gpu_arch);
  auto gpu_arch_enum             = origami::hardware_t::arch_name_to_enum(gpu_arch_str);

  return origami::hardware_t(gpu_arch_enum,
                             n_cu,
                             lds_capacity,
                             num_xcd,
                             mem1_perf_ratio,
                             mem2_perf_ratio,
                             mem3_perf_ratio,
                             l2_capacity,
                             compute_clock_ghz,
                             parallel_mi_cu,
                             mem_bw_per_wg_coefficients);
}