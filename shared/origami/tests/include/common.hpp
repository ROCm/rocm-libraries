/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once
#include <cstdlib>
#include <vector>
#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/origami.hpp"
#include "origami/streamk.hpp"

// Uses _putenv_s on Windows, setenv on POSIX
inline int portable_setenv(const char* name, const char* value, int overwrite) {
#ifdef _WIN32
  (void)overwrite;
  return _putenv_s(name, value);
#else
  return setenv(name, value, overwrite);
#endif
}

inline int portable_unsetenv(const char* name) {
#ifdef _WIN32
  return _putenv_s(name, "");
#else
  return unsetenv(name);
#endif
}

// List of GPU architectures to test
inline const std::vector<int> test_architectures = {942, 950, 1250};

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
                                     size_t mi_m                   = 16,
                                     size_t mi_n                   = 16,
                                     size_t mi_k                   = 16,
                                     bool hand_optimized_main_loop = false,
                                     int wgm                       = 1,
                                     int occupancy                 = 1,
                                     int non_temporal_a            = 0,
                                     int non_temporal_b            = 0) {
  origami::config_t config;
  config.mt.m                     = mt_m;
  config.mt.n                     = mt_n;
  config.mt.k                     = mt_k;
  config.mi.m                     = mi_m;
  config.mi.n                     = mi_n;
  config.mi.k                     = mi_k;
  config.hand_optimized_main_loop = hand_optimized_main_loop;
  config.occupancy                = occupancy;
  config.workgroup_mapping        = wgm;
  config.cache_hints_a            = non_temporal_a;
  config.cache_hints_b            = non_temporal_b;
  return config;
}

// Helper function to construct hardware_t with all parameters
inline origami::hardware_t make_hardware(int gpu_arch) {
  // Initialize the constants
  size_t n_cu                                                   = 0;
  size_t lds_capacity                                           = 0;
  size_t num_xcd                                                = 0;
  double mem1_perf_ratio                                        = 0.0;
  double mem2_perf_ratio                                        = 0.0;
  double mem3_perf_ratio                                        = 0.0;
  size_t l2_capacity                                            = 0;
  double compute_clock_ghz                                      = 0.0;
  size_t parallel_mi_cu                                         = 0;
  std::tuple<double, double, double> mem_bw_per_wg_coefficients = std::make_tuple(0, 0, 0);

  if (gpu_arch == 942) {
    n_cu                       = 304;
    lds_capacity               = 65536;
    num_xcd                    = 8;
    mem1_perf_ratio            = 1.0;
    mem2_perf_ratio            = 1.0;
    mem3_perf_ratio            = 1.0;
    l2_capacity                = 4000000;
    compute_clock_ghz          = 1;
    parallel_mi_cu             = 1;
    mem_bw_per_wg_coefficients = std::make_tuple(0, 0.015, 0);
  } else if (gpu_arch == 950) {
    n_cu                       = 256;
    lds_capacity               = 163840;
    num_xcd                    = 8;
    mem1_perf_ratio            = 1.0;
    mem2_perf_ratio            = 1.0;
    mem3_perf_ratio            = 1.0;
    l2_capacity                = 4000000;
    compute_clock_ghz          = 1.2;
    parallel_mi_cu             = 1;
    mem_bw_per_wg_coefficients = std::make_tuple(0, 0.008, 0);
  } else if(gpu_arch == 1250) {
    // TODO: using gfx950 placeholders for most fields, update lds_capacity and l2_capacity later
    auto hw = make_hardware(950);
    hw.arch = origami::hardware_t::architecture_t::gfx1250;
    hw.mem_bw_per_wg_coefficients = std::make_tuple(0, 0.016, 0);
    return hw;
  }

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
