// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cmath>
#include <iostream>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/logger.hpp"
#include "origami/types.hpp"

namespace origami {

// ============================================================================
// heuristic_params_t Implementation
// ============================================================================

void heuristic_params_t::merge_with(const heuristic_params_t& other) {
  // Latency component weights
  weight_mem_l2        = other.weight_mem_l2;
  weight_mem_mall      = other.weight_mem_mall;
  weight_mem_dram      = other.weight_mem_dram;
  weight_compute       = other.weight_compute;
  weight_memory        = other.weight_memory;
  weight_wg_setup      = other.weight_wg_setup;
  weight_prologue      = other.weight_prologue;
  weight_epilogue      = other.weight_epilogue;
  weight_loop_overhead = other.weight_loop_overhead;
  weight_tile_total    = other.weight_tile_total;

  // Empirical constants
  main_memory_load_latency            = other.main_memory_load_latency;
  occupancy_decay_base                = other.occupancy_decay_base;
  mall_depth_sq                       = other.mall_depth_sq;
  mall_cold_floor                     = other.mall_cold_floor;
  l2_depth_sq                         = other.l2_depth_sq;
  l2_cold_floor                       = other.l2_cold_floor;
  l2_pollution_penalty                = other.l2_pollution_penalty;
  l2_amp_ceiling_batched              = other.l2_amp_ceiling_batched;
  l2_amp_ceiling_k_split              = other.l2_amp_ceiling_k_split;
  l2_amp_ceiling_skinny               = other.l2_amp_ceiling_skinny;
  l2_depth_penalty                    = other.l2_depth_penalty;
  l1_hit_rate_ceiling_skinny          = other.l1_hit_rate_ceiling_skinny;
  epilogue_cycles_per_acc_read        = other.epilogue_cycles_per_acc_read;
  epilogue_acc_read_parallelism       = other.epilogue_acc_read_parallelism;
  epilogue_cycles_per_bounds_check    = other.epilogue_cycles_per_bounds_check;
  epilogue_scalar_store_penalty       = other.epilogue_scalar_store_penalty;
  epilogue_threads_per_wave           = other.epilogue_threads_per_wave;
  epilogue_bytes_per_vectorized_store = other.epilogue_bytes_per_vectorized_store;
  epilogue_cache_line_bytes           = other.epilogue_cache_line_bytes;
  epilogue_workspace_bytes_per_elem   = other.epilogue_workspace_bytes_per_elem;
  epilogue_salu_overhead              = other.epilogue_salu_overhead;
  epilogue_l_barrier                  = other.epilogue_l_barrier;
  epilogue_l_smem                     = other.epilogue_l_smem;
  epilogue_k_padding_penalty          = other.epilogue_k_padding_penalty;
  postgsu_compute_bytes               = other.postgsu_compute_bytes;
  postgsu_kernel_launch_overhead      = other.postgsu_kernel_launch_overhead;
  postgsu_threads_per_wg              = other.postgsu_threads_per_wg;
  postgsu_wavefront_size              = other.postgsu_wavefront_size;

  // Main loop efficiency
  main_loop_efficiency = other.main_loop_efficiency;
}

// ============================================================================
// heuristic_key_t Implementation
// ============================================================================

bool heuristic_key_t::matches(const problem_t& problem,
                              const hardware_t& hardware,
                              const config_t& config) const {
  // Check each field - if optional is set, it must match
  if (arch.has_value() && arch.value() != hardware.arch) return false;
  if (a_dtype.has_value() && a_dtype.value() != problem.a_dtype) return false;
  if (b_dtype.has_value() && b_dtype.value() != problem.b_dtype) return false;
  if (mi_dtype.has_value() && mi_dtype.value() != problem.mi_dtype) return false;
  if (a_transpose.has_value() && a_transpose.value() != problem.a_transpose) return false;
  if (b_transpose.has_value() && b_transpose.value() != problem.b_transpose) return false;
  if (mt_m.has_value() && mt_m.value() != config.mt.m) return false;
  if (mt_n.has_value() && mt_n.value() != config.mt.n) return false;
  if (mt_k.has_value() && mt_k.value() != config.mt.k) return false;
  if (hand_optimized_main_loop.has_value() &&
      hand_optimized_main_loop.value() != config.hand_optimized_main_loop)
    return false;

  // Problem size ranges
  if (min_m.has_value() && problem.size.m < min_m.value()) return false;
  if (max_m.has_value() && problem.size.m > max_m.value()) return false;
  if (min_n.has_value() && problem.size.n < min_n.value()) return false;
  if (max_n.has_value() && problem.size.n > max_n.value()) return false;
  if (min_k.has_value() && problem.size.k < min_k.value()) return false;
  if (max_k.has_value() && problem.size.k > max_k.value()) return false;

  return true;
}

size_t heuristic_key_t::specificity() const {
  size_t count = 0;
  if (arch.has_value()) count++;
  if (a_dtype.has_value()) count++;
  if (b_dtype.has_value()) count++;
  if (mi_dtype.has_value()) count++;
  if (a_transpose.has_value()) count++;
  if (b_transpose.has_value()) count++;
  if (mt_m.has_value()) count++;
  if (mt_n.has_value()) count++;
  if (mt_k.has_value()) count++;
  if (hand_optimized_main_loop.has_value()) count++;
  if (min_m.has_value()) count++;
  if (max_m.has_value()) count++;
  if (min_n.has_value()) count++;
  if (max_n.has_value()) count++;
  if (min_k.has_value()) count++;
  if (max_k.has_value()) count++;
  return count;
}

// ============================================================================
// heuristics_database_t Implementation
// ============================================================================

heuristics_database_t::heuristics_database_t() { initialize_defaults(); }

heuristics_database_t& heuristics_database_t::get_instance() {
  static heuristics_database_t instance;
  return instance;
}

/**
 * @brief Apply TF32 emulation heuristics based on runtime arithmetic intensity.
 *
 * These heuristics cannot be precomputed since they depend on problem size.
 */
static void apply_tf32_heuristics(heuristic_params_t& params,
                                  const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config) {
  // Check if this is TF32 emulation on gfx950
  const bool is_gfx950   = (hardware.arch == hardware_t::architecture_t::gfx950);
  const bool is_tf32_emu = (problem.mi_dtype == data_type_t::XFloat32) && is_gfx950;

  if (!is_tf32_emu) return;

  const size_t M = problem.size.m;
  const size_t N = problem.size.n;
  const size_t K = problem.size.k;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;
  const size_t MT_K = config.mt.k;

  const bool a_trans = (problem.a_transpose == transpose_t::T);
  const bool b_trans = (problem.b_transpose == transpose_t::T);

  const auto a_bytes = data_type_to_bytes(problem.a_dtype);

  // Compute arithmetic intensity for this specific problem
  double arith     = emulated_tf32_arithmetic_intensity(M, N, K, static_cast<double>(a_bytes));
  double threshold = heuristic_defaults_t::TF32_ARITH_INTENSITY_THRESHOLD;

  // Custom kernel optimizations based on transpose mode and tile config
  // NT: N-transpose configuration
  if ((!a_trans && b_trans) && MT_M == 256 && MT_N == 256 && MT_K == 32) {
    if (arith < threshold) {
      params.weight_tile_total *= 0.6;
    } else {
      params.weight_tile_total *= 0.4;
    }
  }

  // NN: No-transpose configuration
  if ((!a_trans && !b_trans) && MT_M == 256 && MT_N == 256 && MT_K == 32) {
    if (arith < threshold) {
      params.weight_tile_total *= 0.8;
    } else {
      params.weight_tile_total *= 0.4;
    }
  }

  // TN: Transpose-A configuration
  if ((a_trans && !b_trans) && MT_M == 256 && MT_N == 256 && MT_K == 32) {
    if (arith < threshold) {
      params.weight_tile_total *= 0.8;
    } else {
      params.weight_tile_total *= 0.4;
    }
  }

  // Bias for large K-dimension (depth upscaling)
  if ((K >= (M * 16) && K >= (N * 16)) && (MT_K >= 128)) { params.weight_tile_total *= 0.5; }
}

heuristic_params_t heuristics_database_t::lookup(const problem_t& problem,
                                                 const hardware_t& hardware,
                                                 const config_t& config) const {
  // When heuristics are disabled, always return default parameters (no overrides).
  if (!origami::runtime_options::get().heuristics_enabled) { return default_params_; }

  // Start with default parameters
  heuristic_params_t result = default_params_;

  // Fast path: O(1) lookup for hand-optimized kernels
  if (config.hand_optimized_main_loop) {
    hand_optimized_kernel_key_t fast_key{hardware.arch,
                                         problem.mi_dtype,
                                         problem.a_transpose,
                                         problem.b_transpose,
                                         config.mt.m,
                                         config.mt.n,
                                         config.mt.k};

    auto it = hand_optimized_map_.find(fast_key);
    if (it != hand_optimized_map_.end()) {
      if (origami::runtime_options::get().debug_enabled) {
        OLOG_DEBUG("Hand-optimized kernel " << fast_key.to_string()
                                            << ", efficiency: " << it->second.main_loop_efficiency);
      }
      result = it->second;
    }
  }

  // Slow path: O(n) hierarchical lookup for general heuristics
  // Find all matching entries and sort by specificity
  std::vector<std::pair<size_t, const heuristic_params_t*>> matches;
  for (const auto& [key, params] : entries_) {
    if (key.matches(problem, hardware, config)) { matches.push_back({key.specificity(), &params}); }
  }

  // Sort by specificity (least specific first, so more specific ones override)
  std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });

  // Apply matches in order of increasing specificity
  for (const auto& [spec, params] : matches) { result.merge_with(*params); }

  // Apply TF32 emulation heuristics (runtime-dependent on arithmetic intensity)
  apply_tf32_heuristics(result, problem, hardware, config);

  return result;
}

void heuristics_database_t::add_entry(const heuristic_key_t& key,
                                      const heuristic_params_t& params) {
  // If this is a hand-optimized kernel, also add to fast lookup map
  if (key.hand_optimized_main_loop.has_value() && key.hand_optimized_main_loop.value()) {
    // Hand-optimized kernels must have all required fields specified
    if (key.arch.has_value() && key.mi_dtype.has_value() && key.a_transpose.has_value() &&
        key.b_transpose.has_value() && key.mt_m.has_value() && key.mt_n.has_value() &&
        key.mt_k.has_value()) {
      hand_optimized_kernel_key_t fast_key{key.arch.value(),
                                           key.mi_dtype.value(),
                                           key.a_transpose.value(),
                                           key.b_transpose.value(),
                                           key.mt_m.value(),
                                           key.mt_n.value(),
                                           key.mt_k.value()};

      hand_optimized_map_[fast_key] = params;
    }
  } else {
    entries_.push_back({key, params});
  }
}

bool heuristics_database_t::has_hand_optimized_entry(hardware_t::architecture_t arch,
                                                     data_type_t mi_dtype,
                                                     transpose_t transA,
                                                     transpose_t transB,
                                                     size_t mt_m,
                                                     size_t mt_n,
                                                     size_t mt_k) const {
  hand_optimized_kernel_key_t key{arch, mi_dtype, transA, transB, mt_m, mt_n, mt_k};
  return hand_optimized_map_.find(key) != hand_optimized_map_.end();
}

void heuristics_database_t::initialize_defaults() {
  // ========================================================================
  // HEURISTIC 1: Problematic tile configuration (MT64x32x32)
  // ========================================================================
  {
    auto key    = make_tile_key(64, 32, 32, transpose_t::N, transpose_t::N);
    key.a_dtype = data_type_t::BFloat16;
    key.b_dtype = data_type_t::BFloat16;

    heuristic_params_t params;
    params.weight_tile_total = 10.0;

    add_entry(key, params);
  }

  // ========================================================================
  // HEURISTIC 2: CMS Kernel Efficiencies (gfx950, BF16)
  // ========================================================================
  {
    // BF16 NT configurations
    struct cms_config {
      size_t m, n, k;
      double eff;
    };
    std::vector<cms_config> bf16_nt_configs = {
        {160, 256, 64, 1.0 / 1.20},
        {192, 256, 64, 1.0 / 1.10},
        {208, 256, 64, 1.0 / 1.20},
        {256, 160, 64, 1.0 / 1.20},
        {256, 192, 64, 1.0 / 1.20},
        {256, 256, 64, 1.0 / 1.15},
    };

    for (const auto& cfg : bf16_nt_configs) {
      auto key = make_hand_optimized_kernel_key(hardware_t::architecture_t::gfx950,
                                                data_type_t::BFloat16,
                                                transpose_t::N,
                                                transpose_t::T,
                                                cfg.m,
                                                cfg.n,
                                                cfg.k);
      heuristic_params_t params;
      params.main_loop_efficiency = cfg.eff;
      add_entry(key, params);
    }

    // BF16 NN configurations
    std::vector<cms_config> bf16_nn_configs = {
        {160, 256, 64, 1.0 / 1.10},
        {208, 256, 64, 1.0 / 1.10},
        {256, 192, 64, 1.0 / 1.00},
        {256, 256, 64, 1.0 / 1.05},
    };

    for (const auto& cfg : bf16_nn_configs) {
      auto key = make_hand_optimized_kernel_key(hardware_t::architecture_t::gfx950,
                                                data_type_t::BFloat16,
                                                transpose_t::N,
                                                transpose_t::N,
                                                cfg.m,
                                                cfg.n,
                                                cfg.k);
      heuristic_params_t params;
      params.main_loop_efficiency = cfg.eff;
      add_entry(key, params);
    }

    // BF16 TN configurations
    std::vector<cms_config> bf16_tn_configs = {
        {160, 256, 64, 1.0 / 1.10},
        {192, 256, 64, 1.0 / 1.05},
        {256, 96, 64, 1.0 / 1.10},
        {256, 192, 64, 1.0 / 1.10},
        {256, 224, 64, 1.0 / 1.05},
        {256, 256, 64, 1.0 / 1.05},
    };

    for (const auto& cfg : bf16_tn_configs) {
      auto key = make_hand_optimized_kernel_key(hardware_t::architecture_t::gfx950,
                                                data_type_t::BFloat16,
                                                transpose_t::T,
                                                transpose_t::N,
                                                cfg.m,
                                                cfg.n,
                                                cfg.k);
      heuristic_params_t params;
      params.main_loop_efficiency = cfg.eff;
      add_entry(key, params);
    }

    // BF16 TT configurations
    std::vector<cms_config> bf16_tt_configs = {
        {256, 256, 64, 1.0 / 1.10},
    };

    for (const auto& cfg : bf16_tt_configs) {
      auto key = make_hand_optimized_kernel_key(hardware_t::architecture_t::gfx950,
                                                data_type_t::BFloat16,
                                                transpose_t::T,
                                                transpose_t::T,
                                                cfg.m,
                                                cfg.n,
                                                cfg.k);
      heuristic_params_t params;
      params.main_loop_efficiency = cfg.eff;
      add_entry(key, params);
    }

    // TF32 NN
    std::vector<cms_config> tf32_nn_configs = {
        {192, 256, 32, 1.0 / 1.23},
    };

    for (const auto& cfg : tf32_nn_configs) {
      auto key = make_hand_optimized_kernel_key(hardware_t::architecture_t::gfx950,
                                                data_type_t::XFloat32,
                                                transpose_t::N,
                                                transpose_t::N,
                                                cfg.m,
                                                cfg.n,
                                                cfg.k);
      heuristic_params_t params;
      params.main_loop_efficiency = cfg.eff;
      add_entry(key, params);
    }

    // TF32 TN
    std::vector<cms_config> tf32_tn_configs = {
        {128, 256, 32, 1.0 / 1.26},
        {192, 256, 32, 1.0 / 1.23},
    };

    for (const auto& cfg : tf32_tn_configs) {
      auto key = make_hand_optimized_kernel_key(hardware_t::architecture_t::gfx950,
                                                data_type_t::XFloat32,
                                                transpose_t::T,
                                                transpose_t::N,
                                                cfg.m,
                                                cfg.n,
                                                cfg.k);
      heuristic_params_t params;
      params.main_loop_efficiency = cfg.eff;
      add_entry(key, params);
    }
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

heuristic_key_t make_hand_optimized_kernel_key(hardware_t::architecture_t arch,
                                               data_type_t mi_dtype,
                                               transpose_t transA,
                                               transpose_t transB,
                                               size_t MT_M,
                                               size_t MT_N,
                                               size_t MT_K) {
  heuristic_key_t key;
  key.arch                     = arch;
  key.mi_dtype                 = mi_dtype;
  key.a_transpose              = transA;
  key.b_transpose              = transB;
  key.mt_m                     = MT_M;
  key.mt_n                     = MT_N;
  key.mt_k                     = MT_K;
  key.hand_optimized_main_loop = true;
  return key;
}

heuristic_key_t make_tile_key(size_t MT_M,
                              size_t MT_N,
                              size_t MT_K,
                              std::optional<transpose_t> transA,
                              std::optional<transpose_t> transB) {
  heuristic_key_t key;
  key.mt_m        = MT_M;
  key.mt_n        = MT_N;
  key.mt_k        = MT_K;
  key.a_transpose = transA;
  key.b_transpose = transB;
  return key;
}

heuristic_key_t make_arch_dtype_key(hardware_t::architecture_t arch, data_type_t mi_dtype) {
  heuristic_key_t key;
  key.arch     = arch;
  key.mi_dtype = mi_dtype;
  return key;
}

}  // namespace origami
