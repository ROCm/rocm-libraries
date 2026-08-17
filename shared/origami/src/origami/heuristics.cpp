// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/logger.hpp"
#include "origami/types.hpp"

namespace {

namespace cms_speedup {
constexpr uint8_t X100 = 100;
constexpr uint8_t X105 = 105;
constexpr uint8_t X110 = 110;
constexpr uint8_t X115 = 115;
constexpr uint8_t X120 = 120;
constexpr uint8_t X123 = 123;
constexpr uint8_t X126 = 126;
}  // namespace cms_speedup

constexpr double cms_efficiency(uint8_t speedup_x100) {
  return 100.0 / static_cast<double>(speedup_x100);
}

struct cms_kernel_entry_t {
  origami::data_type_t mi_dtype;
  origami::transpose_t trans_a;
  origami::transpose_t trans_b;
  uint16_t m;
  uint16_t n;
  uint16_t k;
  uint8_t speedup_x100;
};

// Register new architectures by adding a table here.
constexpr std::array<cms_kernel_entry_t, 38> gfx950_cms_kernels = {{
    // BF16 NT
    {origami::data_type_t::BFloat16, origami::transpose_t::N, origami::transpose_t::T, 160, 256, 64,
     cms_speedup::X120},
    {origami::data_type_t::BFloat16, origami::transpose_t::N, origami::transpose_t::T, 192, 256, 64,
     cms_speedup::X110},
    {origami::data_type_t::BFloat16, origami::transpose_t::N, origami::transpose_t::T, 208, 256, 64,
     cms_speedup::X120},
    {origami::data_type_t::BFloat16, origami::transpose_t::N, origami::transpose_t::T, 256, 160, 64,
     cms_speedup::X120},
    {origami::data_type_t::BFloat16, origami::transpose_t::N, origami::transpose_t::T, 256, 192, 64,
     cms_speedup::X120},
    {origami::data_type_t::BFloat16, origami::transpose_t::N, origami::transpose_t::T, 256, 256, 64,
     cms_speedup::X115},
    // BF16 NN
    {origami::data_type_t::BFloat16, origami::transpose_t::N, origami::transpose_t::N, 160, 256, 64,
     cms_speedup::X110},
    {origami::data_type_t::BFloat16, origami::transpose_t::N, origami::transpose_t::N, 208, 256, 64,
     cms_speedup::X110},
    {origami::data_type_t::BFloat16, origami::transpose_t::N, origami::transpose_t::N, 256, 192, 64,
     cms_speedup::X100},
    {origami::data_type_t::BFloat16, origami::transpose_t::N, origami::transpose_t::N, 256, 256, 64,
     cms_speedup::X105},
    // BF16 TN
    {origami::data_type_t::BFloat16, origami::transpose_t::T, origami::transpose_t::N, 160, 256, 64,
     cms_speedup::X110},
    {origami::data_type_t::BFloat16, origami::transpose_t::T, origami::transpose_t::N, 192, 256, 64,
     cms_speedup::X105},
    {origami::data_type_t::BFloat16, origami::transpose_t::T, origami::transpose_t::N, 256, 96, 64,
     cms_speedup::X110},
    {origami::data_type_t::BFloat16, origami::transpose_t::T, origami::transpose_t::N, 256, 192, 64,
     cms_speedup::X110},
    {origami::data_type_t::BFloat16, origami::transpose_t::T, origami::transpose_t::N, 256, 224, 64,
     cms_speedup::X105},
    {origami::data_type_t::BFloat16, origami::transpose_t::T, origami::transpose_t::N, 256, 256, 64,
     cms_speedup::X105},
    // BF16 TT
    {origami::data_type_t::BFloat16, origami::transpose_t::T, origami::transpose_t::T, 256, 256, 64,
     cms_speedup::X110},
    // FP16 NT
    {origami::data_type_t::Half, origami::transpose_t::N, origami::transpose_t::T, 192, 320, 64,
     cms_speedup::X110},
    {origami::data_type_t::Half, origami::transpose_t::N, origami::transpose_t::T, 208, 256, 64,
     cms_speedup::X120},
    {origami::data_type_t::Half, origami::transpose_t::N, origami::transpose_t::T, 224, 128, 64,
     cms_speedup::X120},
    {origami::data_type_t::Half, origami::transpose_t::N, origami::transpose_t::T, 256, 192, 64,
     cms_speedup::X120},
    {origami::data_type_t::Half, origami::transpose_t::N, origami::transpose_t::T, 256, 256, 64,
     cms_speedup::X115},
    // FP16 NN
    {origami::data_type_t::Half, origami::transpose_t::N, origami::transpose_t::N, 128, 256, 64,
     cms_speedup::X110},
    {origami::data_type_t::Half, origami::transpose_t::N, origami::transpose_t::N, 160, 256, 64,
     cms_speedup::X110},
    {origami::data_type_t::Half, origami::transpose_t::N, origami::transpose_t::N, 256, 160, 64,
     cms_speedup::X110},
    {origami::data_type_t::Half, origami::transpose_t::N, origami::transpose_t::N, 192, 256, 64,
     cms_speedup::X110},
    {origami::data_type_t::Half, origami::transpose_t::N, origami::transpose_t::N, 256, 192, 64,
     cms_speedup::X100},
    {origami::data_type_t::Half, origami::transpose_t::N, origami::transpose_t::N, 256, 256, 64,
     cms_speedup::X105},
    // FP16 TN
    {origami::data_type_t::Half, origami::transpose_t::T, origami::transpose_t::N, 160, 256, 64,
     cms_speedup::X110},
    {origami::data_type_t::Half, origami::transpose_t::T, origami::transpose_t::N, 192, 256, 64,
     cms_speedup::X105},
    {origami::data_type_t::Half, origami::transpose_t::T, origami::transpose_t::N, 256, 96, 64,
     cms_speedup::X110},
    {origami::data_type_t::Half, origami::transpose_t::T, origami::transpose_t::N, 256, 192, 64,
     cms_speedup::X110},
    {origami::data_type_t::Half, origami::transpose_t::T, origami::transpose_t::N, 256, 224, 64,
     cms_speedup::X105},
    {origami::data_type_t::Half, origami::transpose_t::T, origami::transpose_t::N, 256, 256, 64,
     cms_speedup::X105},
    // FP16 TT
    {origami::data_type_t::Half, origami::transpose_t::T, origami::transpose_t::T, 256, 256, 64,
     cms_speedup::X110},
    // TF32 NN
    {origami::data_type_t::XFloat32, origami::transpose_t::N, origami::transpose_t::N, 192, 256, 32,
     cms_speedup::X123},
    // TF32 TN
    {origami::data_type_t::XFloat32, origami::transpose_t::T, origami::transpose_t::N, 128, 256, 32,
     cms_speedup::X126},
    {origami::data_type_t::XFloat32, origami::transpose_t::T, origami::transpose_t::N, 192, 256, 32,
     cms_speedup::X123},
}};

constexpr size_t cms_kernel_entry_count() { return gfx950_cms_kernels.size(); }

void register_cms_kernels(origami::heuristics_database_t& db) {
  for (const auto& entry : gfx950_cms_kernels) {
    db.add_hand_optimized_efficiency(
        origami::hand_optimized_kernel_key_t{origami::hardware_t::architecture_t::gfx950,
                                           entry.mi_dtype,
                                           entry.trans_a,
                                           entry.trans_b,
                                           entry.m,
                                           entry.n,
                                           entry.k},
        cms_efficiency(entry.speedup_x100));
  }
}

}  // namespace

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

  // Resource and edge terms
  resource_residency_weight = other.resource_residency_weight;
  resource_residency_target = other.resource_residency_target;
  edge_tile_penalty_weight  = other.edge_tile_penalty_weight;
  depth_u_edge_weight       = other.depth_u_edge_weight;
  deep_k_pipeline_weight    = other.deep_k_pipeline_weight;

  // Kernel rejection
  reject = other.reject;
  // VGPR / single-wave occupancy penalty
  vgpr_penalty_weight = other.vgpr_penalty_weight;
  vgpr_per_simd       = other.vgpr_per_simd;
  vgpr_threads_per_wg = other.vgpr_threads_per_wg;
  vgpr_overhead       = other.vgpr_overhead;
  vgpr_operand_coeff  = other.vgpr_operand_coeff;
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
  if (subtile.has_value() && subtile.value() != config.subtile) return false;

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
  if (subtile.has_value()) count++;
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

heuristics_database_t::heuristics_database_t() {
  hand_optimized_map_.reserve(cms_kernel_entry_count());
  initialize_defaults();
}

heuristics_database_t& heuristics_database_t::get_instance() {
  static heuristics_database_t instance;
  return instance;
}

void heuristics_database_t::reset_defaults() {
  entries_.clear();
  hand_optimized_map_.clear();
  default_params_ = heuristic_params_t{};
  initialize_defaults();
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
  double arith     = gemm::emulated_tf32_arithmetic_intensity(M, N, K, static_cast<double>(a_bytes));
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

void heuristics_database_t::add_hand_optimized_efficiency(hand_optimized_kernel_key_t key,
                                                          double main_loop_efficiency) {
  hand_optimized_map_[key].main_loop_efficiency = main_loop_efficiency;
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
  // HEURISTIC 0: gfx1100 HHS (FP16) TN tuned heuristics (per shape-size class)
  // Discovered by a genetic-algorithm campaign over 8542 measured gfx1100 HHS
  // TN GEMM shapes, validated on a held-out 20% test split with a strict
  // no-per-category-regression gate. Two tuned parameter sets are used:
  //   - "v2c" (rank-focused GA): default for small/medium shapes. Held-out
  //     test vs the untuned model: top-10 46%->73%, top-5 29%->51%.
  //   - "v1"  (composite GA): retained for degenerate (a dim == 1) and large
  //     (a dim >= 8192) shapes, where it ranks at least as well as v2c.
  // Net vs v1-everywhere on held-out test: top-10 +1.5pp, top-5 +2.3pp,
  // top-1 +1.1pp, zero per-category regression. Both sets include the new
  // VGPR single-wave penalty (penalizes macro tiles fitting < 2 waves/SIMD,
  // e.g. MT256x256); the GA drove that weight high in both.
  // ========================================================================
  {
    // v1 parameter set (degenerate + large shapes).
    heuristic_params_t v1;
    v1.weight_compute            = 1.610029;
    v1.weight_memory             = 0.761594;
    v1.weight_prologue           = 3.0;
    v1.weight_epilogue           = 4.0;
    v1.weight_loop_overhead      = 71.217592;
    v1.weight_tile_total         = 1.5;
    v1.weight_mem_l2             = 4.0;
    v1.weight_mem_mall           = 1.764576;
    v1.weight_mem_dram           = 0.492963;
    v1.weight_wg_setup           = 7.228919;
    v1.main_memory_load_latency  = 255.690018;
    v1.occupancy_decay_base      = 1.0;
    v1.epilogue_k_padding_penalty   = 0.0;
    v1.epilogue_salu_overhead       = 54.675904;
    v1.epilogue_l_smem              = 1832.520767;
    v1.epilogue_cycles_per_acc_read = 24.0;
    v1.vgpr_penalty_weight = 2.0;
    v1.vgpr_per_simd       = 256.0;
    v1.vgpr_overhead       = 78.797305;

    // v2c parameter set (small + medium shapes; the default for this key).
    heuristic_params_t v2c;
    v2c.weight_compute               = 0.344637;
    v2c.weight_memory                = 0.113557;
    v2c.weight_prologue              = 3.357214;
    v2c.weight_epilogue              = 9.262855;
    v2c.weight_loop_overhead         = 133.395615;
    v2c.weight_tile_total            = 2.698667;
    v2c.weight_mem_l2                = 5.795668;
    v2c.weight_mem_mall              = 1.357384;
    v2c.weight_mem_dram              = 0.000000;
    v2c.weight_wg_setup              = 0.996566;
    v2c.main_memory_load_latency     = 400.000000;
    v2c.occupancy_decay_base         = 0.700000;
    v2c.epilogue_k_padding_penalty   = 0.000000;
    v2c.epilogue_salu_overhead       = 100.000000;
    v2c.epilogue_l_smem              = 0.000000;
    v2c.epilogue_cycles_per_acc_read = 64.000000;
    v2c.vgpr_penalty_weight          = 3.983455;
    v2c.vgpr_per_simd                = 347.235575;
    v2c.vgpr_overhead                = 125.800302;

    auto base_key = []() {
      heuristic_key_t k;
      k.arch        = hardware_t::architecture_t::gfx1100;
      k.mi_dtype    = data_type_t::Half;
      k.a_transpose = transpose_t::T;  // TN: transA = T
      k.b_transpose = transpose_t::N;  // transB = N
      return k;
    };

    // Default for this arch/dtype/layout: v2c (covers small + medium, the bulk).
    add_entry(base_key(), v2c);

    // Large shapes (any dimension >= 8192): override back to v1. Three keys
    // (one per dimension) union to "any dim >= 8192"; overlap is harmless since
    // all install the same params. More specific than the base key, so they win.
    {
      auto k = base_key(); k.min_m = 8192; add_entry(k, v1);
    }
    {
      auto k = base_key(); k.min_n = 8192; add_entry(k, v1);
    }
    {
      auto k = base_key(); k.min_k = 8192; add_entry(k, v1);
    }

    // Degenerate shapes (any dimension == 1): override back to v1. Same union
    // trick via max_* = 1 on each dimension.
    {
      auto k = base_key(); k.max_m = 1; add_entry(k, v1);
    }
    {
      auto k = base_key(); k.max_n = 1; add_entry(k, v1);
    }
    {
      auto k = base_key(); k.max_k = 1; add_entry(k, v1);
    }
  }

  // ========================================================================
  // HEURISTIC 0b: gfx1100 FP32 (Float) TN tuned heuristics
  // Tuned by a GA against full per-kernel FP32 TN BenchmarkData matrices (850
  // shapes, up to 3060 measured kernels each), optimizing winner-capture AND
  // pairwise ranking concordance. Held-out test (vs untuned): concordance
  // 72.7%->74.9%, top-5 62.8%->65.1%, top-1 26.7%->28.5%, no per-category
  // regression. FP32 on gfx1100 is the SIMD MAC path (no WMMA). Applied to
  // small+medium shapes; degenerate (a dim == 1) and large (a dim >= 8192)
  // keep stock Float behavior where the tuned set showed no gain / risk.
  // ========================================================================
  {
    heuristic_params_t fp32;
    fp32.weight_compute               = 1.818770;
    fp32.weight_memory                = 0.760139;
    fp32.weight_prologue              = 5.969580;
    fp32.weight_epilogue              = 3.907783;
    fp32.weight_loop_overhead         = 1000.000000;
    fp32.weight_tile_total            = 0.300000;
    fp32.weight_mem_l2                = 6.411981;
    fp32.weight_mem_mall              = 2.000000;
    fp32.weight_mem_dram              = 0.589558;
    fp32.weight_wg_setup              = 8.948394;
    fp32.main_memory_load_latency     = 111.446748;
    fp32.occupancy_decay_base         = 0.713312;
    fp32.epilogue_k_padding_penalty   = 0.000000;
    fp32.epilogue_salu_overhead       = 0.000000;
    fp32.epilogue_l_smem              = 39.204087;
    fp32.epilogue_cycles_per_acc_read = 60.323274;
    fp32.vgpr_penalty_weight          = 6.000000;
    fp32.vgpr_per_simd                = 219.078480;
    fp32.vgpr_overhead                = 45.561929;

    // Stock Float defaults for degenerate/large (revert overrides).
    heuristic_params_t stock;

    auto fp32_key = []() {
      heuristic_key_t k;
      k.arch        = hardware_t::architecture_t::gfx1100;
      k.mi_dtype    = data_type_t::Float;
      k.a_transpose = transpose_t::T;
      k.b_transpose = transpose_t::N;
      return k;
    };

    add_entry(fp32_key(), fp32);  // default for small+medium

    // Large (any dim >= 8192): kept on stock. A 5-fold CV showed a large-tune
    // generalizes on composite fitness (+0.61), but on the held-out split it
    // traded top-1/top-5 for top-10/concordance — fails the strict
    // no-regression gate, so TN-large stays stock. (NN/NT large DID bank a
    // tune; see 0c/0d.)
    { auto k = fp32_key(); k.min_m = 8192; add_entry(k, stock); }
    { auto k = fp32_key(); k.min_n = 8192; add_entry(k, stock); }
    { auto k = fp32_key(); k.min_k = 8192; add_entry(k, stock); }
    { auto k = fp32_key(); k.max_m = 1; add_entry(k, stock); }
    { auto k = fp32_key(); k.max_n = 1; add_entry(k, stock); }
    { auto k = fp32_key(); k.max_k = 1; add_entry(k, stock); }
  }

  // ========================================================================
  // HEURISTIC 0c: gfx1100 FP32 (Float) NN tuned heuristics
  // GA-tuned vs full per-kernel FP32 NN BenchmarkData (837 shapes). NN was
  // Origami's weakest layout out of the box (top-10 only 55%). Held-out test
  // (vs untuned): top-10 55.9%->67.6%, top-5 44.7%->52.9%, concordance
  // 74.6%->78.2%, no per-category regression -> applied to all size classes.
  // ========================================================================
  {
    heuristic_params_t p;
    p.weight_compute                  = 0.100000;
    p.weight_memory                   = 3.000000;
    p.weight_prologue                 = 8.000000;
    p.weight_epilogue                 = 0.866067;
    p.weight_loop_overhead            = 1000.000000;
    p.weight_tile_total               = 0.300000;
    p.weight_mem_l2                   = 10.600009;
    p.weight_mem_mall                 = 0.235351;
    p.weight_mem_dram                 = 2.000000;
    p.weight_wg_setup                 = 9.011839;
    p.main_memory_load_latency        = 400.000000;
    p.occupancy_decay_base            = 0.967072;
    p.epilogue_k_padding_penalty      = 14336.713083;
    p.epilogue_salu_overhead          = 100.000000;
    p.epilogue_l_smem                 = 6000.000000;
    p.epilogue_cycles_per_acc_read    = 64.000000;
    p.vgpr_penalty_weight             = 1.489363;
    p.vgpr_per_simd                   = 1134.906654;
    p.vgpr_overhead                   = 57.404322;

    auto nn_key = []() {
      heuristic_key_t k;
      k.arch        = hardware_t::architecture_t::gfx1100;
      k.mi_dtype    = data_type_t::Float;
      k.a_transpose = transpose_t::N;  // NN
      k.b_transpose = transpose_t::N;
      return k;
    };
    add_entry(nn_key(), p);  // default (degenerate+small+medium)

    // Large (any dim >= 8192): dedicated large-tuned set. 5-fold CV mean +1.87
    // vs stock (5/5 folds; held-out test large top-10 26%->53%). NN large was
    // very poorly served by the generic default — biggest large-class gain.
    heuristic_params_t L;
    L.weight_compute                  = 0.200000;
    L.weight_memory                   = 1.345074;
    L.weight_prologue                 = 2.495361;
    L.weight_epilogue                 = 3.157302;
    L.weight_loop_overhead            = 1000.000000;
    L.weight_tile_total               = 0.500000;
    L.weight_mem_l2                   = 3.668571;
    L.weight_mem_mall                 = 0.997982;
    L.weight_mem_dram                 = 1.027456;
    L.weight_wg_setup                 = 0.200286;
    L.main_memory_load_latency        = 385.182145;
    L.occupancy_decay_base            = 0.878377;
    L.epilogue_k_padding_penalty      = 100000.000000;
    L.epilogue_salu_overhead          = 100.000000;
    L.epilogue_l_smem                 = 1294.782362;
    L.epilogue_cycles_per_acc_read    = 24.000000;
    L.vgpr_penalty_weight             = 0.669928;
    L.vgpr_per_simd                   = 496.222847;
    L.vgpr_overhead                   = 0.000000;
    { auto k = nn_key(); k.min_m = 8192; add_entry(k, L); }
    { auto k = nn_key(); k.min_n = 8192; add_entry(k, L); }
    { auto k = nn_key(); k.min_k = 8192; add_entry(k, L); }
  }

  // ========================================================================
  // HEURISTIC 0d: gfx1100 FP32 (Float) NT tuned heuristics
  // GA-tuned vs full per-kernel FP32 NT BenchmarkData (821 shapes). Held-out
  // test (vs untuned): top-1 20.4%->28.7%, top-5 61.7%->65.9%, top-10
  // 87.4%->89.8%, concordance 71.8%->72.5%. Medium shapes keep stock (the
  // tuned set dipped their pairwise concordance); all other classes use opt.
  // No per-category regression.
  // ========================================================================
  {
    heuristic_params_t p;
    p.weight_compute                  = 1.107503;
    p.weight_memory                   = 0.100000;
    p.weight_prologue                 = 1.264417;
    p.weight_epilogue                 = 1.320345;
    p.weight_loop_overhead            = 857.532443;
    p.weight_tile_total               = 0.342988;
    p.weight_mem_l2                   = 3.650370;
    p.weight_mem_mall                 = 1.122812;
    p.weight_mem_dram                 = 1.159342;
    p.weight_wg_setup                 = 0.016999;
    p.main_memory_load_latency        = 364.106727;
    p.occupancy_decay_base            = 0.985556;
    p.epilogue_k_padding_penalty      = 0.000000;
    p.epilogue_salu_overhead          = 44.133680;
    p.epilogue_l_smem                 = 3856.659889;
    p.epilogue_cycles_per_acc_read    = 64.000000;
    p.vgpr_penalty_weight             = 6.000000;
    p.vgpr_per_simd                   = 156.891423;
    p.vgpr_overhead                   = 4.575689;

    heuristic_params_t stock;  // medium shapes revert to stock Float behavior

    auto nt_key = []() {
      heuristic_key_t k;
      k.arch        = hardware_t::architecture_t::gfx1100;
      k.mi_dtype    = data_type_t::Float;
      k.a_transpose = transpose_t::N;  // NT
      k.b_transpose = transpose_t::T;
      return k;
    };
    add_entry(nt_key(), p);  // default opt

    // Medium = max(M,N,K) in [2048, 8192). Approximate with per-dim bands that
    // union to "largest dim in [2048,8192)" while excluding small (<2048) and
    // large (>=8192). A dim in [2048,8191] AND nothing >= 8192 is the medium
    // band; we revert those to stock. Use min/max per dim; overlap is harmless.
    {
      auto k = nt_key(); k.min_m = 2048; k.max_m = 8191; add_entry(k, stock);
    }
    {
      auto k = nt_key(); k.min_n = 2048; k.max_n = 8191; add_entry(k, stock);
    }
    {
      auto k = nt_key(); k.min_k = 2048; k.max_k = 8191; add_entry(k, stock);
    }

    // Large (any dim >= 8192): dedicated large-tuned set (beats the opt default
    // on large). 5-fold CV mean +0.29 (5/5 folds); held-out test large top-1
    // 21%->27%, top-5 55%->58%, concordance 67.6%->68.6%. Higher specificity
    // than the medium [2048,8191] keys, so it wins for the >=8192 band.
    heuristic_params_t Lnt;
    Lnt.weight_compute                  = 0.200000;
    Lnt.weight_memory                   = 0.765162;
    Lnt.weight_prologue                 = 3.000000;
    Lnt.weight_epilogue                 = 0.367286;
    Lnt.weight_loop_overhead            = 1000.000000;
    Lnt.weight_tile_total               = 0.910252;
    Lnt.weight_mem_l2                   = 1.830131;
    Lnt.weight_mem_mall                 = 0.427080;
    Lnt.weight_mem_dram                 = 1.222174;
    Lnt.weight_wg_setup                 = 0.000000;
    Lnt.main_memory_load_latency        = 274.346992;
    Lnt.occupancy_decay_base            = 0.931033;
    Lnt.epilogue_k_padding_penalty      = 100000.000000;
    Lnt.epilogue_salu_overhead          = 100.000000;
    Lnt.epilogue_l_smem                 = 94.366803;
    Lnt.epilogue_cycles_per_acc_read    = 18.938906;
    Lnt.vgpr_penalty_weight             = 0.091201;
    Lnt.vgpr_per_simd                   = 1493.463039;
    Lnt.vgpr_overhead                   = 44.623100;
    { auto k = nt_key(); k.min_m = 8192; add_entry(k, Lnt); }
    { auto k = nt_key(); k.min_n = 8192; add_entry(k, Lnt); }
    { auto k = nt_key(); k.min_k = 8192; add_entry(k, Lnt); }
  }

  // ========================================================================
  // HEURISTIC 0e: gfx1100 bf16 (BFloat16) TN tuned heuristics
  // GA-tuned (rank-focused) vs measured bf16 TN winners (7942 shapes). Held-out
  // test (vs untuned): top-1 8.2%->12.5%, top-5 35.7%->45.3%, top-10
  // 55.4%->65.3%, no per-category regression. Applied to degenerate+small+
  // medium; large keeps stock (tuned set was high-variance / regressed there,
  // same pattern as HHS/FP32). bf16 uses the WMMA path (MI 16x16x16).
  // ========================================================================
  {
    heuristic_params_t p;
    p.weight_compute                  = 1.915326;
    p.weight_memory                   = 2.567097;
    p.weight_prologue                 = 8.000000;
    p.weight_epilogue                 = 12.000000;
    p.weight_loop_overhead            = 31.178772;
    p.weight_tile_total               = 0.323618;
    p.weight_mem_l2                   = 1.704287;
    p.weight_mem_mall                 = 0.084943;
    p.weight_mem_dram                 = 1.473683;
    p.weight_wg_setup                 = 10.000000;
    p.main_memory_load_latency        = 280.118853;
    p.occupancy_decay_base            = 0.718153;
    p.epilogue_k_padding_penalty      = 0.000000;
    p.epilogue_salu_overhead          = 27.742258;
    p.epilogue_l_smem                 = 215.034220;
    p.epilogue_cycles_per_acc_read    = 36.941965;
    p.vgpr_penalty_weight             = 0.949166;
    p.vgpr_per_simd                   = 350.890620;
    p.vgpr_overhead                   = 125.211991;

    auto bf16_key = []() {
      heuristic_key_t k;
      k.arch        = hardware_t::architecture_t::gfx1100;
      k.mi_dtype    = data_type_t::BFloat16;
      k.a_transpose = transpose_t::T;  // TN
      k.b_transpose = transpose_t::N;
      return k;
    };
    add_entry(bf16_key(), p);  // default opt (degenerate+small+medium)

    // Large (any dim >= 8192): dedicated large-tuned set. bf16 large was
    // previously stock (untuned). 5-fold CV mean +0.81 (5/5 folds); held-out
    // test large top-1 6%->14%, top-5 46%->57%, top-10 66%->94% — the single
    // largest large-class gain in the campaign (bf16 large had no prior tune).
    heuristic_params_t Lb;
    Lb.weight_compute                  = 0.451134;
    Lb.weight_memory                   = 0.306237;
    Lb.weight_prologue                 = 2.787278;
    Lb.weight_epilogue                 = 3.830711;
    Lb.weight_loop_overhead            = 12.138367;
    Lb.weight_tile_total               = 0.706003;
    Lb.weight_mem_l2                   = 3.283460;
    Lb.weight_mem_mall                 = 1.314143;
    Lb.weight_mem_dram                 = 1.448698;
    Lb.weight_wg_setup                 = 10.000000;
    Lb.main_memory_load_latency        = 400.000000;
    Lb.occupancy_decay_base            = 1.000000;
    Lb.epilogue_k_padding_penalty      = 0.000000;
    Lb.epilogue_salu_overhead          = 0.000000;
    Lb.epilogue_l_smem                 = 882.391133;
    Lb.epilogue_cycles_per_acc_read    = 7.531706;
    Lb.vgpr_penalty_weight             = 1.652957;
    Lb.vgpr_per_simd                   = 380.064732;
    Lb.vgpr_overhead                   = 128.000000;
    { auto k = bf16_key(); k.min_m = 8192; add_entry(k, Lb); }
    { auto k = bf16_key(); k.min_n = 8192; add_entry(k, Lb); }
    { auto k = bf16_key(); k.min_k = 8192; add_entry(k, Lb); }
  }

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
  // HEURISTIC 2: CMS Kernel Efficiencies
  // ========================================================================
  register_cms_kernels(*this);

  // ========================================================================
  // HEURISTIC 3: Reject gfx950 BF16 TN subtile kernels for small K
  // ========================================================================
  // Subtile kernels are not competitive when the reduction dimension is small
  // (K < 512). Scoped to gfx950 BF16 TN (a_transpose=T, b_transpose=N).
  {
    heuristic_params_t reject_params;
    reject_params.reject = true;

    // K < 512
    heuristic_key_t key;
    key.arch        = hardware_t::architecture_t::gfx950;
    key.mi_dtype    = data_type_t::BFloat16;
    key.a_transpose = transpose_t::T;
    key.b_transpose = transpose_t::N;
    key.subtile     = true;
    key.max_k       = 511;
    add_entry(key, reject_params);
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
