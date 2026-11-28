// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

#include "model.hpp"

namespace origami {

// ============================================================================
// heuristic_params_t Implementation
// ============================================================================

void heuristic_params_t::merge_with(const heuristic_params_t& other) {
  // Latency component weights
  weight_tile_total = other.weight_tile_total;

  // Main loop efficiency
  main_loop_efficiency = other.main_loop_efficiency;

  // Kernel rejection
  reject = other.reject;
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
  size_t hand_optimized_kernels = 0;
  gemm::for_each_family(
      [&](auto family) { hand_optimized_kernels += family.hand_optimized_kernel_count(); });
  hand_optimized_map_.reserve(hand_optimized_kernels);
  initialize_defaults();
}

heuristics_database_t& heuristics_database_t::get_instance() {
  static heuristics_database_t instance;
  return instance;
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

  // Apply the family's rules that depend on the problem (e.g. on its arithmetic intensity)
  gemm::for_family(
      hardware, [&](auto family) { family.adjust_heuristics(result, problem, hardware, config); });

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
  gemm::for_each_family([&](auto family) { family.register_heuristics(*this); });
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
