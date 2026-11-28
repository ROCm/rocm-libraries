// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <optional>
#include <string_view>
#include <unordered_map>

#include "origami/cost_models/gemm/constants.hpp"
#include "origami/hardware.hpp"
#include "origami/math.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

namespace origami {
/**
 * @brief Structure containing all trainable heuristic parameters.
 *
 * This structure consolidates all empirical constants and weights used in
 * the latency model, making them trainable and configuration-driven.
 */
struct ORIGAMI_EXPORT heuristic_params_t {
  // === Latency Component Weights ===
  double weight_tile_total = heuristic_defaults_t::WEIGHT_TILE_TOTAL;

  // === Main Loop Efficiency ===
  double main_loop_efficiency = heuristic_defaults_t::MAIN_LOOP_EFFICIENCY;

  // === Kernel Rejection ===
  /// When true, the kernel is rejected: its predicted latency is forced to the
  /// maximum so that rank_configs() drops it from selection entirely.
  bool reject = false;

  /**
   * @brief Merge this parameter set with another (for hierarchical lookup).
   * Only non-default values from 'other' override values in 'this'.
   */
  void merge_with(const heuristic_params_t& other);
};

/**
 * @brief Key structure for looking up heuristics in the unified map.
 *
 * This key captures all relevant characteristics that might affect
 * heuristic parameter selection. Fields can be wildcards (using optional).
 */
struct ORIGAMI_EXPORT heuristic_key_t {
  std::optional<hardware_t::architecture_t> arch;
  std::optional<data_type_t> a_dtype;
  std::optional<data_type_t> b_dtype;
  std::optional<data_type_t> mi_dtype;
  std::optional<transpose_t> a_transpose;
  std::optional<transpose_t> b_transpose;
  std::optional<size_t> mt_m;
  std::optional<size_t> mt_n;
  std::optional<size_t> mt_k;
  std::optional<bool> hand_optimized_main_loop;
  std::optional<bool> subtile;

  // For problem-size dependent heuristics
  std::optional<size_t> min_m;
  std::optional<size_t> max_m;
  std::optional<size_t> min_n;
  std::optional<size_t> max_n;
  std::optional<size_t> min_k;
  std::optional<size_t> max_k;

  /**
   * @brief Check if this key matches the given problem/hardware/config.
   * @return true if all specified fields match (wildcards match everything)
   */
  bool matches(const problem_t& problem, const hardware_t& hardware, const config_t& config) const;

  /**
   * @brief Get specificity score (number of non-wildcard fields).
   * Used for prioritizing more specific matches over general ones.
   */
  size_t specificity() const;
};

/**
 * @brief Key for hand-optimized kernels.
 *
 * Hand-optimized kernels such as CMS(Custom Main-loop Scheduling) kernels have fully specified
 * characteristics (arch, dtype, layout, MT sizes).
 */
struct hand_optimized_kernel_key_t {
  hardware_t::architecture_t arch;
  data_type_t mi_dtype;
  transpose_t a_transpose;
  transpose_t b_transpose;
  size_t mt_m;
  size_t mt_n;
  size_t mt_k;

  std::string to_string() const {
    return std::string(hardware_t::arch_enum_to_name(arch)) + "_" +
           origami::datatype_to_string(mi_dtype) + "_" +
           (a_transpose == transpose_t::T ? "T" : "N") +
           (b_transpose == transpose_t::T ? "T" : "N") + "_" + std::to_string(mt_m) + "x" +
           std::to_string(mt_n) + "x" + std::to_string(mt_k);
  }

  bool operator==(const hand_optimized_kernel_key_t& other) const {
    return arch == other.arch && mi_dtype == other.mi_dtype && a_transpose == other.a_transpose &&
           b_transpose == other.b_transpose && mt_m == other.mt_m && mt_n == other.mt_n &&
           mt_k == other.mt_k;
  }

  std::size_t hash() const {
    return math::hash_combine(static_cast<int>(arch),
                              static_cast<int>(mi_dtype),
                              static_cast<int>(a_transpose),
                              static_cast<int>(b_transpose),
                              mt_m,
                              mt_n,
                              mt_k);
  }
};

struct hand_optimized_kernel_key_hash {
  std::size_t operator()(const hand_optimized_kernel_key_t& k) const { return k.hash(); }
};

/**
 * @brief Unified heuristics database.
 *
 * This is the single source of truth for all heuristic parameters.
 * Maps from heuristic keys (problem characteristics) to parameter sets.
 *
 * @note Thread Safety:
 * - Singleton initialization is thread-safe (C++11 magic statics)
 * - lookup() is thread-safe for concurrent reads (const method)
 * - add_entry() is NOT thread-safe and should only be called during
 *   initialization (from constructor) before any concurrent access
 * - Do NOT call add_entry() after the singleton is initialized and
 *   multiple threads may be accessing the database
 */
class ORIGAMI_EXPORT heuristics_database_t {
 public:
  /**
   * @brief Lookup heuristic parameters for given problem/hardware/config.
   *
   * Performs hierarchical lookup:
   * 1. For hand-optimized kernels: O(1) hash map lookup
   * 2. For general heuristics: Linear search with specificity ordering
   * 3. Falls back to default parameters
   *
   * @param problem Problem definition
   * @param hardware Hardware characteristics
   * @param config Kernel configuration
   * @return heuristic_params_t Merged parameter set
   */
  heuristic_params_t lookup(const problem_t& problem,
                            const hardware_t& hardware,
                            const config_t& config) const;

  /**
   * @brief Add or update a heuristic entry.
   */
  void add_entry(const heuristic_key_t& key, const heuristic_params_t& params);

  /**
   * @brief Add or update a hand-optimized kernel efficiency entry.
   *
   * Writes directly to the fast lookup map. Intended for initialization only.
   */
  void add_hand_optimized_efficiency(hand_optimized_kernel_key_t key, double main_loop_efficiency);

  /**
   * @brief Return true if the database has a hand-optimized entry for the given (arch, dtype,
   * layout, MT).
   */
  bool has_hand_optimized_entry(hardware_t::architecture_t arch,
                                data_type_t mi_dtype,
                                transpose_t transA,
                                transpose_t transB,
                                size_t mt_m,
                                size_t mt_n,
                                size_t mt_k) const;

  /**
   * @brief Get the global heuristics database instance.
   */
  static heuristics_database_t& get_instance();

 private:
  heuristics_database_t();  // Private constructor for singleton

  // General heuristics storage (linear search)
  std::vector<std::pair<heuristic_key_t, heuristic_params_t>> entries_;

  // unordered_map for hand-optimized kernels
  std::
      unordered_map<hand_optimized_kernel_key_t, heuristic_params_t, hand_optimized_kernel_key_hash>
          hand_optimized_map_;

  heuristic_params_t default_params_;

  // Initialize with default heuristics
  void initialize_defaults();
};

/**
 * @brief Convenience function to get heuristic parameters.
 *
 * This is the main entry point that replaces compute_heuristic_weights().
 */
inline heuristic_params_t get_heuristic_params(const problem_t& problem,
                                               const hardware_t& hardware,
                                               const config_t& config) {
  return heuristics_database_t::get_instance().lookup(problem, hardware, config);
}

/**
 * @brief Whether the W4A16 model applies to @p problem on @p hardware.
 *
 * The latency model and the Stream-K grid rule both use it, so they apply together.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @return bool True for an int4 A with a bf16/fp16 MI type on gfx1150 and gfx1151.
 */
inline bool is_w4a16(const problem_t& problem, const hardware_t& hardware) {
  if (problem.a_dtype != data_type_t::Int4) return false;
  if (problem.mi_dtype != data_type_t::BFloat16 && problem.mi_dtype != data_type_t::Half)
    return false;
  switch (hardware.arch) {
    case hardware_t::architecture_t::gfx1150:
    case hardware_t::architecture_t::gfx1151: return true;
    default: return false;
  }
}

/**
 * @brief Peak DRAM read bandwidth of the device for the W4A16 decode model.
 *
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @return std::optional<double> Bandwidth in bytes/us (measured), or nullopt where the device has
 * no decode model.
 */
// TODO(AIESW-51613): these values should come from hardware_t (mem3_perf_ratio is the DRAM
// bandwidth), not from here.  The values hardware_t holds today are wrong: about 623 GB/s on
// gfx1151 and 31 GB/s on gfx1150, against the 247 and 84.1 GB/s measured.  The tile model is
// calibrated with them (it underestimates the memory latency and gfx1150's value compensates
// for that), so the Origami model has to be fixed before they can change.  Until then the
// decode model keeps the measured bandwidths here.
inline std::optional<double> w4a16_decode_dram_bytes_per_us(const hardware_t& hardware) {
  switch (hardware.arch) {
    case hardware_t::architecture_t::gfx1150: return 84.1e3;   // [MEASURED] Strix Point: 78.3 GiB/s
    case hardware_t::architecture_t::gfx1151: return 247.0e3;  // [MEASURED] Strix Halo: 230 GiB/s
    default: return std::nullopt;
  }
}

/**
 * @brief Helper to create a key for hand-optimized kernels
 */
ORIGAMI_EXPORT heuristic_key_t make_hand_optimized_kernel_key(hardware_t::architecture_t arch,
                                                              data_type_t mi_dtype,
                                                              transpose_t transA,
                                                              transpose_t transB,
                                                              size_t MT_M,
                                                              size_t MT_N,
                                                              size_t MT_K);

/**
 * @brief Helper to create a key for tile configuration.
 */
ORIGAMI_EXPORT heuristic_key_t make_tile_key(size_t MT_M,
                                             size_t MT_N,
                                             size_t MT_K,
                                             std::optional<transpose_t> transA = std::nullopt,
                                             std::optional<transpose_t> transB = std::nullopt);

/**
 * @brief Helper to create a key for architecture/datatype combination.
 */
ORIGAMI_EXPORT heuristic_key_t make_arch_dtype_key(hardware_t::architecture_t arch,
                                                   data_type_t mi_dtype);

}  // namespace origami
