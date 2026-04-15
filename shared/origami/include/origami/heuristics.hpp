/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
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

#include <functional>
#include <optional>
#include <string_view>
#include <unordered_map>

#include "origami/hardware.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

namespace origami {
/**
 * @brief Default values for heuristic parameters.
 * Centralized location for all default constants.
 */
struct heuristic_defaults_t {
  // Latency Component Weights
  static constexpr double WEIGHT_MEM_L2        = 1.0;
  static constexpr double WEIGHT_MEM_MALL      = 1.0;
  static constexpr double WEIGHT_MEM_DRAM      = 1.0;
  static constexpr double WEIGHT_COMPUTE       = 1.0;
  static constexpr double WEIGHT_MEMORY        = 1.0;
  static constexpr double WEIGHT_WG_SETUP      = 1.0;
  static constexpr double WEIGHT_PROLOGUE      = 1.5;
  static constexpr double WEIGHT_EPILOGUE      = 2.0;
  static constexpr double WEIGHT_LOOP_OVERHEAD = 500.0;
  static constexpr double WEIGHT_TILE_TOTAL    = 1.0;

  // Empirical Constants
  static constexpr double MAIN_MEMORY_LOAD_LATENCY         = 200.0;
  static constexpr double OCCUPANCY_DECAY_BASE             = 0.95;
  static constexpr double MALL_DEPTH_SQ                    = 2.0;
  static constexpr double MALL_COLD_FLOOR                  = 0.85;
  static constexpr double L2_DEPTH_SQ                      = 4.0;
  static constexpr double L2_COLD_FLOOR                    = 0.75;
  static constexpr double L2_POLLUTION_PENALTY             = 0.7;
  static constexpr double L2_AMP_CEILING_BATCHED           = 0.9;
  static constexpr double L2_AMP_CEILING_K_SPLIT           = 0.4;
  static constexpr double L2_AMP_CEILING_SKINNY            = 0.6;
  static constexpr double L2_DEPTH_PENALTY                 = 0.9;
  static constexpr double L1_HIT_RATE_CEILING_SKINNY       = 0.7;
  static constexpr double EPILOGUE_CYCLES_PER_ACC_READ     = 8.0;
  static constexpr double EPILOGUE_ACC_READ_PARALLELISM    = 0.9;
  static constexpr double EPILOGUE_CYCLES_PER_BOUNDS_CHECK = 6.0;
  static constexpr double EPILOGUE_SCALAR_STORE_PENALTY    = 1.1;
  static constexpr size_t EPILOGUE_THREADS_PER_WAVE        = 64;
  static constexpr size_t EPILOGUE_BYTES_PER_VECTORIZED_STORE = 16;  // buffer_store_dwordx4
  static constexpr size_t EPILOGUE_CACHE_LINE_BYTES         = 128;
  static constexpr size_t EPILOGUE_WORKSPACE_BYTES_PER_ELEM = 4;
  static constexpr double EPILOGUE_SALU_OVERHEAD            = 35.0;
  static constexpr double EPILOGUE_L_BARRIER                = 100.0;
  static constexpr double EPILOGUE_L_SMEM = 900.0;  // s_load_dword(glc) cross-XCD flag poll
  static constexpr double EPILOGUE_K_PADDING_PENALTY     = 50000.0;
  static constexpr size_t POSTGSU_COMPUTE_BYTES          = 4;  // workspace partials stored as f32
  static constexpr double POSTGSU_KERNEL_LAUNCH_OVERHEAD = 12000.0;
  static constexpr size_t POSTGSU_THREADS_PER_WG         = 256;
  static constexpr size_t POSTGSU_WAVEFRONT_SIZE         = 64;

  // Main Loop Efficiency
  static constexpr double MAIN_LOOP_EFFICIENCY = 1.0;

  // TF32 Emulation Constants
  static constexpr double TF32_ARITH_INTENSITY_THRESHOLD = 1000.0;
};

/**
 * @brief Structure containing all trainable heuristic parameters.
 *
 * This structure consolidates all empirical constants and weights used in
 * the latency model, making them trainable and configuration-driven.
 */
struct heuristic_params_t {
  // === Latency Component Weights ===
  double weight_mem_l2        = heuristic_defaults_t::WEIGHT_MEM_L2;
  double weight_mem_mall      = heuristic_defaults_t::WEIGHT_MEM_MALL;
  double weight_mem_dram      = heuristic_defaults_t::WEIGHT_MEM_DRAM;
  double weight_compute       = heuristic_defaults_t::WEIGHT_COMPUTE;
  double weight_memory        = heuristic_defaults_t::WEIGHT_MEMORY;
  double weight_wg_setup      = heuristic_defaults_t::WEIGHT_WG_SETUP;
  double weight_prologue      = heuristic_defaults_t::WEIGHT_PROLOGUE;
  double weight_epilogue      = heuristic_defaults_t::WEIGHT_EPILOGUE;
  double weight_loop_overhead = heuristic_defaults_t::WEIGHT_LOOP_OVERHEAD;
  double weight_tile_total    = heuristic_defaults_t::WEIGHT_TILE_TOTAL;

  // === Empirical Constants ===
  double main_memory_load_latency         = heuristic_defaults_t::MAIN_MEMORY_LOAD_LATENCY;
  double occupancy_decay_base             = heuristic_defaults_t::OCCUPANCY_DECAY_BASE;
  double mall_depth_sq                    = heuristic_defaults_t::MALL_DEPTH_SQ;
  double mall_cold_floor                  = heuristic_defaults_t::MALL_COLD_FLOOR;
  double l2_depth_sq                      = heuristic_defaults_t::L2_DEPTH_SQ;
  double l2_cold_floor                    = heuristic_defaults_t::L2_COLD_FLOOR;
  double l2_pollution_penalty             = heuristic_defaults_t::L2_POLLUTION_PENALTY;
  double l2_amp_ceiling_batched           = heuristic_defaults_t::L2_AMP_CEILING_BATCHED;
  double l2_amp_ceiling_k_split           = heuristic_defaults_t::L2_AMP_CEILING_K_SPLIT;
  double l2_amp_ceiling_skinny            = heuristic_defaults_t::L2_AMP_CEILING_SKINNY;
  double l2_depth_penalty                 = heuristic_defaults_t::L2_DEPTH_PENALTY;
  double l1_hit_rate_ceiling_skinny       = heuristic_defaults_t::L1_HIT_RATE_CEILING_SKINNY;
  double epilogue_cycles_per_acc_read     = heuristic_defaults_t::EPILOGUE_CYCLES_PER_ACC_READ;
  double epilogue_acc_read_parallelism    = heuristic_defaults_t::EPILOGUE_ACC_READ_PARALLELISM;
  double epilogue_cycles_per_bounds_check = heuristic_defaults_t::EPILOGUE_CYCLES_PER_BOUNDS_CHECK;
  double epilogue_scalar_store_penalty    = heuristic_defaults_t::EPILOGUE_SCALAR_STORE_PENALTY;
  size_t epilogue_threads_per_wave        = heuristic_defaults_t::EPILOGUE_THREADS_PER_WAVE;
  size_t epilogue_bytes_per_vectorized_store =
      heuristic_defaults_t::EPILOGUE_BYTES_PER_VECTORIZED_STORE;
  size_t epilogue_cache_line_bytes = heuristic_defaults_t::EPILOGUE_CACHE_LINE_BYTES;
  size_t epilogue_workspace_bytes_per_elem =
      heuristic_defaults_t::EPILOGUE_WORKSPACE_BYTES_PER_ELEM;
  double epilogue_salu_overhead         = heuristic_defaults_t::EPILOGUE_SALU_OVERHEAD;
  double epilogue_l_barrier             = heuristic_defaults_t::EPILOGUE_L_BARRIER;
  double epilogue_l_smem                = heuristic_defaults_t::EPILOGUE_L_SMEM;
  double epilogue_k_padding_penalty     = heuristic_defaults_t::EPILOGUE_K_PADDING_PENALTY;
  size_t postgsu_compute_bytes          = heuristic_defaults_t::POSTGSU_COMPUTE_BYTES;
  double postgsu_kernel_launch_overhead = heuristic_defaults_t::POSTGSU_KERNEL_LAUNCH_OVERHEAD;
  size_t postgsu_threads_per_wg         = heuristic_defaults_t::POSTGSU_THREADS_PER_WG;
  size_t postgsu_wavefront_size         = heuristic_defaults_t::POSTGSU_WAVEFRONT_SIZE;

  // === Main Loop Efficiency ===
  double main_loop_efficiency = heuristic_defaults_t::MAIN_LOOP_EFFICIENCY;

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
struct heuristic_key_t {
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
class heuristics_database_t {
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
 * @brief Helper to create a key for hand-optimized kernels
 */
heuristic_key_t make_hand_optimized_kernel_key(hardware_t::architecture_t arch,
                                               data_type_t mi_dtype,
                                               transpose_t transA,
                                               transpose_t transB,
                                               size_t MT_M,
                                               size_t MT_N,
                                               size_t MT_K);

/**
 * @brief Helper to create a key for tile configuration.
 */
heuristic_key_t make_tile_key(size_t MT_M,
                              size_t MT_N,
                              size_t MT_K,
                              std::optional<transpose_t> transA = std::nullopt,
                              std::optional<transpose_t> transB = std::nullopt);

/**
 * @brief Helper to create a key for architecture/datatype combination.
 */
heuristic_key_t make_arch_dtype_key(hardware_t::architecture_t arch, data_type_t mi_dtype);

}  // namespace origami
