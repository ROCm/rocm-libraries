/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include "origami/math.hpp"

namespace origami {

/**
 * @brief Enumeration of supported data types.
 *
 */
enum class data_type_t : int {
  Float,
  Double,
  ComplexFloat,
  ComplexDouble,
  Half,
  Int8x4,
  Int32,
  BFloat16,
  Int8,
  Int4,
  Int64,
  XFloat32,
  Float8_fnuz,
  BFloat8_fnuz,
  Float8BFloat8_fnuz,
  BFloat8Float8_fnuz,
  Float8,
  BFloat8,
  Float8BFloat8,
  BFloat8Float8,
  Float6,
  BFloat6,
  Float4,
  Count,
  None = Count
};

/**
 * @brief Convert integer to data_type_t enum.
 *
 * @param dt Integer value to convert
 * @return data_type_t Corresponding data type
 */
inline data_type_t int_to_data_type(int dt) { return static_cast<data_type_t>(dt); }

/**
 * @brief Convert data_type_t to number of bits.
 *
 * @param type Data type
 * @return int Number of bits
 */
int datatype_to_bits(data_type_t type);

/**
 * @brief Convert data_type_t to number of bytes.
 *
 * @param type Data type
 * @return int Number of bytes
 */
inline double data_type_to_bytes(data_type_t type) {
  return static_cast<double>(datatype_to_bits(type)) / 8.0;
}

/**
 * @brief Convert data_type_t to string.
 *
 * @param type Data type
 * @return std::string String representation of data type
 */
std::string datatype_to_string(data_type_t type);

/**
 * @brief Convert string to data_type_t enum.
 *
 * @param s String value to convert
 * @return data_type_t Corresponding data type
 */
data_type_t string_to_datatype(std::string s);

/**
 * @brief Struct to define a matrix instruction.
 *
 * Contains the dimensions and data type of a matrix instruction.
 */
struct matrix_instruction {
  size_t MI_M;
  size_t MI_N;
  size_t MI_K;
  data_type_t mi_input_type;

  matrix_instruction() : MI_M(0), MI_N(0), MI_K(0), mi_input_type(data_type_t::Float) {}

  matrix_instruction(size_t m, size_t n, size_t k, data_type_t mi_input_type)
      : MI_M(m), MI_N(n), MI_K(k), mi_input_type(mi_input_type) {}

  matrix_instruction(const matrix_instruction& other)
      : MI_M(other.MI_M), MI_N(other.MI_N), MI_K(other.MI_K), mi_input_type(other.mi_input_type) {}

  bool operator<(const matrix_instruction& other) const {
    return std::tie(MI_M, MI_N, MI_K, mi_input_type) <
           std::tie(other.MI_M, other.MI_N, other.MI_K, other.mi_input_type);
  }

  bool operator==(const matrix_instruction& other) const {
    return MI_M == other.MI_M && MI_N == other.MI_N && MI_K == other.MI_K &&
           mi_input_type == other.mi_input_type;
  }

  std::size_t hash() const {
    return std::hash<size_t>()(MI_M) ^ std::hash<size_t>()(MI_N) ^ std::hash<size_t>()(MI_K) ^
           std::hash<data_type_t>()(mi_input_type);
  }
};

/**
 * @brief Grid selection algorithms for StreamK.
 *
 * Different algorithms to select the grid size for kernel execution.
 */
enum class grid_selection_t : std::uint32_t {
  number_of_cus        = 0,  ///< Use number of compute units
  min_resources        = 1,  ///< Use minimum required resources
  energy_aware         = 2,  ///< Energy-aware selection
  reduction_cost_aware = 3,  ///< Reduction cost-aware selection
  data_parallel        = 4,  ///< Data parallel approach
  analytical           = 5,  ///< Analytical model-based selection
  k_split_aware        = 6,  ///< K-split aware selection
  count,                     ///< Count of Grid selection algos
  none = 0xFFFFFFFFu         ///< Explicitly invalid
};

/**
 * @brief Reduction strategy types for StreamK.
 *
 * Different algorithms for reduction operations in StreamK.
 */
enum class reduction_t : std::uint32_t {
  spinlock = 0,       ///< Spinlock-based reduction
  tree     = 1,       ///< Tree-based reduction
  parallel = 2,       ///< Parallel reduction
  atomic   = 3,       ///< Atomic Add-based reduction
  count,              ///< Count of reduction types
  none = 0xFFFFFFFFu  ///< Explicitly invalid / no reduction
};

/**
 * @brief Prediction mode types for latency estimation.
 *
 * Different approaches for predicting kernel performance.
 */
enum class prediction_modes_t : std::uint32_t {
  estimation = 0,     ///< Fast analytical estimation-based prediction (typically faster)
  simulation = 1,     ///< Slow simulation-like prediction (typically more accurate)
  ml_recommender = 2, ///< ML-based tile prediction (per-cluster two-tower)
  count,              ///< Count of prediction modes
  none = 0xFFFFFFFFu  ///< Explicitly invalid
};

/**
 * @brief Target backend types for kernel execution.
 *
 * Different backends that kernels can target.
 */
enum class target_t : std::uint32_t {
  generic           = 0,  ///< Generic backend (backend agnostic, not supported yet)
  tensilelite       = 1,  ///< hipBLASLt (tensilelite) backend
  rocroller         = 2,  ///< hipBLASLt (rocroller) backend
  triton            = 3,  ///< Triton backend
  composable_kernel = 4,  ///< Composable Kernel backend (Not supported yet)
  count,                  ///< Count of target types
  none = 0xFFFFFFFFu      ///< Explicitly invalid
};

/**
 * @brief Convert integer to reduction_t enum.
 *
 * @param rt Integer value to convert
 * @return reduction_t Corresponding reduction type
 */
inline constexpr reduction_t int_to_reduction_t(int rt) { return static_cast<reduction_t>(rt); }

/**
 * @brief Indicates whether a matrix is supplied in transposed or not.
 */
enum class transpose_t {
  T,
  N,

  Count
};

/**
 * @brief A compact 3-D dimension triple (M, N, K).
 *
 * Provides convenient accessors for common GEMM tiling parameters
 * and helpers like mnk() for volume.
 */
struct dim3_t {
  /// M dimension (rows).
  std::size_t m;

  /// N dimension (columns).
  std::size_t n;

  /// K dimension (reduction).
  std::size_t k;

  constexpr bool operator==(const dim3_t& o) const noexcept {
    return m == o.m && n == o.n && k == o.k;
  }

  constexpr bool operator!=(const dim3_t& o) const noexcept { return !(*this == o); }

  /// @return Product m*n.
  constexpr std::size_t mn() const noexcept { return m * n; }

  /// @return Product m*k.
  constexpr std::size_t mk() const noexcept { return m * k; }

  /// @return Product n*k.
  constexpr std::size_t nk() const noexcept { return n * k; }

  /// @return Product m*n*k.
  constexpr std::size_t mnk() const noexcept { return m * n * k; }
};

/**
 * @brief 4-dimensional size/coordinate: (k, m, n, b).
 *
 * Used for tile coordinates and unique tile counts across the GEMM grid.
 */
struct dim4_t {
  /// K dimension (reduction / split).
  std::size_t k = 0;

  /// M dimension (rows).
  std::size_t m = 0;

  /// N dimension (columns).
  std::size_t n = 0;

  /// B dimension (batch).
  std::size_t b = 0;

  constexpr bool operator==(const dim4_t& o) const noexcept {
    return k == o.k && m == o.m && n == o.n && b == o.b;
  }

  constexpr bool operator!=(const dim4_t& o) const noexcept { return !(*this == o); }

  /// @return Product m*n.
  constexpr std::size_t mn() const noexcept { return m * n; }

  /// @return Product m*n*k.
  constexpr std::size_t mnk() const noexcept { return m * n * k; }

  /// @return Product k*m*n*b.
  constexpr std::size_t total() const noexcept { return k * m * n * b; }
};

/**
 * @brief Runtime options for controlling debug, heuristics, and other behaviors.
 *
 * Provides programmatic access to runtime configuration options that can be
 * set either programmatically or via environment variables.
 */
struct runtime_options {
  /// Enable debug logging (reads from ANALYTICAL_GEMM_DEBUG env var)
  bool debug_enabled;

  /// Enable heuristics (reads from ANALYTICAL_GEMM_HEURISTICS env var)
  bool heuristics_enabled;

  /// Heuristics variance threshold (reads from ANALYTICAL_GEMM_HEURISTICS_VARIANCE env var)
  double heuristics_variance;

  /**
   * @brief Constructor with explicit values (does not read from environment).
   */
  runtime_options(bool debug, bool heuristics, double variance);

  /**
   * @brief Get the global runtime options instance.
   *
   * Inline to prevent ODR violations when included in multiple shared libraries.
   * Static local variable ensures only one instance exists across all translation units. (PR#1862)
   */
  static inline runtime_options& get() {
    static runtime_options instance;
    return instance;
  }

  /**
   * @brief Read debug setting from environment variable.
   * @return true if ANALYTICAL_GEMM_DEBUG is set to "1", false otherwise
   */
  static bool read_debug_from_env();

  /**
   * @brief Read heuristics setting from environment variable.
   * @return true if ANALYTICAL_GEMM_HEURISTICS is set to "1", false otherwise
   */
  static bool read_heuristics_from_env();

  /**
   * @brief Read heuristics variance from environment variable.
   * @return double Variance value from ANALYTICAL_GEMM_HEURISTICS_VARIANCE, or 0.01 if not set
   */
  static double read_heuristics_variance_from_env();

  /**
   * @brief Update runtime options from environment variables.
   */
  void update_from_env();

 private:
  /**
   * @brief Default constructor that reads from environment variables.
   *
   * This is made private because it should only be used through the static get() member.
   */
  runtime_options();
};

/**
 * @brief Tensile/TensileLite-specific configuration parameters.
 *
 * Contains parameters specific to TensileLite-generated GEMM kernels,
 * used by the Formocast simulation model. These parameters are ignored
 * by the estimation-based prediction model.
 */
struct tensile_params_t {
  /// Depth unroll factor (0 = use mt.k)
  std::size_t depth_u = 0;

  /// Global split-K factor
  std::int16_t global_split_u = 1;

  /// GSU accumulation method (0=none, 2=MultiBuffer, 3=MultiBufferSingleKernel)
  int global_accumulation = 0;

  /// Local split-K factor
  int local_split_u = 1;

  /// DirectToVGPR flags - bypass LDS for register file
  bool direct_to_vgpr_a = false;
  bool direct_to_vgpr_b = false;

  /// DirectToLDS flags - direct global memory to LDS
  bool direct_to_lds_a = false;
  bool direct_to_lds_b = false;

  /// Number of loads that can be coalesced
  int num_loads_coalesced_a = 1;
  int num_loads_coalesced_b = 1;

  /// Number of waves per workgroup
  std::size_t wave_num = 4;

  /// Wave group dimensions [wave_group_m, wave_group_n]
  int wave_group_m = 2;
  int wave_group_n = 2;

  /// Prefetch global read depth
  int prefetch_global_read = 2;

  /// Math clocks per unrolled loop iteration (0 = auto-calculate)
  int math_clocks_unrolled_loop = 0;

  /// Swizzled memory layout flags
  bool swizzle_a = false;
  bool swizzle_b = false;

  /// Workgroup mapping XCC parameters
  int workgroup_mapping_xcc = 1;
  int workgroup_mapping_xcc_group = 0;
  bool global_split_u_coalesced = false;
  bool global_split_u_wgm_round_robin = false;

  constexpr bool operator==(const tensile_params_t& o) const noexcept {
    return depth_u == o.depth_u && global_split_u == o.global_split_u &&
           global_accumulation == o.global_accumulation && local_split_u == o.local_split_u &&
           direct_to_vgpr_a == o.direct_to_vgpr_a && direct_to_vgpr_b == o.direct_to_vgpr_b &&
           direct_to_lds_a == o.direct_to_lds_a && direct_to_lds_b == o.direct_to_lds_b &&
           num_loads_coalesced_a == o.num_loads_coalesced_a &&
           num_loads_coalesced_b == o.num_loads_coalesced_b && wave_num == o.wave_num &&
           wave_group_m == o.wave_group_m && wave_group_n == o.wave_group_n &&
           prefetch_global_read == o.prefetch_global_read &&
           math_clocks_unrolled_loop == o.math_clocks_unrolled_loop && swizzle_a == o.swizzle_a &&
           swizzle_b == o.swizzle_b && workgroup_mapping_xcc == o.workgroup_mapping_xcc &&
           workgroup_mapping_xcc_group == o.workgroup_mapping_xcc_group &&
           global_split_u_coalesced == o.global_split_u_coalesced &&
           global_split_u_wgm_round_robin == o.global_split_u_wgm_round_robin;
  }

  std::size_t hash() const {
    return math::hash_combine(depth_u,
                              global_split_u,
                              global_accumulation,
                              local_split_u,
                              direct_to_vgpr_a,
                              direct_to_vgpr_b,
                              direct_to_lds_a,
                              direct_to_lds_b,
                              num_loads_coalesced_a,
                              num_loads_coalesced_b,
                              wave_num,
                              wave_group_m,
                              wave_group_n,
                              prefetch_global_read,
                              math_clocks_unrolled_loop,
                              swizzle_a,
                              swizzle_b,
                              workgroup_mapping_xcc,
                              workgroup_mapping_xcc_group,
                              global_split_u_coalesced,
                              global_split_u_wgm_round_robin);
  }
};

/// Variant holding backend-specific parameters.
/// std::monostate represents no backend-specific params (default).
using backend_params_t = std::variant<std::monostate, tensile_params_t>;

/**
 * @brief Full kernel configuration (tile shape + execution parameters).
 *
 * Holds the geometric tile sizes along with occupancy,
 * work-group mapping (WGM), and cache-control hints.
 */
struct config_t {
  /// Macro tile and matrix-instruction shape.
  dim3_t mt{0, 0, 0};
  dim3_t mi{0, 0, 0};

  /// Main loop optimization flag (indicates use of any optimized kernel variant)
  bool hand_optimized_main_loop = false;

  /// Occupancy (number of wavefronts resident per CU).
  int occupancy = -1;

  /// Reorder workgroup id for L2 reuse.
  int workgroup_mapping = 0;

  /// Whether operand A is accessed with cache-flags.
  int cache_hints_a = 0;

  /// Whether operand B is accessed with cache-flags.
  int cache_hints_b = 0;

  /// Workspace size parameters.
  std::size_t workspace_size            = 0;
  std::size_t workspace_size_per_elem_c = 0;

  /// Reduction strategy.
  reduction_t reduction_strategy = reduction_t::none;

  /// Prediction mode for latency estimation.
  prediction_modes_t prediction_mode = prediction_modes_t::estimation;

  /// Target backend for kernel execution.
  target_t target = target_t::tensilelite;
  /// Grid selection algorithm.
  grid_selection_t grid_selection = grid_selection_t::k_split_aware;

  /// Index of config, not used by Origami but can be used by the user
  std::size_t index = 0;

  /// Global read vector width for matrix A (elements per load)
  std::size_t grvw_a = 1;

  /// Global read vector width for matrix B (elements per load)
  std::size_t grvw_b = 1;

  /// Global write vector width for matrix D (elements per store)
  std::size_t gwvw_d = 1;

  /// LDS load vector width for matrix A (elements per LDS read)
  int vector_width_a = 1;

  /// LDS load vector width for matrix B (elements per LDS read)
  int vector_width_b = 1;

  /// Backend-specific parameters (type should match target).
  /// Use tensile() accessor to get/set Tensile-specific params.
  backend_params_t backend{};

  /// Get mutable reference to Tensile params. Initializes if not already set.
  tensile_params_t& tensile() {
    if (!std::holds_alternative<tensile_params_t>(backend)) { backend = tensile_params_t{}; }
    return std::get<tensile_params_t>(backend);
  }

  /// Get const reference to Tensile params. Throws if not set.
  const tensile_params_t& tensile() const { return std::get<tensile_params_t>(backend); }

  /// Check if Tensile params are currently set.
  bool has_tensile_params() const noexcept {
    return std::holds_alternative<tensile_params_t>(backend);
  }

  bool operator==(const config_t& o) const noexcept {
    return mt == o.mt && mi == o.mi && hand_optimized_main_loop == o.hand_optimized_main_loop &&
           cache_hints_a == o.cache_hints_a && cache_hints_b == o.cache_hints_b &&
           workgroup_mapping == o.workgroup_mapping && reduction_strategy == o.reduction_strategy &&
           prediction_mode == o.prediction_mode && target == o.target && grvw_a == o.grvw_a &&
           grvw_b == o.grvw_b && gwvw_d == o.gwvw_d && vector_width_a == o.vector_width_a &&
           vector_width_b == o.vector_width_b && backend == o.backend;
  }

  std::size_t hash() const {
    std::size_t seed = math::hash_combine(mt.m,
                                          mt.n,
                                          mt.k,
                                          mi.m,
                                          mi.n,
                                          mi.k,
                                          hand_optimized_main_loop,
                                          cache_hints_a,
                                          cache_hints_b,
                                          workgroup_mapping,
                                          static_cast<std::uint32_t>(reduction_strategy),
                                          static_cast<std::uint32_t>(prediction_mode),
                                          static_cast<std::uint32_t>(target),
                                          grvw_a,
                                          grvw_b,
                                          gwvw_d,
                                          vector_width_a,
                                          vector_width_b);
    // Hash backend-specific parameters if present. The visitor pattern allows
    // automatic handling of any backend type that provides a hash() method,
    // while std::monostate (no backend params) is a no-op.
    std::visit(
        [&seed](const auto& params) {
          if constexpr (!std::is_same_v<std::decay_t<decltype(params)>, std::monostate>) {
            math::hash_combine(seed, params.hash());
          }
        },
        backend);
    return seed;
  }

  void validate() const {
    if (!is_valid()) { throw std::runtime_error("Invalid config_t"); }
  }

  bool is_valid() const {
    return mt.m > 0 && mt.n > 0 && mt.k > 0 && mi.m > 0 && mi.n > 0 && mi.k > 0 && occupancy > 0;
  }
};

/**
 * @brief Latency prediction result given kernel configuration.
 *
 * Combines a configuration with its estimated latency.
 */
struct prediction_result_t {
  double latency;
  config_t config;
};

/**
 * @brief Struct to define the GEMM problem characteristics.
 *
 * Contains all the parameters needed to describe a GEMM operation,
 * including matrix dimensions, data types, and operation flags.
 */
struct problem_t {
  /// Size of the problem: M, N, K.
  dim3_t size{0, 0, 0};

  /// Batch size.
  std::size_t batch = 1;

  /// Transpose types (TT, TN, NT, TT.)
  transpose_t a_transpose = transpose_t::N;
  transpose_t b_transpose = transpose_t::N;

  /// Data types: A, B, C, D.
  data_type_t a_dtype = data_type_t::None;
  data_type_t b_dtype = data_type_t::None;
  data_type_t c_dtype = data_type_t::None;
  data_type_t d_dtype = data_type_t::None;

  /// Compute type.
  data_type_t mi_dtype = data_type_t::None;

  /// MX block size.
  std::size_t a_mx_block_size = 0;
  std::size_t b_mx_block_size = 0;
};

/**
 * @brief Struct to define various workgroup mapping parameters.
 *
 * Contains all the parameters needed to describe various workgroup mapping parameters.
 */
struct workgroup_mapping_t {
  /// Workgroup mapping chunk size.
  std::size_t wgmxccchunk = 0;

  /// Workgroup mapping size.
  std::size_t wgmxcc = 8;

  /// Workgroup mapping size.
  int32_t wgm = 1;
};

/**
 * @brief Struct to define various staggerU parameters.
 *
 * Contains all the parameters needed to describe various staggerU parameters.
 */
struct staggerU_t {
  /// StaggerU mapping size.
  std::size_t staggerUMapping = 0;

  /// StaggerU size.
  std::size_t staggerU = 0;

  /// StaggerUStrideShift size.
  std::size_t staggerUStrideShift = 0;
};

/**
 * @brief Get runtime options (always uses global singleton).
 *
 * @param config Configuration struct (unused, kept for API compatibility)
 * @return const runtime_options& Reference to runtime options singleton
 */
inline const runtime_options& get_runtime_options(const config_t& config) {
  (void)config;  // Unused parameter - kept for API compatibility
  return runtime_options::get();
}

}  // namespace origami

// Specialization of std::hash in the std namespace for use of std::unordered_map with
// matrix_instruction and config_t as keys.
// Inline to prevent ODR violations when included in multiple shared libraries. (PR#1862)
namespace std {
template <>
struct hash<origami::matrix_instruction> {
  inline std::size_t operator()(const origami::matrix_instruction& k) const { return k.hash(); }
};

template <>
struct hash<origami::config_t> {
  inline std::size_t operator()(const origami::config_t& config) const noexcept {
    return config.hash();
  }
};
}  // namespace std
