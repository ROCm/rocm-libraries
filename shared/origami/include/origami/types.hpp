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

#include "origami/log.hpp"
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
inline int data_type_to_bytes(data_type_t type) {
  return math::safe_ceil_div(datatype_to_bits(type), 8);
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
   * @brief Default constructor that reads from environment variables.
   */
  runtime_options();

  /**
   * @brief Constructor with explicit values (does not read from environment).
   */
  runtime_options(bool debug, bool heuristics, double variance);

  /**
   * @brief Get the global runtime options instance.
   */
  static runtime_options& get();

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
   * @return double Variance value from ANALYTICAL_GEMM_HEURISTICS_VARIANCE, or 0.0 if not set
   */
  static double read_heuristics_variance_from_env();

  /**
   * @brief Update runtime options from environment variables.
   */
  void update_from_env();
};

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

  /// Occupancy (number of waves resident per CU).
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

  /// Runtime options (if null, uses global singleton)
  const runtime_options* runtime_opts{nullptr};

  /// Logger for analytical metrics
  mutable logger_t logger;

  constexpr bool operator==(const config_t& o) const noexcept {
    return mt == o.mt && mi == o.mi && cache_hints_a == o.cache_hints_a &&
           cache_hints_b == o.cache_hints_b && workgroup_mapping == o.workgroup_mapping;
  }

  std::size_t hash() const {
    return std::hash<size_t>()(mt.m) ^ std::hash<size_t>()(mt.n) ^ std::hash<size_t>()(mt.k) ^
           std::hash<size_t>()(mi.m) ^ std::hash<size_t>()(mi.n) ^ std::hash<size_t>()(mi.k) ^
           std::hash<int>()(cache_hints_a) ^ std::hash<int>()(cache_hints_b) ^
           std::hash<int>()(workgroup_mapping);
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
 * @brief Get runtime options from config, or global singleton if config doesn't specify.
 *
 * @param config Configuration struct (may contain runtime_opts pointer)
 * @return const runtime_options& Reference to runtime options
 */
inline const runtime_options& get_runtime_options(const config_t& config) {
  return config.runtime_opts ? *config.runtime_opts : runtime_options::get();
}

}  // namespace origami

// Specialization of std::hash in the std namespace for use of std::unordered_map with
// matrix_instruction and config_t as keys.
namespace std {
template <>
struct hash<origami::matrix_instruction> {
  std::size_t operator()(const origami::matrix_instruction& k) const { return k.hash(); }
};

template <>
struct hash<origami::config_t> {
  std::size_t operator()(const origami::config_t& config) const noexcept { return config.hash(); }
};
}  // namespace std
