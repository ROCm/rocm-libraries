// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <tuple>

#include "origami/math.hpp"

namespace origami {

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

inline data_type_t int_to_data_type(int dt) { return (data_type_t)dt; }

inline int data_type_to_bits(data_type_t type) {
  switch (type) {
    case data_type_t::Float: return 32;
    case data_type_t::Double: return 64;
    case data_type_t::ComplexFloat: return 64;
    case data_type_t::ComplexDouble: return 128;
    case data_type_t::Half: return 16;
    case data_type_t::Int8x4: return 32;
    case data_type_t::Int32: return 32;
    case data_type_t::BFloat16: return 16;
    case data_type_t::Int8: return 8;
    case data_type_t::Int4: return 4;
    case data_type_t::Int64: return 64;
    case data_type_t::XFloat32: return 32;
    case data_type_t::Float8_fnuz: return 8;
    case data_type_t::BFloat8_fnuz: return 8;
    case data_type_t::Float8BFloat8_fnuz: return 8;
    case data_type_t::BFloat8Float8_fnuz: return 8;
    case data_type_t::Float8: return 8;
    case data_type_t::BFloat8: return 8;
    case data_type_t::Float8BFloat8: return 8;
    case data_type_t::BFloat8Float8: return 8;
    case data_type_t::Float6: return 6;
    case data_type_t::BFloat6: return 6;
    case data_type_t::Float4: return 4;
    default: return -1;  // Invalid type
  }
}

inline int data_type_to_bytes(data_type_t type) {
  return math::safe_ceil_div(data_type_to_bits(type), 8);
}

inline std::string to_string(data_type_t type) {
  switch (type) {
    case data_type_t::Float: return "Float";
    case data_type_t::Double: return "Double";
    case data_type_t::ComplexFloat: return "ComplexFloat";
    case data_type_t::ComplexDouble: return "ComplexDouble";
    case data_type_t::Half: return "Half";
    case data_type_t::Int8x4: return "Int8x4";
    case data_type_t::Int32: return "Int32";
    case data_type_t::BFloat16: return "BFloat16";
    case data_type_t::Int8: return "Int8";
    case data_type_t::Int4: return "Int4";
    case data_type_t::Int64: return "Int64";
    case data_type_t::XFloat32: return "XFloat32";
    case data_type_t::Float8_fnuz: return "Float8_fnuz";
    case data_type_t::BFloat8_fnuz: return "BFloat8_fnuz";
    case data_type_t::Float8BFloat8_fnuz: return "Float8BFloat8_fnuz";
    case data_type_t::BFloat8Float8_fnuz: return "BFloat8Float8_fnuz";
    case data_type_t::Float8: return "Float8";
    case data_type_t::BFloat8: return "BFloat8";
    case data_type_t::Float8BFloat8: return "Float8BFloat8";
    case data_type_t::BFloat8Float8: return "BFloat8Float8";
    case data_type_t::Float6: return "Float6";
    case data_type_t::BFloat6: return "BFloat6";
    case data_type_t::Float4: return "Float4";
    default: return "Invalid";
  }
  return "Invalid";
}

inline data_type_t string_to_data_type(std::string s) {
  if (s == "f32") return data_type_t::Float;
  if (s == "c32") return data_type_t::ComplexFloat;
  if (s == "c64") return data_type_t::ComplexDouble;
  if (s == "f64") return data_type_t::Double;
  if (s == "f16") return data_type_t::Half;
  if (s == "i32") return data_type_t::Int32;
  if (s == "bf16") return data_type_t::BFloat16;
  if (s == "i8") return data_type_t::Int8;
  if (s == "i4") return data_type_t::Int4;
  if (s == "xf32") return data_type_t::XFloat32;
  if (s == "f8") return data_type_t::Float8;
  if (s == "bf8") return data_type_t::BFloat8;
  if (s == "f6") return data_type_t::Float6;
  if (s == "bf6") return data_type_t::BFloat6;
  if (s == "f4") return data_type_t::Float4;
  return data_type_t::None;
}

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
 * @brief Full kernel configuration (tile shape + execution parameters).
 *
 * Holds the geometric tile sizes along with occupancy,
 * work-group mapping (WGM), and cache-control hints.
 */
struct config_t {
  /// Macro tile and matrix-instruction shape.
  dim3_t mt;
  dim3_t mi;

  /// Occupancy (number of waves resident per CU).
  int occupancy;

  /// Reorder workgroup id for L2 reuse.
  int workgroup_mapping{};

  /// Whether operand A is accessed with cache-flags.
  int cache_hints_a{};

  /// Whether operand B is accessed with cache-flags.
  int cache_hints_b{};

  /// Workspace size parameters.
  std::size_t workspace_size{};
  std::size_t workspace_size_per_elem_c{};

  /// Reduction strategy.
  reduction_t reduction_strategy{};

  constexpr bool operator==(const config_t& o) const noexcept {
    return mt == o.mt && mi == o.mi && cache_hints_a == o.cache_hints_a &&
           cache_hints_b == o.cache_hints_b;
  }

  std::size_t hash() const {
    return std::hash<size_t>()(mt.m) ^ std::hash<size_t>()(mt.n) ^ std::hash<size_t>()(mt.k) ^
           std::hash<size_t>()(mi.m) ^ std::hash<size_t>()(mi.n) ^ std::hash<size_t>()(mi.k) ^
           std::hash<int>()(cache_hints_a) ^ std::hash<int>()(cache_hints_b);
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
  dim3_t size;

  /// Batch size.
  std::size_t batch;

  /// Transpose types (TT, TN, NT, TT.)
  transpose_t a_transpose;
  transpose_t b_transpose;

  /// Data types: A, B, C, D.
  data_type_t a_dtype;
  data_type_t b_dtype;
  data_type_t c_dtype;
  data_type_t d_dtype;

  /// Compute type.
  data_type_t mi_dtype;

  /// MX block size.
  std::size_t a_mx_block_size;
  std::size_t b_mx_block_size;
};

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