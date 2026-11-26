// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <tuple>

#include "origami/origami_export.h"

namespace origami {

/**
 * @brief Enumeration of supported data types.
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
 * @return int Number of bits, or -1 for an invalid type
 */
ORIGAMI_EXPORT int datatype_to_bits(data_type_t type);

/**
 * @brief Convert data_type_t to number of bytes.
 *
 * @param type Data type
 * @return double Number of bytes (fractional for sub-byte types)
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
ORIGAMI_EXPORT std::string datatype_to_string(data_type_t type);

/**
 * @brief Convert string to data_type_t enum.
 *
 * @param s Short type name such as "f32", "bf16" or "i8"
 * @return data_type_t Corresponding data type, or data_type_t::None for an unknown name
 */
ORIGAMI_EXPORT data_type_t string_to_datatype(std::string s);

/**
 * @brief Struct to define a matrix instruction.
 *
 * Contains the dimensions and data type of a matrix instruction.
 */
struct matrix_instruction {
  /// M dimension of the instruction.
  size_t MI_M;

  /// N dimension of the instruction.
  size_t MI_N;

  /// K dimension of the instruction.
  size_t MI_K;

  /// Data type of the instruction's inputs.
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

  /// @return Hash over the dimensions and the input type.
  std::size_t hash() const {
    return std::hash<size_t>()(MI_M) ^ std::hash<size_t>()(MI_N) ^ std::hash<size_t>()(MI_K) ^
           std::hash<data_type_t>()(mi_input_type);
  }
};

}  // namespace origami

namespace std {

/**
 * @brief std::hash for origami::matrix_instruction, so it can key unordered containers.
 *
 * Inline so that several shared libraries including this header do not violate the ODR.
 */
template <>
struct hash<origami::matrix_instruction> {
  inline std::size_t operator()(const origami::matrix_instruction& k) const { return k.hash(); }
};

}  // namespace std
