// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include "origami/core/data_type.hpp"
#include "origami/core/dimensions.hpp"
#include "origami/core/enums.hpp"

namespace origami {

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

  /// Number of compute units the caller intends to use for this GEMM.
  /// 0 (default) means "use all CUs" and preserves the legacy behaviour of
  /// modelling against the full hardware CU count. When set to a non-zero
  /// value, solution selection models the problem as if only this many CUs
  /// were available (e.g. CU masking / partitioned execution), which changes
  /// grid launch, timesteps, occupancy, and the ranked config.
  std::size_t num_cus = 0;

  /// Number of query heads (for attention workloads).
  std::size_t q_heads = 32;

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

}  // namespace origami
