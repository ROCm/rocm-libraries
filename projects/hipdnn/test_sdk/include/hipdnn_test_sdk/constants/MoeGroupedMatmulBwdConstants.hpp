// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard MoeGroupedMatmulBwd constants for testing get/set of valid operations.
// These represent "any valid moegroupedmatmulbwd" — specific values are not significant.

constexpr int64_t K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID = 1910;
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS = {1, 8, 32};
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES = {256, 32, 1};

constexpr int64_t K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID = 1911;
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS = {1, 8, 16};
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES = {128, 16, 1};

constexpr int64_t K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID = 1912;
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS
    = {2, 1, 1};
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES
    = {1, 1, 1};

constexpr int64_t K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID = 1913;
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS = {2, 16, 32};
// Column-major [K*N, 1, K], matching the layout MoeGroupedMatmulBwdNode infers for dweight.
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES = {512, 1, 16};

} // namespace hipdnn_tests::constants
