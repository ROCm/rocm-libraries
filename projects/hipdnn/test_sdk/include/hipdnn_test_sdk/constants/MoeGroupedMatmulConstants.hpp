// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard MoeGroupedMatmul constants for testing get/set of valid operations.
// These represent "any valid moegroupedmatmul" — specific values are not significant.

constexpr int64_t K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID = 1900;
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS = {1, 8, 16};
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES = {128, 16, 1};

constexpr int64_t K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID = 1901;
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS = {2, 16, 32};
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES = {512, 32, 1};

constexpr int64_t K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID = 1902;
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS = {2, 1, 1};
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES = {1, 1, 1};

constexpr int64_t K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID = 1903;
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_DIMS = {1, 8, 1};
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_STRIDES = {8, 1, 1};

constexpr int64_t K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID = 1904;
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_DIMS = {1, 8, 1};
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_STRIDES = {8, 1, 1};

constexpr int64_t K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID = 1905;
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_DIMS = {1, 8, 32};
constexpr std::array<int64_t, 3> K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_STRIDES = {256, 32, 1};

} // namespace hipdnn_tests::constants
