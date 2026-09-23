// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard 3D batched matmul constants for testing get/set of valid matmul operations.
// Represents: A(2,3,4) x B(2,4,5) -> C(2,3,5) with batch=2, M=3, K=4, N=5.

constexpr int64_t K_MATMUL_TENSOR_A_UID = 1500;
constexpr std::array<int64_t, 3> K_MATMUL_TENSOR_A_DIMS = {2, 3, 4};
constexpr std::array<int64_t, 3> K_MATMUL_TENSOR_A_STRIDES = {12, 4, 1};

constexpr int64_t K_MATMUL_TENSOR_B_UID = 1501;
constexpr std::array<int64_t, 3> K_MATMUL_TENSOR_B_DIMS = {2, 4, 5};
constexpr std::array<int64_t, 3> K_MATMUL_TENSOR_B_STRIDES = {20, 5, 1};

constexpr int64_t K_MATMUL_TENSOR_C_UID = 1502;
constexpr std::array<int64_t, 3> K_MATMUL_TENSOR_C_DIMS = {2, 3, 5};
constexpr std::array<int64_t, 3> K_MATMUL_TENSOR_C_STRIDES = {15, 5, 1};

} // namespace hipdnn_tests::constants
