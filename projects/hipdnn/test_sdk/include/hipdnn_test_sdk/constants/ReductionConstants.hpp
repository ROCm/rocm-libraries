// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard Reduction constants for testing get/set of valid operations.
// These represent "any valid reduction" — specific values are not significant.

constexpr int64_t K_REDUCTION_TENSOR_X_UID = 1700;
constexpr std::array<int64_t, 4> K_REDUCTION_TENSOR_X_DIMS = {2, 3, 4, 4};
constexpr std::array<int64_t, 4> K_REDUCTION_TENSOR_X_STRIDES = {48, 16, 4, 1};

constexpr int64_t K_REDUCTION_TENSOR_Y_UID = 1701;
constexpr std::array<int64_t, 4> K_REDUCTION_TENSOR_Y_DIMS = {2, 3, 1, 1};
constexpr std::array<int64_t, 4> K_REDUCTION_TENSOR_Y_STRIDES = {3, 1, 1, 1};

} // namespace hipdnn_tests::constants
