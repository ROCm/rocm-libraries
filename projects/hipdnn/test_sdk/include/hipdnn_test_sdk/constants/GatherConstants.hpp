// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard Gather constants for testing get/set of valid operations.
// These represent "any valid gather" — specific values are not significant.

constexpr int64_t K_GATHER_TENSOR_X_UID = 9400;
constexpr std::array<int64_t, 2> K_GATHER_TENSOR_X_DIMS = {16, 8};
constexpr std::array<int64_t, 2> K_GATHER_TENSOR_X_STRIDES = {8, 1};

constexpr int64_t K_GATHER_TENSOR_INDICES_UID = 9401;
constexpr std::array<int64_t, 2> K_GATHER_TENSOR_INDICES_DIMS = {2, 3};
constexpr std::array<int64_t, 2> K_GATHER_TENSOR_INDICES_STRIDES = {3, 1};

constexpr int64_t K_GATHER_TENSOR_Y_UID = 9402;
constexpr std::array<int64_t, 3> K_GATHER_TENSOR_Y_DIMS = {2, 3, 8};
constexpr std::array<int64_t, 3> K_GATHER_TENSOR_Y_STRIDES = {24, 8, 1};

} // namespace hipdnn_tests::constants
