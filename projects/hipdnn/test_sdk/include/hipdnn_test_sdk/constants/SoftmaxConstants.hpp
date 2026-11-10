// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard Softmax constants for testing get/set of valid operations.
// These represent "any valid softmax" — specific values are not significant.

constexpr int64_t K_SOFTMAX_TENSOR_P_UID = 9200;
constexpr std::array<int64_t, 4> K_SOFTMAX_TENSOR_P_DIMS = {2, 4, 8, 16};
constexpr std::array<int64_t, 4> K_SOFTMAX_TENSOR_P_STRIDES = {512, 128, 16, 1};

constexpr int64_t K_SOFTMAX_TENSOR_S_UID = 9201;
constexpr std::array<int64_t, 4> K_SOFTMAX_TENSOR_S_DIMS = {2, 4, 8, 16};
constexpr std::array<int64_t, 4> K_SOFTMAX_TENSOR_S_STRIDES = {512, 128, 16, 1};

constexpr int64_t K_SOFTMAX_TENSOR_STATS_UID = 9202;
constexpr std::array<int64_t, 4> K_SOFTMAX_TENSOR_STATS_DIMS = {2, 4, 8, 1};
constexpr std::array<int64_t, 4> K_SOFTMAX_TENSOR_STATS_STRIDES = {32, 8, 1, 1};

} // namespace hipdnn_tests::constants
