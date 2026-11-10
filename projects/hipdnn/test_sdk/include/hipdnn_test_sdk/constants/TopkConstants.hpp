// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard Topk constants for testing get/set of valid operations.
// These represent "any valid topk" — specific values are not significant.

constexpr int64_t K_TOPK_TENSOR_X_UID = 9300;
constexpr std::array<int64_t, 3> K_TOPK_TENSOR_X_DIMS = {2, 3, 8};
constexpr std::array<int64_t, 3> K_TOPK_TENSOR_X_STRIDES = {24, 8, 1};

constexpr int64_t K_TOPK_TENSOR_INDICES_UID = 9301;
constexpr std::array<int64_t, 3> K_TOPK_TENSOR_INDICES_DIMS = {2, 3, 4};
constexpr std::array<int64_t, 3> K_TOPK_TENSOR_INDICES_STRIDES = {12, 4, 1};

constexpr int64_t K_TOPK_TENSOR_VALUES_UID = 9302;
constexpr std::array<int64_t, 3> K_TOPK_TENSOR_VALUES_DIMS = {2, 3, 4};
constexpr std::array<int64_t, 3> K_TOPK_TENSOR_VALUES_STRIDES = {12, 4, 1};

} // namespace hipdnn_tests::constants
