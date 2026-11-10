// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard MoeRouting constants for testing get/set of valid operations.
// These represent "any valid moerouting" — specific values are not significant.

constexpr int64_t K_MOE_ROUTING_TENSOR_TOPK_IDS_UID = 9700;
constexpr std::array<int64_t, 2> K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS = {4, 2};
constexpr std::array<int64_t, 2> K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES = {2, 1};

constexpr int64_t K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID = 9701;
constexpr std::array<int64_t, 3> K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS = {5, 1, 1};
constexpr std::array<int64_t, 3> K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES = {1, 1, 1};

constexpr int64_t K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID = 9702;
constexpr std::array<int64_t, 3> K_MOE_ROUTING_TENSOR_TOKEN_INDEX_DIMS = {1, 8, 1};
constexpr std::array<int64_t, 3> K_MOE_ROUTING_TENSOR_TOKEN_INDEX_STRIDES = {8, 1, 1};

constexpr int64_t K_MOE_ROUTING_TENSOR_TOKEN_KS_UID = 9703;
constexpr std::array<int64_t, 3> K_MOE_ROUTING_TENSOR_TOKEN_KS_DIMS = {1, 8, 1};
constexpr std::array<int64_t, 3> K_MOE_ROUTING_TENSOR_TOKEN_KS_STRIDES = {8, 1, 1};

} // namespace hipdnn_tests::constants
