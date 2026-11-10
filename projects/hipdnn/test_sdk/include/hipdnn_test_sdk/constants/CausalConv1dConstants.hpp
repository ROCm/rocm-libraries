// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard CausalConv1d constants for testing get/set of valid operations.
// These represent "any valid causalconv1d" — specific values are not significant.

constexpr int64_t K_CAUSAL_CONV1D_TENSOR_X_UID = 9600;
constexpr std::array<int64_t, 2> K_CAUSAL_CONV1D_TENSOR_X_DIMS = {8, 4};
constexpr std::array<int64_t, 2> K_CAUSAL_CONV1D_TENSOR_X_STRIDES = {4, 1};

constexpr int64_t K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID = 9601;
constexpr std::array<int64_t, 2> K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS = {4, 4};
constexpr std::array<int64_t, 2> K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES = {4, 1};

constexpr int64_t K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID = 9602;
constexpr std::array<int64_t, 1> K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS = {3};
constexpr std::array<int64_t, 1> K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES = {1};

constexpr int64_t K_CAUSAL_CONV1D_TENSOR_Y_UID = 9603;
constexpr std::array<int64_t, 2> K_CAUSAL_CONV1D_TENSOR_Y_DIMS = {8, 4};
constexpr std::array<int64_t, 2> K_CAUSAL_CONV1D_TENSOR_Y_STRIDES = {4, 1};

constexpr int64_t K_CAUSAL_CONV1D_TENSOR_BIAS_UID = 9604;
constexpr std::array<int64_t, 1> K_CAUSAL_CONV1D_TENSOR_BIAS_DIMS = {4};
constexpr std::array<int64_t, 1> K_CAUSAL_CONV1D_TENSOR_BIAS_STRIDES = {1};

constexpr int64_t K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID = 9605;
constexpr std::array<int64_t, 3> K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_DIMS = {2, 4, 3};
constexpr std::array<int64_t, 3> K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_STRIDES = {12, 3, 1};

constexpr int64_t K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID = 9606;
constexpr std::array<int64_t, 1> K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_DIMS = {2};
constexpr std::array<int64_t, 1> K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_STRIDES = {1};

constexpr int64_t K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID = 9607;
constexpr std::array<int64_t, 3> K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_DIMS = {2, 4, 3};
constexpr std::array<int64_t, 3> K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_STRIDES = {12, 3, 1};

} // namespace hipdnn_tests::constants
