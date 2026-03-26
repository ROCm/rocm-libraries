// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants::conv_wgrad
{

// Standard 2D convolution wgrad constants for testing get/set of valid conv operations.
// These represent "any valid conv" — specific values are not significant.

constexpr int64_t K_TENSOR_X_UID = 20;
constexpr std::array<int64_t, 4> K_TENSOR_X_DIMS = {1, 3, 32, 32};
constexpr std::array<int64_t, 4> K_TENSOR_X_STRIDES = {3072, 1024, 32, 1};

constexpr int64_t K_TENSOR_DY_UID = 21;
constexpr std::array<int64_t, 4> K_TENSOR_DY_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_TENSOR_DY_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_TENSOR_DW_UID = 22;
constexpr std::array<int64_t, 4> K_TENSOR_DW_DIMS = {64, 3, 3, 3};
constexpr std::array<int64_t, 4> K_TENSOR_DW_STRIDES = {27, 9, 3, 1};

constexpr std::array<int64_t, 2> K_CONV_PADDING = {1, 1};
constexpr std::array<int64_t, 2> K_CONV_STRIDE = {1, 1};
constexpr std::array<int64_t, 2> K_CONV_DILATION = {1, 1};

} // namespace hipdnn_tests::constants::conv_wgrad

namespace hipdnn_tests::constants::conv_wgrad::integration
{
constexpr int64_t K_TENSOR_X_UID = 20;
constexpr int64_t K_TENSOR_DY_UID = 21;
constexpr int64_t K_TENSOR_DW_UID = 22;

constexpr std::array<int64_t, 4> K_TENSOR_X_DIMS = {2, 3, 14, 14};
constexpr std::array<int64_t, 4> K_TENSOR_X_STRIDES = {588, 196, 14, 1};
constexpr std::array<int64_t, 4> K_TENSOR_DY_DIMS = {2, 8, 7, 7};
constexpr std::array<int64_t, 4> K_TENSOR_DY_STRIDES = {392, 49, 7, 1};

constexpr std::array<int64_t, 2> K_CONV_PRE_PADDING = {1, 1};
constexpr std::array<int64_t, 2> K_CONV_POST_PADDING = {1, 1};
constexpr std::array<int64_t, 2> K_CONV_STRIDE = {2, 2};
constexpr std::array<int64_t, 2> K_CONV_DILATION = {1, 1};
} // namespace hipdnn_tests::constants::conv_wgrad::integration
