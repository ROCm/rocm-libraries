// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard 2D convolution wgrad constants for testing get/set of valid conv operations.
// These represent "any valid conv" — specific values are not significant.

constexpr int64_t K_WGRAD_TENSOR_X_UID = 1200;
constexpr std::array<int64_t, 4> K_WGRAD_TENSOR_X_DIMS = {1, 3, 32, 32};
constexpr std::array<int64_t, 4> K_WGRAD_TENSOR_X_STRIDES = {3072, 1024, 32, 1};

constexpr int64_t K_WGRAD_TENSOR_DY_UID = 1201;
constexpr std::array<int64_t, 4> K_WGRAD_TENSOR_DY_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_WGRAD_TENSOR_DY_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_WGRAD_TENSOR_DW_UID = 1202;
constexpr std::array<int64_t, 4> K_WGRAD_TENSOR_DW_DIMS = {64, 3, 3, 3};
constexpr std::array<int64_t, 4> K_WGRAD_TENSOR_DW_STRIDES = {27, 9, 3, 1};

constexpr std::array<int64_t, 2> K_WGRAD_CONV_PADDING = {1, 1};
constexpr std::array<int64_t, 2> K_WGRAD_CONV_STRIDE = {1, 1};
constexpr std::array<int64_t, 2> K_WGRAD_CONV_DILATION = {1, 1};

} // namespace hipdnn_tests::constants
