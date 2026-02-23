// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard 2D convolution weight-gradient (wrw) constants for testing.
// These represent "any valid conv wrw" — specific values are not significant.
// Convolution parameters (padding, stride, dilation) are shared with ConvFpropConstants.hpp.

constexpr int64_t K_WRW_TENSOR_X_UID = 20;
constexpr std::array<int64_t, 4> K_WRW_TENSOR_X_DIMS = {1, 3, 32, 32};
constexpr std::array<int64_t, 4> K_WRW_TENSOR_X_STRIDES = {3072, 1024, 32, 1};

constexpr int64_t K_WRW_TENSOR_DY_UID = 21;
constexpr std::array<int64_t, 4> K_WRW_TENSOR_DY_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_WRW_TENSOR_DY_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_WRW_TENSOR_DW_UID = 22;
constexpr std::array<int64_t, 4> K_WRW_TENSOR_DW_DIMS = {64, 3, 3, 3};
constexpr std::array<int64_t, 4> K_WRW_TENSOR_DW_STRIDES = {27, 9, 3, 1};

} // namespace hipdnn_tests::constants
