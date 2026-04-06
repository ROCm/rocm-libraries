// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard 2D convolution dgrad constants for testing get/set of valid conv operations.
// These represent the backward data gradient of the fprop constants in ConvFpropConstants.hpp:
//   dy has the shape of the forward output y: [N, K, H, W]
//   w  has the forward filter shape:          [K, C, R, S]
//   dx has the shape of the forward input x:  [N, C, H, W]

constexpr int64_t K_DGRAD_TENSOR_DY_UID = 1100;
constexpr std::array<int64_t, 4> K_DGRAD_TENSOR_DY_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_DGRAD_TENSOR_DY_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_DGRAD_TENSOR_W_UID = 1101;
constexpr std::array<int64_t, 4> K_DGRAD_TENSOR_W_DIMS = {64, 3, 3, 3};
constexpr std::array<int64_t, 4> K_DGRAD_TENSOR_W_STRIDES = {27, 9, 3, 1};

constexpr int64_t K_DGRAD_TENSOR_DX_UID = 1102;
constexpr std::array<int64_t, 4> K_DGRAD_TENSOR_DX_DIMS = {1, 3, 32, 32};
constexpr std::array<int64_t, 4> K_DGRAD_TENSOR_DX_STRIDES = {3072, 1024, 32, 1};

constexpr std::array<int64_t, 2> K_DGRAD_CONV_PADDING = {1, 1};
constexpr std::array<int64_t, 2> K_DGRAD_CONV_STRIDE = {1, 1};
constexpr std::array<int64_t, 2> K_DGRAD_CONV_DILATION = {1, 1};

} // namespace hipdnn_tests::constants
