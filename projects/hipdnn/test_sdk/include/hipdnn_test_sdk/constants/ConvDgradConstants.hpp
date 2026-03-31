// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants::dgrad
{

// Standard 2D convolution dgrad constants for testing get/set of valid conv operations.
// These represent the backward data gradient of the fprop constants in ConvFpropConstants.hpp:
//   dy has the shape of the forward output y: [N, K, H, W]
//   w  has the forward filter shape:          [K, C, R, S]
//   dx has the shape of the forward input x:  [N, C, H, W]

constexpr int64_t K_TENSOR_DY_UID = 10;
constexpr std::array<int64_t, 4> K_TENSOR_DY_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_TENSOR_DY_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_TENSOR_W_UID = 11;
constexpr std::array<int64_t, 4> K_TENSOR_W_DIMS = {64, 3, 3, 3};
constexpr std::array<int64_t, 4> K_TENSOR_W_STRIDES = {27, 9, 3, 1};

constexpr int64_t K_TENSOR_DX_UID = 12;
constexpr std::array<int64_t, 4> K_TENSOR_DX_DIMS = {1, 3, 32, 32};
constexpr std::array<int64_t, 4> K_TENSOR_DX_STRIDES = {3072, 1024, 32, 1};

constexpr std::array<int64_t, 2> K_CONV_PADDING = {1, 1};
constexpr std::array<int64_t, 2> K_CONV_STRIDE = {1, 1};
constexpr std::array<int64_t, 2> K_CONV_DILATION = {1, 1};

} // namespace hipdnn_tests::constants::dgrad

namespace hipdnn_tests::constants::dgrad::integration
{

// Integration test constants: backward data gradient of the fprop integration constants.
// Uses stride={2,2} to produce different spatial dims between dy and dx.

constexpr int64_t K_TENSOR_DY_UID = 30;
constexpr std::array<int64_t, 4> K_TENSOR_DY_DIMS = {2, 8, 7, 7};
constexpr std::array<int64_t, 4> K_TENSOR_DY_STRIDES = {392, 49, 7, 1};

constexpr int64_t K_TENSOR_W_UID = 31;
constexpr std::array<int64_t, 4> K_TENSOR_W_DIMS = {8, 3, 3, 3};
constexpr std::array<int64_t, 4> K_TENSOR_W_STRIDES = {27, 9, 3, 1};

constexpr int64_t K_TENSOR_DX_UID = 32;

constexpr std::array<int64_t, 2> K_CONV_PRE_PADDING = {1, 1};
constexpr std::array<int64_t, 2> K_CONV_POST_PADDING = {1, 1};
constexpr std::array<int64_t, 2> K_CONV_STRIDE = {2, 2};
constexpr std::array<int64_t, 2> K_CONV_DILATION = {1, 1};

} // namespace hipdnn_tests::constants::dgrad::integration
