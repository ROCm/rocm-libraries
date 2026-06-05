// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard 2D convolution fprop constants for testing get/set of valid conv operations.
// These represent "any valid conv" — specific values are not significant.

constexpr int64_t K_FPROP_TENSOR_X_UID = 1000;
constexpr std::array<int64_t, 4> K_FPROP_TENSOR_X_DIMS = {1, 3, 32, 32};
constexpr std::array<int64_t, 4> K_FPROP_TENSOR_X_STRIDES = {3072, 1024, 32, 1};

constexpr int64_t K_FPROP_TENSOR_W_UID = 1001;
constexpr std::array<int64_t, 4> K_FPROP_TENSOR_W_DIMS = {64, 3, 3, 3};
constexpr std::array<int64_t, 4> K_FPROP_TENSOR_W_STRIDES = {27, 9, 3, 1};

constexpr int64_t K_FPROP_TENSOR_Y_UID = 1002;
constexpr std::array<int64_t, 4> K_FPROP_TENSOR_Y_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_FPROP_TENSOR_Y_STRIDES = {65536, 1024, 32, 1};

constexpr std::array<int64_t, 2> K_FPROP_CONV_PADDING = {1, 1};
constexpr std::array<int64_t, 2> K_FPROP_CONV_STRIDE = {1, 1};
constexpr std::array<int64_t, 2> K_FPROP_CONV_DILATION = {1, 1};

// Second convolution layer tensors (for multi-operation graph tests)
constexpr int64_t K_FPROP_TENSOR_X2_UID = 1003;
constexpr std::array<int64_t, 4> K_FPROP_TENSOR_X2_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_FPROP_TENSOR_X2_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_FPROP_TENSOR_W2_UID = 1004;
constexpr std::array<int64_t, 4> K_FPROP_TENSOR_W2_DIMS = {128, 64, 3, 3};
constexpr std::array<int64_t, 4> K_FPROP_TENSOR_W2_STRIDES = {576, 9, 3, 1};

constexpr int64_t K_FPROP_TENSOR_Y2_UID = 1005;
constexpr std::array<int64_t, 4> K_FPROP_TENSOR_Y2_DIMS = {1, 128, 32, 32};
constexpr std::array<int64_t, 4> K_FPROP_TENSOR_Y2_STRIDES = {131072, 1024, 32, 1};

} // namespace hipdnn_tests::constants
