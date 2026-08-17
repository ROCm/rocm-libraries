// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard block-scale quantize constants for testing get/set of valid operations.
// Represents: X(1,1,64,128) -> Y(1,1,64,128), Scale(1,1,64,4) with block_size=32.

constexpr int64_t K_BSQ_TENSOR_X_UID = 1400;
constexpr std::array<int64_t, 4> K_BSQ_TENSOR_X_DIMS = {1, 1, 64, 128};
constexpr std::array<int64_t, 4> K_BSQ_TENSOR_X_STRIDES = {8192, 8192, 128, 1};

constexpr int64_t K_BSQ_TENSOR_Y_UID = 1401;
constexpr std::array<int64_t, 4> K_BSQ_TENSOR_Y_DIMS = {1, 1, 64, 128};
constexpr std::array<int64_t, 4> K_BSQ_TENSOR_Y_STRIDES = {8192, 8192, 128, 1};

constexpr int64_t K_BSQ_TENSOR_SCALE_UID = 1402;
constexpr std::array<int64_t, 4> K_BSQ_TENSOR_SCALE_DIMS = {1, 1, 64, 4};
constexpr std::array<int64_t, 4> K_BSQ_TENSOR_SCALE_STRIDES = {256, 256, 4, 1};

constexpr int32_t K_BSQ_BLOCK_SIZE = 32;

} // namespace hipdnn_tests::constants
