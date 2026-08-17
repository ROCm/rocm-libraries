// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard block scale dequantize constants for testing get/set of valid operations.
// Represents: X(1,1,64,128) with Scale(1,1,64,4) -> Y(1,1,64,128), block_size=32.

constexpr int64_t K_BSD_TENSOR_X_UID = 50;
constexpr std::array<int64_t, 4> K_BSD_TENSOR_X_DIMS = {1, 1, 64, 128};
constexpr std::array<int64_t, 4> K_BSD_TENSOR_X_STRIDES = {8192, 8192, 128, 1};

constexpr int64_t K_BSD_TENSOR_SCALE_UID = 51;
constexpr std::array<int64_t, 4> K_BSD_TENSOR_SCALE_DIMS = {1, 1, 64, 4};
constexpr std::array<int64_t, 4> K_BSD_TENSOR_SCALE_STRIDES = {256, 256, 4, 1};

constexpr int64_t K_BSD_TENSOR_Y_UID = 52;
constexpr std::array<int64_t, 4> K_BSD_TENSOR_Y_DIMS = {1, 1, 64, 128};
constexpr std::array<int64_t, 4> K_BSD_TENSOR_Y_STRIDES = {8192, 8192, 128, 1};

constexpr int32_t K_BSD_BLOCK_SIZE = 32;

} // namespace hipdnn_tests::constants
