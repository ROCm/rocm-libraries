// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard Rope constants for testing get/set of valid operations.
// These represent "any valid rope" — specific values are not significant.

constexpr int64_t K_ROPE_TENSOR_X_UID = 9100;
constexpr std::array<int64_t, 4> K_ROPE_TENSOR_X_DIMS = {2, 4, 16, 8};
constexpr std::array<int64_t, 4> K_ROPE_TENSOR_X_STRIDES = {512, 128, 8, 1};

constexpr int64_t K_ROPE_TENSOR_FREQS_UID = 9101;
constexpr std::array<int64_t, 4> K_ROPE_TENSOR_FREQS_DIMS = {1, 1, 16, 4};
constexpr std::array<int64_t, 4> K_ROPE_TENSOR_FREQS_STRIDES = {64, 64, 4, 1};

constexpr int64_t K_ROPE_TENSOR_Y_UID = 9102;
constexpr std::array<int64_t, 4> K_ROPE_TENSOR_Y_DIMS = {2, 4, 16, 8};
constexpr std::array<int64_t, 4> K_ROPE_TENSOR_Y_STRIDES = {512, 128, 8, 1};

} // namespace hipdnn_tests::constants
