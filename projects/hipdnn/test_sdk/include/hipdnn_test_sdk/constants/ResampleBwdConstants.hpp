// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard 2D resample backward constants for testing.
// Represents: DY(1,3,16,16) -> DX(1,3,32,32) with 3x3 window, stride 2, padding 1.

constexpr int64_t K_TENSOR_DY_UID = 50;
constexpr std::array<int64_t, 4> K_TENSOR_DY_DIMS = {1, 3, 16, 16};
constexpr std::array<int64_t, 4> K_TENSOR_DY_STRIDES = {768, 256, 16, 1};

constexpr int64_t K_TENSOR_DX_UID = 51;
constexpr std::array<int64_t, 4> K_TENSOR_DX_DIMS = {1, 3, 32, 32};
constexpr std::array<int64_t, 4> K_TENSOR_DX_STRIDES = {3072, 1024, 32, 1};

constexpr int64_t K_TENSOR_INDEX_UID = 52;
constexpr std::array<int64_t, 4> K_TENSOR_INDEX_DIMS = {1, 3, 16, 16};
constexpr std::array<int64_t, 4> K_TENSOR_INDEX_STRIDES = {768, 256, 16, 1};

constexpr std::array<int64_t, 2> K_PRE_PADDING = {1, 1};
constexpr std::array<int64_t, 2> K_POST_PADDING = {1, 1};
constexpr std::array<int64_t, 2> K_STRIDE = {2, 2};
constexpr std::array<int64_t, 2> K_WINDOW = {3, 3};

} // namespace hipdnn_tests::constants
