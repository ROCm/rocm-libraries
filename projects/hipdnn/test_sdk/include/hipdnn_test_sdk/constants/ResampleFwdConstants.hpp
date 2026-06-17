// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard 2D resample forward constants for testing.
// Represents: X(1,3,32,32) -> Y(1,3,16,16) with 3x3 window, stride 2, padding 1.

constexpr int64_t K_RESAMPLE_FWD_TENSOR_X_UID = 40;
constexpr std::array<int64_t, 4> K_RESAMPLE_FWD_TENSOR_X_DIMS = {1, 3, 32, 32};
constexpr std::array<int64_t, 4> K_RESAMPLE_FWD_TENSOR_X_STRIDES = {3072, 1024, 32, 1};

constexpr int64_t K_RESAMPLE_FWD_TENSOR_Y_UID = 41;
constexpr std::array<int64_t, 4> K_RESAMPLE_FWD_TENSOR_Y_DIMS = {1, 3, 16, 16};
constexpr std::array<int64_t, 4> K_RESAMPLE_FWD_TENSOR_Y_STRIDES = {768, 256, 16, 1};

constexpr int64_t K_RESAMPLE_FWD_TENSOR_INDEX_UID = 42;
constexpr std::array<int64_t, 4> K_RESAMPLE_FWD_TENSOR_INDEX_DIMS = {1, 3, 16, 16};
constexpr std::array<int64_t, 4> K_RESAMPLE_FWD_TENSOR_INDEX_STRIDES = {768, 256, 16, 1};

constexpr std::array<int64_t, 2> K_RESAMPLE_FWD_PRE_PADDING = {1, 1};
constexpr std::array<int64_t, 2> K_RESAMPLE_FWD_POST_PADDING = {1, 1};
constexpr std::array<int64_t, 2> K_RESAMPLE_FWD_STRIDE = {2, 2};
constexpr std::array<int64_t, 2> K_RESAMPLE_FWD_WINDOW = {3, 3};

} // namespace hipdnn_tests::constants
