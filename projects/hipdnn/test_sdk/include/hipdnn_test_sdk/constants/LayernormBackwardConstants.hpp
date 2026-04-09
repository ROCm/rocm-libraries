// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard LayernormBackward constants for testing get/set of valid operations.
// These represent "any valid layernormbackward" — specific values are not significant.

constexpr int64_t K_LAYERNORMBACKWARD_TENSOR_DY_UID = 10;
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_DY_DIMS = {16, 64, 32, 32};
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_LAYERNORMBACKWARD_TENSOR_X_UID = 11;
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_X_DIMS = {16, 64, 32, 32};
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_X_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_LAYERNORMBACKWARD_TENSOR_SCALE_UID = 12;
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_LAYERNORMBACKWARD_TENSOR_MEAN_UID = 13;
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS = {16, 1, 1, 1};
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES = {1, 1, 1, 1};

constexpr int64_t K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID = 14;
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS = {16, 1, 1, 1};
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES = {1, 1, 1, 1};

constexpr int64_t K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID = 15;
constexpr std::array<int64_t, 1> K_LAYERNORMBACKWARD_TENSOR_EPSILON_DIMS = {1};
constexpr std::array<int64_t, 1> K_LAYERNORMBACKWARD_TENSOR_EPSILON_STRIDES = {1};

constexpr int64_t K_LAYERNORMBACKWARD_TENSOR_DX_UID = 16;
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_DX_DIMS = {16, 64, 32, 32};
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID = 17;
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID = 18;
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES = {65536, 1024, 32, 1};

} // namespace hipdnn_tests::constants
