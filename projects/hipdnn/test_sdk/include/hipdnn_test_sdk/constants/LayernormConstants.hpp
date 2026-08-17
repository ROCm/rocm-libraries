// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard layernorm constants for testing get/set of valid layernorm operations.
// Represents: LayerNorm(X[2, 64, 32, 32]) with scale/bias[1, 64, 32, 32], epsilon[1,1,1,1]
// Output: Y[2, 64, 32, 32], Mean[2, 1, 1, 1], InvVariance[2, 1, 1, 1]

constexpr int64_t K_LAYERNORM_TENSOR_X_UID = 700;
constexpr std::array<int64_t, 4> K_LAYERNORM_TENSOR_X_DIMS = {2, 64, 32, 32};
constexpr std::array<int64_t, 4> K_LAYERNORM_TENSOR_X_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_LAYERNORM_TENSOR_SCALE_UID = 701;
constexpr std::array<int64_t, 4> K_LAYERNORM_TENSOR_SCALE_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_LAYERNORM_TENSOR_SCALE_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_LAYERNORM_TENSOR_BIAS_UID = 702;
constexpr std::array<int64_t, 4> K_LAYERNORM_TENSOR_BIAS_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_LAYERNORM_TENSOR_BIAS_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_LAYERNORM_TENSOR_EPSILON_UID = 703;
constexpr std::array<int64_t, 1> K_LAYERNORM_TENSOR_EPSILON_DIMS = {1};
constexpr std::array<int64_t, 1> K_LAYERNORM_TENSOR_EPSILON_STRIDES = {1};

constexpr int64_t K_LAYERNORM_TENSOR_Y_UID = 704;
constexpr std::array<int64_t, 4> K_LAYERNORM_TENSOR_Y_DIMS = {2, 64, 32, 32};
constexpr std::array<int64_t, 4> K_LAYERNORM_TENSOR_Y_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_LAYERNORM_TENSOR_MEAN_UID = 705;
constexpr std::array<int64_t, 4> K_LAYERNORM_TENSOR_MEAN_DIMS = {2, 1, 1, 1};
constexpr std::array<int64_t, 4> K_LAYERNORM_TENSOR_MEAN_STRIDES = {1, 1, 1, 1};

constexpr int64_t K_LAYERNORM_TENSOR_INV_VARIANCE_UID = 706;
constexpr std::array<int64_t, 4> K_LAYERNORM_TENSOR_INV_VARIANCE_DIMS = {2, 1, 1, 1};
constexpr std::array<int64_t, 4> K_LAYERNORM_TENSOR_INV_VARIANCE_STRIDES = {1, 1, 1, 1};

} // namespace hipdnn_tests::constants
