// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard 4D RMSNorm constants for testing get/set of valid rmsnorm operations.
// Represents: X(1,64,32,32) normalized per-channel with Scale(1,64,1,1),
// Epsilon scalar, optional Bias(1,64,1,1), output Y(1,64,32,32),
// optional Inv_rms(1,1,32,32).

constexpr int64_t K_RMSNORM_TENSOR_X_UID = 1600;
constexpr std::array<int64_t, 4> K_RMSNORM_TENSOR_X_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_RMSNORM_TENSOR_X_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_RMSNORM_TENSOR_SCALE_UID = 1601;
constexpr std::array<int64_t, 4> K_RMSNORM_TENSOR_SCALE_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_RMSNORM_TENSOR_SCALE_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_RMSNORM_TENSOR_EPSILON_UID = 1602;
constexpr std::array<int64_t, 4> K_RMSNORM_TENSOR_EPSILON_DIMS = {1, 1, 1, 1};
constexpr std::array<int64_t, 4> K_RMSNORM_TENSOR_EPSILON_STRIDES = {1, 1, 1, 1};

constexpr int64_t K_RMSNORM_TENSOR_Y_UID = 1603;
constexpr std::array<int64_t, 4> K_RMSNORM_TENSOR_Y_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_RMSNORM_TENSOR_Y_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_RMSNORM_TENSOR_BIAS_UID = 1604;
constexpr std::array<int64_t, 4> K_RMSNORM_TENSOR_BIAS_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_RMSNORM_TENSOR_BIAS_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_RMSNORM_TENSOR_INV_RMS_UID = 1605;
constexpr std::array<int64_t, 4> K_RMSNORM_TENSOR_INV_RMS_DIMS = {1, 1, 32, 32};
constexpr std::array<int64_t, 4> K_RMSNORM_TENSOR_INV_RMS_STRIDES = {1024, 1024, 32, 1};

} // namespace hipdnn_tests::constants
