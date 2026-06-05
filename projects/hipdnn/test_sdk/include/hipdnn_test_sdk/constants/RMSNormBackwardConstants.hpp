// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard RMSNormBackward constants for testing get/set of valid operations.
// These represent "any valid rmsnormbackward" — specific values are not significant.

constexpr int64_t K_RMSNORMBACKWARD_TENSOR_DY_UID = 70;
constexpr std::array<int64_t, 4> K_RMSNORMBACKWARD_TENSOR_DY_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_RMSNORMBACKWARD_TENSOR_DY_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_RMSNORMBACKWARD_TENSOR_X_UID = 71;
constexpr std::array<int64_t, 4> K_RMSNORMBACKWARD_TENSOR_X_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_RMSNORMBACKWARD_TENSOR_X_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_RMSNORMBACKWARD_TENSOR_SCALE_UID = 72;
constexpr std::array<int64_t, 4> K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID = 73;
constexpr std::array<int64_t, 4> K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS = {1, 1, 1, 1};
constexpr std::array<int64_t, 4> K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES = {1, 1, 1, 1};

constexpr int64_t K_RMSNORMBACKWARD_TENSOR_DX_UID = 74;
constexpr std::array<int64_t, 4> K_RMSNORMBACKWARD_TENSOR_DX_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_RMSNORMBACKWARD_TENSOR_DX_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_RMSNORMBACKWARD_TENSOR_DSCALE_UID = 75;
constexpr std::array<int64_t, 4> K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_RMSNORMBACKWARD_TENSOR_DBIAS_UID = 76;
constexpr std::array<int64_t, 4> K_RMSNORMBACKWARD_TENSOR_DBIAS_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_RMSNORMBACKWARD_TENSOR_DBIAS_STRIDES = {65536, 1024, 32, 1};

} // namespace hipdnn_tests::constants
