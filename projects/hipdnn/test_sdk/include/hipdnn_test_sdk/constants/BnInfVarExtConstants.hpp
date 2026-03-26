// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Batchnorm inference variance ext constants for descriptor unit tests and graph tests.
// UIDs and tensor shapes match the DescriptorGenerator YAML config.

constexpr int64_t K_BN_INF_VAR_EXT_X_UID = 80;
constexpr std::array<int64_t, 4> K_BN_INF_VAR_EXT_X_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_BN_INF_VAR_EXT_X_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_BN_INF_VAR_EXT_MEAN_UID = 81;
constexpr std::array<int64_t, 4> K_BN_INF_VAR_EXT_MEAN_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BN_INF_VAR_EXT_MEAN_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_BN_INF_VAR_EXT_VARIANCE_UID = 82;
constexpr std::array<int64_t, 4> K_BN_INF_VAR_EXT_VARIANCE_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BN_INF_VAR_EXT_VARIANCE_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_BN_INF_VAR_EXT_SCALE_UID = 83;
constexpr std::array<int64_t, 4> K_BN_INF_VAR_EXT_SCALE_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BN_INF_VAR_EXT_SCALE_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_BN_INF_VAR_EXT_BIAS_UID = 84;
constexpr std::array<int64_t, 4> K_BN_INF_VAR_EXT_BIAS_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BN_INF_VAR_EXT_BIAS_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_BN_INF_VAR_EXT_Y_UID = 85;
constexpr std::array<int64_t, 4> K_BN_INF_VAR_EXT_Y_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_BN_INF_VAR_EXT_Y_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_BN_INF_VAR_EXT_EPSILON_UID = 86;
constexpr std::array<int64_t, 4> K_BN_INF_VAR_EXT_EPSILON_DIMS = {1, 1, 1, 1};
constexpr std::array<int64_t, 4> K_BN_INF_VAR_EXT_EPSILON_STRIDES = {1, 1, 1, 1};

// Generic aliases used by generated test code
constexpr auto K_TENSOR_X_UID = K_BN_INF_VAR_EXT_X_UID;
constexpr auto& K_TENSOR_X_DIMS = K_BN_INF_VAR_EXT_X_DIMS;
constexpr auto& K_TENSOR_X_STRIDES = K_BN_INF_VAR_EXT_X_STRIDES;

constexpr auto K_TENSOR_MEAN_UID = K_BN_INF_VAR_EXT_MEAN_UID;
constexpr auto& K_TENSOR_MEAN_DIMS = K_BN_INF_VAR_EXT_MEAN_DIMS;
constexpr auto& K_TENSOR_MEAN_STRIDES = K_BN_INF_VAR_EXT_MEAN_STRIDES;

constexpr auto K_TENSOR_VARIANCE_UID = K_BN_INF_VAR_EXT_VARIANCE_UID;
constexpr auto& K_TENSOR_VARIANCE_DIMS = K_BN_INF_VAR_EXT_VARIANCE_DIMS;
constexpr auto& K_TENSOR_VARIANCE_STRIDES = K_BN_INF_VAR_EXT_VARIANCE_STRIDES;

constexpr auto K_TENSOR_SCALE_UID = K_BN_INF_VAR_EXT_SCALE_UID;
constexpr auto& K_TENSOR_SCALE_DIMS = K_BN_INF_VAR_EXT_SCALE_DIMS;
constexpr auto& K_TENSOR_SCALE_STRIDES = K_BN_INF_VAR_EXT_SCALE_STRIDES;

constexpr auto K_TENSOR_BIAS_UID = K_BN_INF_VAR_EXT_BIAS_UID;
constexpr auto& K_TENSOR_BIAS_DIMS = K_BN_INF_VAR_EXT_BIAS_DIMS;
constexpr auto& K_TENSOR_BIAS_STRIDES = K_BN_INF_VAR_EXT_BIAS_STRIDES;

constexpr auto K_TENSOR_Y_UID = K_BN_INF_VAR_EXT_Y_UID;
constexpr auto& K_TENSOR_Y_DIMS = K_BN_INF_VAR_EXT_Y_DIMS;
constexpr auto& K_TENSOR_Y_STRIDES = K_BN_INF_VAR_EXT_Y_STRIDES;

constexpr auto K_TENSOR_EPSILON_UID = K_BN_INF_VAR_EXT_EPSILON_UID;
constexpr auto& K_TENSOR_EPSILON_DIMS = K_BN_INF_VAR_EXT_EPSILON_DIMS;
constexpr auto& K_TENSOR_EPSILON_STRIDES = K_BN_INF_VAR_EXT_EPSILON_STRIDES;

} // namespace hipdnn_tests::constants
