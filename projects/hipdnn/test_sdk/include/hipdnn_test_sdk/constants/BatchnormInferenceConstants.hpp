// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants::batchnorm_inference
{

// Standard batchnorm inference constants for testing get/set of valid operations.
// These represent "any valid batchnorm inference" — specific values are not significant.

constexpr int64_t K_TENSOR_X_UID = 70;
constexpr int64_t K_TENSOR_MEAN_UID = 71;
constexpr int64_t K_TENSOR_INV_VARIANCE_UID = 72;
constexpr int64_t K_TENSOR_SCALE_UID = 73;
constexpr int64_t K_TENSOR_BIAS_UID = 74;
constexpr int64_t K_TENSOR_Y_UID = 75;

constexpr std::array<int64_t, 4> K_SPATIAL_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_SPATIAL_STRIDES = {65536, 1024, 32, 1};
constexpr std::array<int64_t, 4> K_CHANNEL_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_CHANNEL_STRIDES = {64, 1, 1, 1};

} // namespace hipdnn_tests::constants::batchnorm_inference

namespace hipdnn_tests::constants::batchnorm_inference::integration
{
constexpr int64_t K_TENSOR_X_UID = 70;
constexpr int64_t K_TENSOR_MEAN_UID = 71;
constexpr int64_t K_TENSOR_INV_VARIANCE_UID = 72;
constexpr int64_t K_TENSOR_SCALE_UID = 73;
constexpr int64_t K_TENSOR_BIAS_UID = 74;
constexpr int64_t K_TENSOR_Y_UID = 75;

constexpr std::array<int64_t, 4> K_SPATIAL_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_SPATIAL_STRIDES = {65536, 1024, 32, 1};
constexpr std::array<int64_t, 4> K_CHANNEL_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_CHANNEL_STRIDES = {64, 1, 1, 1};
} // namespace hipdnn_tests::constants::batchnorm_inference::integration
