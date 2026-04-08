// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard batchnorm training forward constants for testing.
// Represents: BatchNorm(X[2, 64, 16, 16]) with scale/bias[1, 64, 1, 1], epsilon[1]
// Output: Y[2, 64, 16, 16], optional Mean[1, 64, 1, 1], InvVariance[1, 64, 1, 1]
// Optional running stats: prev/next running mean/variance [1, 64, 1, 1], momentum[1]

// -- Required tensors --

constexpr int64_t K_BATCHNORM_TENSOR_X_UID = 500;
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_X_DIMS = {2, 64, 16, 16};
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_X_STRIDES = {16384, 256, 16, 1};

constexpr int64_t K_BATCHNORM_TENSOR_SCALE_UID = 501;
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_SCALE_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_SCALE_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_BATCHNORM_TENSOR_BIAS_UID = 502;
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_BIAS_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_BIAS_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_BATCHNORM_TENSOR_EPSILON_UID = 503;
constexpr std::array<int64_t, 1> K_BATCHNORM_TENSOR_EPSILON_DIMS = {1};
constexpr std::array<int64_t, 1> K_BATCHNORM_TENSOR_EPSILON_STRIDES = {1};

constexpr int64_t K_BATCHNORM_TENSOR_Y_UID = 504;
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_Y_DIMS = {2, 64, 16, 16};
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_Y_STRIDES = {16384, 256, 16, 1};

// -- Optional tensors: batch statistics outputs --

constexpr int64_t K_BATCHNORM_TENSOR_MEAN_UID = 505;
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_MEAN_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_MEAN_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_BATCHNORM_TENSOR_INV_VARIANCE_UID = 506;
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_INV_VARIANCE_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_INV_VARIANCE_STRIDES = {64, 1, 1, 1};

// -- Optional tensors: running statistics --

constexpr int64_t K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID = 507;
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID = 508;
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_BATCHNORM_TENSOR_MOMENTUM_UID = 509;
constexpr std::array<int64_t, 1> K_BATCHNORM_TENSOR_MOMENTUM_DIMS = {1};
constexpr std::array<int64_t, 1> K_BATCHNORM_TENSOR_MOMENTUM_STRIDES = {1};

constexpr int64_t K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID = 510;
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID = 511;
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_STRIDES = {64, 1, 1, 1};

// -- Peer stats test UIDs --

constexpr int64_t K_BATCHNORM_TENSOR_PEER_STAT_0_UID = 515;
constexpr int64_t K_BATCHNORM_TENSOR_PEER_STAT_1_UID = 516;
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_PEER_STAT_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BATCHNORM_TENSOR_PEER_STAT_STRIDES = {64, 1, 1, 1};

// For MinimalRequired integration test
constexpr int64_t K_BATCHNORM_MINIMAL_TENSOR_X_UID = 540;
constexpr int64_t K_BATCHNORM_MINIMAL_TENSOR_SCALE_UID = 541;
constexpr int64_t K_BATCHNORM_MINIMAL_TENSOR_BIAS_UID = 542;
constexpr int64_t K_BATCHNORM_MINIMAL_TENSOR_EPSILON_UID = 543;
constexpr int64_t K_BATCHNORM_MINIMAL_TENSOR_Y_UID = 544;
constexpr int64_t K_BATCHNORM_MINIMAL_TENSOR_MEAN_UID = 545;
constexpr int64_t K_BATCHNORM_MINIMAL_TENSOR_INV_VARIANCE_UID = 546;

// Distinct dims for AutoAssignedUids integration test
constexpr std::array<int64_t, 4> K_BATCHNORM_AUTO_DATA_DIMS = {4, 32, 8, 8};
constexpr std::array<int64_t, 4> K_BATCHNORM_AUTO_DATA_STRIDES = {2048, 64, 8, 1};
constexpr std::array<int64_t, 4> K_BATCHNORM_AUTO_PARAM_DIMS = {1, 32, 1, 1};
constexpr std::array<int64_t, 4> K_BATCHNORM_AUTO_PARAM_STRIDES = {32, 1, 1, 1};

} // namespace hipdnn_tests::constants
