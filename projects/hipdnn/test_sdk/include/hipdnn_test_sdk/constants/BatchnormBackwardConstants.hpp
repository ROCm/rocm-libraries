// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard batchnorm backward constants for testing.
// Represents: BatchNormBackward(DY[1, 64, 32, 32], X[1, 64, 32, 32], Scale[1, 64, 1, 1])
// Output: DX[1, 64, 32, 32], DScale[1, 64, 1, 1], DBias[1, 64, 1, 1]
// Optional: Mean[1, 64, 1, 1], InvVariance[1, 64, 1, 1]

// -- Required tensors --

constexpr int64_t K_BN_BWD_TENSOR_DY_UID = 60;
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_DY_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_DY_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_BN_BWD_TENSOR_X_UID = 61;
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_X_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_X_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_BN_BWD_TENSOR_SCALE_UID = 62;
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_SCALE_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_SCALE_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_BN_BWD_TENSOR_DX_UID = 63;
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_DX_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_DX_STRIDES = {65536, 1024, 32, 1};

constexpr int64_t K_BN_BWD_TENSOR_DSCALE_UID = 64;
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_DSCALE_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_DSCALE_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_BN_BWD_TENSOR_DBIAS_UID = 65;
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_DBIAS_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_DBIAS_STRIDES = {64, 1, 1, 1};

// -- Optional tensors --

constexpr int64_t K_BN_BWD_TENSOR_MEAN_UID = 66;
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_MEAN_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_MEAN_STRIDES = {64, 1, 1, 1};

constexpr int64_t K_BN_BWD_TENSOR_INV_VARIANCE_UID = 67;
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_INV_VARIANCE_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_INV_VARIANCE_STRIDES = {64, 1, 1, 1};

// -- Peer stats test UIDs --

constexpr int64_t K_BN_BWD_TENSOR_PEER_STAT_0_UID = 110;
constexpr int64_t K_BN_BWD_TENSOR_PEER_STAT_1_UID = 111;
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_PEER_STAT_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_BN_BWD_TENSOR_PEER_STAT_STRIDES = {64, 1, 1, 1};

// For MinimalRequired integration test (no optional tensors)
constexpr int64_t K_BN_BWD_MINIMAL_TENSOR_DY_UID = 620;
constexpr int64_t K_BN_BWD_MINIMAL_TENSOR_X_UID = 621;
constexpr int64_t K_BN_BWD_MINIMAL_TENSOR_SCALE_UID = 622;
constexpr int64_t K_BN_BWD_MINIMAL_TENSOR_DX_UID = 623;
constexpr int64_t K_BN_BWD_MINIMAL_TENSOR_DSCALE_UID = 624;
constexpr int64_t K_BN_BWD_MINIMAL_TENSOR_DBIAS_UID = 625;

// Distinct dims for AutoAssignedUids integration test
constexpr std::array<int64_t, 4> K_BN_BWD_AUTO_DATA_DIMS = {4, 32, 8, 8};
constexpr std::array<int64_t, 4> K_BN_BWD_AUTO_DATA_STRIDES = {2048, 64, 8, 1};
constexpr std::array<int64_t, 4> K_BN_BWD_AUTO_PARAM_DIMS = {1, 32, 1, 1};
constexpr std::array<int64_t, 4> K_BN_BWD_AUTO_PARAM_STRIDES = {32, 1, 1, 1};

} // namespace hipdnn_tests::constants
