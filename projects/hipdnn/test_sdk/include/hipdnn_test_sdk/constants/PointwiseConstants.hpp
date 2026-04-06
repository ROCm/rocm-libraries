// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard pointwise tensor constants for backend unit tests.
// UIDs and dims match the pointwise.yaml test_data fixture.
constexpr int64_t K_PW_TENSOR_IN0_UID = 1300;
constexpr int64_t K_PW_TENSOR_OUT0_UID = 1301;
constexpr int64_t K_PW_TENSOR_IN1_UID = 1302;
constexpr int64_t K_PW_TENSOR_IN2_UID = 1303;
constexpr std::array<int64_t, 4> K_PW_TENSOR_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_PW_TENSOR_STRIDES = {65536, 1024, 32, 1};

} // namespace hipdnn_tests::constants

namespace hipdnn_tests::constants::integration
{

// Pointwise tensor constants for frontend integration tests.
// UIDs are distinct from unit test UIDs to avoid confusion.
constexpr int64_t K_PW_TENSOR_IN0_UID = 1350;
constexpr int64_t K_PW_TENSOR_IN1_UID = 1351;
constexpr int64_t K_PW_TENSOR_IN2_UID = 1352;
constexpr int64_t K_PW_TENSOR_OUT0_UID = 1353;
constexpr std::array<int64_t, 4> K_PW_TENSOR_DIMS = {2, 64, 32, 32};
constexpr std::array<int64_t, 4> K_PW_TENSOR_STRIDES = {65536, 1024, 32, 1};

// Fusion test constants
constexpr int64_t K_PW_RELU_OUT_UID = 1354;

} // namespace hipdnn_tests::constants::integration
