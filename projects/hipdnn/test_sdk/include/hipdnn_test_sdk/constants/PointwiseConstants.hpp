// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard pointwise tensor constants for backend unit tests.
// UIDs and dims match the pointwise.yaml test_data fixture.
constexpr int64_t K_PW_TENSOR_IN0_UID = 40;
constexpr int64_t K_PW_TENSOR_OUT0_UID = 41;
constexpr int64_t K_PW_TENSOR_IN1_UID = 3;
constexpr int64_t K_PW_TENSOR_IN2_UID = 4;
constexpr std::array<int64_t, 4> K_PW_TENSOR_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_PW_TENSOR_STRIDES = {65536, 1024, 32, 1};

} // namespace hipdnn_tests::constants

namespace hipdnn_tests::constants::integration
{

// Pointwise tensor constants for frontend integration tests.
// UIDs are distinct from unit test UIDs to avoid confusion.
constexpr int64_t K_PW_TENSOR_IN0_UID = 100;
constexpr int64_t K_PW_TENSOR_IN1_UID = 101;
constexpr int64_t K_PW_TENSOR_IN2_UID = 102;
constexpr int64_t K_PW_TENSOR_OUT0_UID = 200;
constexpr std::array<int64_t, 4> K_PW_TENSOR_DIMS = {2, 64, 32, 32};
constexpr std::array<int64_t, 4> K_PW_TENSOR_STRIDES = {65536, 1024, 32, 1};

// Fusion test constants
constexpr int64_t K_PW_RELU_OUT_UID = 300;

} // namespace hipdnn_tests::constants::integration
