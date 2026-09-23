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

// Additional output tensor for fusion tests
constexpr int64_t K_PW_RELU_OUT_UID = 1354;

} // namespace hipdnn_tests::constants
