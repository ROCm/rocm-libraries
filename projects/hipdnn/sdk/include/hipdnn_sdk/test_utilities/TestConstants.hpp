// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace hipdnn_sdk
{
namespace test_utilities
{
namespace constants
{

// Mathematical constants
constexpr float PI = 3.14159265f;
constexpr float E = 2.71828183f;
constexpr float SQRT_2 = 1.41421356f;
constexpr float SQRT_3 = 1.73205081f;
constexpr float SQRT_5 = 2.23606798f;
constexpr float GOLDEN_RATIO = 1.61803399f; // φ
constexpr float LN_2 = 0.69314718f;

// Common test values
constexpr float TEST_VALUE_1 = 1.0f;
constexpr float TEST_VALUE_2 = 2.0f;
constexpr float TEST_VALUE_3 = 3.0f;
constexpr float TEST_VALUE_5 = 5.0f;
constexpr float TEST_VALUE_HALF = 0.5f;
constexpr float TEST_VALUE_2_5 = 2.5f;
constexpr float TEST_VALUE_1_5 = 1.5f;
constexpr float TEST_VALUE_4 = 4.0f;

// Precision test values
constexpr float PRECISION_TEST_A = 0.123456789f;
constexpr float PRECISION_TEST_B = 0.987654321f;

// Tolerance values for different floating point types
constexpr float TOLERANCE_HALF = 1e-3f;
constexpr float TOLERANCE_BFLOAT16 = 1e-2f;
constexpr float TOLERANCE_FLOAT = 1e-5f;
constexpr double TOLERANCE_DOUBLE = 1e-6;

// Broadcasting test values
constexpr float BROADCAST_MULTIPLIER_10 = 10.0f;
constexpr float BROADCAST_MULTIPLIER_100 = 100.0f;

} // namespace constants
} // namespace test_utilities
} // namespace hipdnn_sdk
