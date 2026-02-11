// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/NumericLimits.hpp>

using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

using namespace hipdnn_test_sdk::utilities;

TEST(TestNumericLimits, Float)
{
    EXPECT_EQ(getEpsilon<float>(), std::numeric_limits<float>::epsilon());
}

TEST(TestNumericLimits, Double)
{
    EXPECT_EQ(getEpsilon<double>(), std::numeric_limits<double>::epsilon());
}

TEST(TestNumericLimits, Half)
{
    // 2^-10 = 0.0009765625
    EXPECT_NEAR(getEpsilon<half>(), 0.0009765625, 1e-9);
}

TEST(TestNumericLimits, BFloat16)
{
    // 2^-7 = 0.0078125
    EXPECT_NEAR(getEpsilon<bfloat16>(), 0.0078125, 1e-9);
}
