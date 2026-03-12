// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuTypeUtilities.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>

using hipdnn_data_sdk::types::fp8_e4m3;
using hipdnn_test_sdk::detail::safeTestTypeCast;

TEST(TestCpuTypeUtilities, SafeTestTypeCastInRangeIntToInt)
{
    auto v = safeTestTypeCast<int8_t>(127);
    EXPECT_EQ(v, static_cast<int8_t>(127));
}

TEST(TestCpuTypeUtilities, SafeTestTypeCastOutOfRangeIntToIntThrows)
{
    EXPECT_THROW((safeTestTypeCast<int8_t>(128)), std::out_of_range);
    EXPECT_THROW((safeTestTypeCast<int8_t>(-129)), std::out_of_range);
}

TEST(TestCpuTypeUtilities, SafeTestTypeCastOutOfRangeFloatToIntThrows)
{
    EXPECT_THROW((safeTestTypeCast<int8_t>(200.0f)), std::out_of_range);
    EXPECT_THROW((safeTestTypeCast<int8_t>(-200.0f)), std::out_of_range);
}

TEST(TestCpuTypeUtilities, SafeTestTypeCastNonFiniteFloatSourceThrows)
{
    EXPECT_THROW((safeTestTypeCast<int8_t>(std::numeric_limits<float>::infinity())),
                 std::out_of_range);
    EXPECT_THROW((safeTestTypeCast<int8_t>(-std::numeric_limits<float>::infinity())),
                 std::out_of_range);
    EXPECT_THROW((safeTestTypeCast<int8_t>(std::numeric_limits<float>::quiet_NaN())),
                 std::out_of_range);
}

TEST(TestCpuTypeUtilities, SafeTestTypeCastSupportsFp8SourceAndTargetBounds)
{
    fp8_e4m3 src(10.0f);
    EXPECT_EQ(safeTestTypeCast<int8_t>(src), static_cast<int8_t>(10));

    // E4M3 max finite is 448.0f; values above should fail bounds checks.
    EXPECT_THROW((safeTestTypeCast<fp8_e4m3>(1000.0f)), std::out_of_range);
}
