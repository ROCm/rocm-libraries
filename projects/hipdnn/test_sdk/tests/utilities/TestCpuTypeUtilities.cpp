// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

using hipdnn_data_sdk::types::fp8_e4m3;
using hipdnn_test_sdk::detail::safeTestTypeCast;

TEST(TestCpuTypeUtilities, CastInRangeIntToInt)
{
    auto v = safeTestTypeCast<int8_t>(127);
    EXPECT_EQ(v, static_cast<int8_t>(127));
}

TEST(TestCpuTypeUtilities, CastOutOfRangeIntToIntThrows)
{
    EXPECT_THROW((safeTestTypeCast<int8_t>(128)), std::out_of_range);
    EXPECT_THROW((safeTestTypeCast<int8_t>(-129)), std::out_of_range);
}

TEST(TestCpuTypeUtilities, CastOutOfRangeFloatToIntThrows)
{
    EXPECT_THROW((safeTestTypeCast<int8_t>(200.0f)), std::out_of_range);
    EXPECT_THROW((safeTestTypeCast<int8_t>(-200.0f)), std::out_of_range);
}

TEST(TestCpuTypeUtilities, CastNonFiniteFloatSourceThrows)
{
    EXPECT_THROW((safeTestTypeCast<int8_t>(std::numeric_limits<float>::infinity())),
                 std::out_of_range);
    EXPECT_THROW((safeTestTypeCast<int8_t>(-std::numeric_limits<float>::infinity())),
                 std::out_of_range);
    EXPECT_THROW((safeTestTypeCast<int8_t>(std::numeric_limits<float>::quiet_NaN())),
                 std::out_of_range);
}

TEST(TestCpuTypeUtilities, CastFp8SourceAndCheckTargetBounds)
{
    const fp8_e4m3 src(10.0f);
    EXPECT_EQ(safeTestTypeCast<int8_t>(src), static_cast<int8_t>(10));

    // E4M3 max finite is 448.0f; values above should fail bounds checks.
    EXPECT_THROW((safeTestTypeCast<fp8_e4m3>(1000.0f)), std::out_of_range);
}

TEST(TestCpuTypeUtilities, CastInt64BoundaryToInt32)
{
    const int64_t vLowest = std::numeric_limits<int32_t>::lowest();
    EXPECT_EQ(safeTestTypeCast<int32_t>(vLowest), std::numeric_limits<int32_t>::lowest());
    EXPECT_THROW(safeTestTypeCast<int32_t>(vLowest - 1), std::out_of_range);

    const int64_t vMax = std::numeric_limits<int32_t>::max();
    EXPECT_EQ(safeTestTypeCast<int32_t>(vMax), std::numeric_limits<int32_t>::max());
    EXPECT_THROW(safeTestTypeCast<int32_t>(vMax + 1), std::out_of_range);
}

TEST(TestCpuTypeUtilities, CastDoubleToFloatBoundary)
{
    auto vMin = static_cast<double>(std::numeric_limits<float>::lowest());
    EXPECT_FLOAT_EQ(safeTestTypeCast<float>(vMin), std::numeric_limits<float>::lowest());
    EXPECT_THROW((safeTestTypeCast<float>(vMin * 1.1)), std::out_of_range);

    auto vMax = static_cast<double>(std::numeric_limits<float>::max());
    EXPECT_FLOAT_EQ(safeTestTypeCast<float>(vMax), std::numeric_limits<float>::max());
    EXPECT_THROW((safeTestTypeCast<float>(vMax * 1.1)), std::out_of_range);
}
