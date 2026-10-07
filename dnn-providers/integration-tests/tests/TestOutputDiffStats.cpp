// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The element-wise measure behind a record's max_abs_err and mismatch_pct. Host tensors
// only: nothing here touches a device.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>

#include "harness/catalog-sweep/OutputDiffStats.hpp"

using namespace hipdnn_integration_tests::catalog_sweep;
using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_data_sdk::utilities::TensorView;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

// A [1, N] tensor holding `values`.
Tensor<float> row(const std::vector<float>& values)
{
    Tensor<float> tensor(std::vector<int64_t>{1, static_cast<int64_t>(values.size())});
    TensorView<float> view(tensor);
    for(std::size_t i = 0; i < values.size(); ++i)
    {
        view.getHostValue({0, static_cast<int64_t>(i)}) = values[i];
    }
    tensor.markHostModified();
    return tensor;
}

constexpr float K_INF = std::numeric_limits<float>::infinity();
constexpr float K_NAN = std::numeric_limits<float>::quiet_NaN();

} // namespace

TEST(TestOutputDiffStats, IdenticalTensorsMeasureZero)
{
    auto expected = row({1.0f, 2.0f, 3.0f});
    auto actual = row({1.0f, 2.0f, 3.0f});

    const auto stats = computeOutputDiffStats(DataType::FLOAT, expected, actual, 0.0f, 0.0f);

    ASSERT_TRUE(stats.has_value());
    EXPECT_EQ(stats->totalElements, 3U);
    EXPECT_EQ(stats->mismatchedElements, 0U);
    EXPECT_EQ(stats->nanElements, 0U);
    EXPECT_DOUBLE_EQ(stats->maxAbsErr, 0.0);
    EXPECT_DOUBLE_EQ(stats->mismatchPct(), 0.0);
}

// The largest difference is reported whether or not it is out of tolerance: a passing
// kernel's headroom is as useful in the report as a failing one's overshoot.
TEST(TestOutputDiffStats,
     MaxErrorCoversInToleranceElementsAndMismatchesCountOnlyTheOutOfToleranceOnes)
{
    auto expected = row({10.0f, 10.0f, 10.0f, 10.0f});
    auto actual = row({10.0f, 10.4f, 12.0f, 10.0f});

    const auto stats = computeOutputDiffStats(DataType::FLOAT, expected, actual, 0.5f, 0.0f);

    ASSERT_TRUE(stats.has_value());
    EXPECT_NEAR(stats->maxAbsErr, 2.0, 1e-6);
    EXPECT_EQ(stats->mismatchedElements, 1U) << "only 12.0 is more than 0.5 away";
    EXPECT_DOUBLE_EQ(stats->mismatchPct(), 25.0);
}

TEST(TestOutputDiffStats, RelativeToleranceScalesWithTheReferenceMagnitude)
{
    auto expected = row({1.0f, 1000.0f});
    auto actual = row({1.05f, 1005.0f}); // 5% off the first, 0.5% off the second

    const auto stats = computeOutputDiffStats(DataType::FLOAT, expected, actual, 0.0f, 0.01f);

    ASSERT_TRUE(stats.has_value());
    EXPECT_EQ(stats->mismatchedElements, 1U) << "1.05 is outside 1%, 1005 is inside it";
}

// An output the kernel never wrote holds the sentinel NaN: that is the signature of a
// kernel that writes nothing, and it must be counted rather than ignored.
TEST(TestOutputDiffStats, NanInTheActualOutputIsCountedAndMismatched)
{
    auto expected = row({1.0f, 2.0f, 3.0f});
    auto actual = row({1.0f, K_NAN, K_NAN});

    const auto stats = computeOutputDiffStats(DataType::FLOAT, expected, actual, 1e6f, 1e6f);

    ASSERT_TRUE(stats.has_value());
    EXPECT_EQ(stats->nanElements, 2U);
    EXPECT_EQ(stats->mismatchedElements, 2U) << "no tolerance forgives a NaN";
    EXPECT_NEAR(stats->mismatchPct(), 100.0 * 2.0 / 3.0, 1e-9);
    EXPECT_DOUBLE_EQ(stats->maxAbsErr, 0.0) << "a NaN has no difference to be the maximum of";
}

TEST(TestOutputDiffStats, InfinitiesOfOneSignMatchAndOppositeInfinitiesDoNot)
{
    auto expected = row({K_INF, K_INF, -K_INF});
    auto actual = row({K_INF, -K_INF, -K_INF});

    const auto stats = computeOutputDiffStats(DataType::FLOAT, expected, actual, 0.0f, 0.0f);

    ASSERT_TRUE(stats.has_value());
    EXPECT_EQ(stats->mismatchedElements, 1U);
}

TEST(TestOutputDiffStats, ShapesThatDifferCannotBeMeasured)
{
    auto expected = row({1.0f, 2.0f});
    auto actual = row({1.0f, 2.0f, 3.0f});

    EXPECT_FALSE(computeOutputDiffStats(DataType::FLOAT, expected, actual, 0.0f, 0.0f).has_value());
}

TEST(TestOutputDiffStats, ElementTypesWithoutAnElementwiseMeasureAreNotMeasured)
{
    auto expected = row({1.0f});
    auto actual = row({1.0f});

    EXPECT_FALSE(computeOutputDiffStats(DataType::INT32, expected, actual, 0.0f, 0.0f).has_value());
}

// NOLINTEND(readability-identifier-naming)
