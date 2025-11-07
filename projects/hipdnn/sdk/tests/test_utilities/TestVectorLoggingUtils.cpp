// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/VectorLoggingUtils.hpp>
#include <limits>
#include <vector>

class TestVectorLoggingUtils : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Setup if needed
    }
};

TEST_F(TestVectorLoggingUtils, FormatsEmptyVector)
{
    std::vector<int64_t> emptyVec;
    std::string result = fmt::format("{}", emptyVec);
    EXPECT_EQ(result, "[]");
}

TEST_F(TestVectorLoggingUtils, FormatsSingleElement)
{
    std::vector<int64_t> singleVec = {42};
    std::string result = fmt::format("{}", singleVec);
    EXPECT_EQ(result, "[42]");
}

TEST_F(TestVectorLoggingUtils, FormatsMultipleElements)
{
    std::vector<int64_t> multiVec = {1, 2, 3, 4, 5};
    std::string result = fmt::format("{}", multiVec);
    EXPECT_EQ(result, "[1, 2, 3, 4, 5]");
}

TEST_F(TestVectorLoggingUtils, FormatsNegativeNumbers)
{
    std::vector<int64_t> negativeVec = {-100, -50, 0, 50, 100};
    std::string result = fmt::format("{}", negativeVec);
    EXPECT_EQ(result, "[-100, -50, 0, 50, 100]");
}

TEST_F(TestVectorLoggingUtils, FormatsLargeNumbers)
{
    std::vector<int64_t> largeVec = {1000000000000, 2000000000000, 3000000000000};
    std::string result = fmt::format("{}", largeVec);
    EXPECT_EQ(result, "[1000000000000, 2000000000000, 3000000000000]");
}

TEST_F(TestVectorLoggingUtils, FormatsMinMaxValues)
{
    std::vector<int64_t> extremeVec
        = {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    std::string result = fmt::format("{}", extremeVec);
    std::string expected = fmt::format(
        "[{}, {}]", std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max());
    EXPECT_EQ(result, expected);
}

TEST_F(TestVectorLoggingUtils, WorksWithFormatStrings)
{
    std::vector<int64_t> vec = {10, 20, 30};
    std::string result = fmt::format("Vector contents: {}", vec);
    EXPECT_EQ(result, "Vector contents: [10, 20, 30]");
}

TEST_F(TestVectorLoggingUtils, FormatsMultipleVectorsInSameString)
{
    std::vector<int64_t> vec1 = {1, 2};
    std::vector<int64_t> vec2 = {3, 4, 5};
    std::string result = fmt::format("First: {}, Second: {}", vec1, vec2);
    EXPECT_EQ(result, "First: [1, 2], Second: [3, 4, 5]");
}

TEST_F(TestVectorLoggingUtils, FormatsZeroValues)
{
    std::vector<int64_t> zeroVec = {0, 0, 0};
    std::string result = fmt::format("{}", zeroVec);
    EXPECT_EQ(result, "[0, 0, 0]");
}

TEST_F(TestVectorLoggingUtils, PreservesElementOrder)
{
    std::vector<int64_t> orderedVec = {5, 3, 8, 1, 9};
    std::string result = fmt::format("{}", orderedVec);
    EXPECT_EQ(result, "[5, 3, 8, 1, 9]");
}

TEST_F(TestVectorLoggingUtils, HandlesAlternatingSignValues)
{
    std::vector<int64_t> altVec = {1, -1, 2, -2, 3, -3};
    std::string result = fmt::format("{}", altVec);
    EXPECT_EQ(result, "[1, -1, 2, -2, 3, -3]");
}
