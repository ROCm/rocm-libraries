// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/VectorLoggingUtils.hpp>
#include <limits>
#include <sstream>
#include <vector>

using hipdnn_test_sdk::utilities::StreamVec;

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
    const std::vector<int64_t> emptyVec;
    std::ostringstream oss;
    oss << StreamVec(emptyVec);
    EXPECT_EQ(oss.str(), "[]");
}

TEST_F(TestVectorLoggingUtils, FormatsSingleElement)
{
    const std::vector<int64_t> singleVec = {42};
    std::ostringstream oss;
    oss << StreamVec(singleVec);
    EXPECT_EQ(oss.str(), "[42]");
}

TEST_F(TestVectorLoggingUtils, FormatsMultipleElements)
{
    const std::vector<int64_t> multiVec = {1, 2, 3, 4, 5};
    std::ostringstream oss;
    oss << StreamVec(multiVec);
    EXPECT_EQ(oss.str(), "[1, 2, 3, 4, 5]");
}

TEST_F(TestVectorLoggingUtils, FormatsNegativeNumbers)
{
    const std::vector<int64_t> negativeVec = {-100, -50, 0, 50, 100};
    std::ostringstream oss;
    oss << StreamVec(negativeVec);
    EXPECT_EQ(oss.str(), "[-100, -50, 0, 50, 100]");
}

TEST_F(TestVectorLoggingUtils, FormatsLargeNumbers)
{
    const std::vector<int64_t> largeVec = {1000000000000, 2000000000000, 3000000000000};
    std::ostringstream oss;
    oss << StreamVec(largeVec);
    EXPECT_EQ(oss.str(), "[1000000000000, 2000000000000, 3000000000000]");
}

TEST_F(TestVectorLoggingUtils, FormatsMinMaxValues)
{
    const std::vector<int64_t> extremeVec
        = {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    std::ostringstream oss;
    oss << StreamVec(extremeVec);
    std::ostringstream expected;
    expected << "[" << std::numeric_limits<int64_t>::min() << ", "
             << std::numeric_limits<int64_t>::max() << "]";
    EXPECT_EQ(oss.str(), expected.str());
}

TEST_F(TestVectorLoggingUtils, WorksWithFormatStrings)
{
    const std::vector<int64_t> vec = {10, 20, 30};
    std::ostringstream oss;
    oss << "Vector contents: " << StreamVec(vec);
    EXPECT_EQ(oss.str(), "Vector contents: [10, 20, 30]");
}

TEST_F(TestVectorLoggingUtils, FormatsMultipleVectorsInSameString)
{
    const std::vector<int64_t> vec1 = {1, 2};
    const std::vector<int64_t> vec2 = {3, 4, 5};
    std::ostringstream oss;
    oss << "First: " << StreamVec(vec1) << ", Second: " << StreamVec(vec2);
    EXPECT_EQ(oss.str(), "First: [1, 2], Second: [3, 4, 5]");
}

TEST_F(TestVectorLoggingUtils, FormatsZeroValues)
{
    const std::vector<int64_t> zeroVec = {0, 0, 0};
    std::ostringstream oss;
    oss << StreamVec(zeroVec);
    EXPECT_EQ(oss.str(), "[0, 0, 0]");
}

TEST_F(TestVectorLoggingUtils, PreservesElementOrder)
{
    const std::vector<int64_t> orderedVec = {5, 3, 8, 1, 9};
    std::ostringstream oss;
    oss << StreamVec(orderedVec);
    EXPECT_EQ(oss.str(), "[5, 3, 8, 1, 9]");
}

TEST_F(TestVectorLoggingUtils, HandlesAlternatingSignValues)
{
    const std::vector<int64_t> altVec = {1, -1, 2, -2, 3, -3};
    std::ostringstream oss;
    oss << StreamVec(altVec);
    EXPECT_EQ(oss.str(), "[1, -1, 2, -2, 3, -3]");
}
