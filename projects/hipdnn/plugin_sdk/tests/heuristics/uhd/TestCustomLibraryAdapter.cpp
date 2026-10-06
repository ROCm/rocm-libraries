// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestCustomLibraryAdapter.cpp
 * @brief Tests for CustomLibraryAdapter (compiled scorer library) per RFC 0019 §7.3.
 */

#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/CustomLibraryAdapter.hpp>

#include "../../TestResourcePaths.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_plugin_sdk::uhd;

namespace
{
constexpr const char* TEST_HASH = "sha256:test_hash_12345678";

std::string getTestScorerLibPath()
{
    return hipdnn_plugin_sdk::test::testScorerLibrary().string();
}

} // namespace

class TestCustomLibraryAdapter : public ::testing::Test
{
};

TEST_F(TestCustomLibraryAdapter, LoadAndScoreLinear)
{
    const auto libPath = getTestScorerLibPath();
    auto adapter = CustomLibraryAdapter::load(libPath, "testLinearScorer", 3, TEST_HASH);
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->expectedFeatureCount(), 3U);
    EXPECT_EQ(adapter->getFeaturesHash(), TEST_HASH);

    // testLinearScorer sums all features
    EXPECT_DOUBLE_EQ(adapter->score({1.0, 2.0, 3.0}), 6.0);
    EXPECT_DOUBLE_EQ(adapter->score({0.0, 0.0, 0.0}), 0.0);
    EXPECT_DOUBLE_EQ(adapter->score({-1.0, 5.0, 2.0}), 6.0);
}

TEST_F(TestCustomLibraryAdapter, LoadAndScoreConstant)
{
    const auto libPath = getTestScorerLibPath();
    auto adapter = CustomLibraryAdapter::load(libPath, "testConstantScorer", 2, TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    // testConstantScorer always returns 42.0
    EXPECT_DOUBLE_EQ(adapter->score({1.0, 2.0}), 42.0);
    EXPECT_DOUBLE_EQ(adapter->score({999.0, -100.0}), 42.0);
}

TEST_F(TestCustomLibraryAdapter, LoadAndScoreProduct)
{
    const auto libPath = getTestScorerLibPath();
    auto adapter = CustomLibraryAdapter::load(libPath, "testProductScorer", 2, TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    // testProductScorer multiplies first two features
    EXPECT_DOUBLE_EQ(adapter->score({3.0, 4.0}), 12.0);
    EXPECT_DOUBLE_EQ(adapter->score({0.0, 5.0}), 0.0);
    EXPECT_DOUBLE_EQ(adapter->score({-2.0, 3.0}), -6.0);
}

TEST_F(TestCustomLibraryAdapter, LoadFailsMissingLibrary)
{
    auto adapter = CustomLibraryAdapter::load(
        "/nonexistent/path/to/library.so", "some_symbol", 2, TEST_HASH);
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestCustomLibraryAdapter, LoadFailsMissingSymbol)
{
    const auto libPath = getTestScorerLibPath();
    auto adapter = CustomLibraryAdapter::load(libPath, "nonexistent_symbol", 2, TEST_HASH);
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestCustomLibraryAdapter, LoadFailsEmptyLibraryPath)
{
    auto adapter = CustomLibraryAdapter::load("", "testLinearScorer", 2, TEST_HASH);
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestCustomLibraryAdapter, LoadFailsEmptySymbolName)
{
    const auto libPath = getTestScorerLibPath();
    auto adapter = CustomLibraryAdapter::load(libPath, "", 2, TEST_HASH);
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestCustomLibraryAdapter, ScoreThrowsOnFeatureCountMismatch)
{
    const auto libPath = getTestScorerLibPath();
    auto adapter = CustomLibraryAdapter::load(libPath, "testLinearScorer", 3, TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    EXPECT_THROW(adapter->score({1.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(adapter->score({1.0, 2.0, 3.0, 4.0}), std::invalid_argument);
}

TEST_F(TestCustomLibraryAdapter, ScoreBatch)
{
    const auto libPath = getTestScorerLibPath();
    auto adapter = CustomLibraryAdapter::load(libPath, "testLinearScorer", 2, TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<std::vector<double>> batch = {{1.0, 2.0}, {3.0, 4.0}, {0.0, 0.0}};
    auto scores = adapter->scoreBatch(batch);
    ASSERT_EQ(scores.size(), 3U);
    EXPECT_DOUBLE_EQ(scores[0], 3.0);
    EXPECT_DOUBLE_EQ(scores[1], 7.0);
    EXPECT_DOUBLE_EQ(scores[2], 0.0);
}

TEST_F(TestCustomLibraryAdapter, MultipleAdaptersFromSameLibrary)
{
    const auto libPath = getTestScorerLibPath();

    auto adapter1 = CustomLibraryAdapter::load(libPath, "testLinearScorer", 2, TEST_HASH);
    auto adapter2 = CustomLibraryAdapter::load(libPath, "testConstantScorer", 2, TEST_HASH);

    ASSERT_NE(adapter1, nullptr);
    ASSERT_NE(adapter2, nullptr);

    EXPECT_DOUBLE_EQ(adapter1->score({1.0, 2.0}), 3.0);
    EXPECT_DOUBLE_EQ(adapter2->score({1.0, 2.0}), 42.0);

    // Each load holds its own dlopen reference.
    adapter1.reset();
    EXPECT_DOUBLE_EQ(adapter2->score({999.0, -100.0}), 42.0);
}
