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
#include <fstream>
#include <iterator>
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

/// The SHA-256 of the test scorer library. Computed rather than pinned because the
/// library's bytes differ per toolchain.
std::string libraryHash()
{
    std::ifstream file(getTestScorerLibPath(), std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    EXPECT_FALSE(bytes.empty()) << "the test scorer library is missing: " << getTestScorerLibPath();
    return sha256(bytes);
}

/// Loads @p symbol from the test scorer library, declaring the library's real digest.
std::unique_ptr<CustomLibraryAdapter> loadScorer(const std::string& symbol, size_t numFeatures)
{
    return CustomLibraryAdapter::load(
        getTestScorerLibPath(), symbol, numFeatures, TEST_HASH, libraryHash());
}

} // namespace

class TestCustomLibraryAdapter : public ::testing::Test
{
};

TEST_F(TestCustomLibraryAdapter, LoadAndScoreLinear)
{
    auto adapter = loadScorer("testLinearScorer", 3);
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
    auto adapter = loadScorer("testConstantScorer", 2);
    ASSERT_NE(adapter, nullptr);

    // testConstantScorer always returns 42.0
    EXPECT_DOUBLE_EQ(adapter->score({1.0, 2.0}), 42.0);
    EXPECT_DOUBLE_EQ(adapter->score({999.0, -100.0}), 42.0);
}

TEST_F(TestCustomLibraryAdapter, LoadAndScoreProduct)
{
    auto adapter = loadScorer("testProductScorer", 2);
    ASSERT_NE(adapter, nullptr);

    // testProductScorer multiplies first two features
    EXPECT_DOUBLE_EQ(adapter->score({3.0, 4.0}), 12.0);
    EXPECT_DOUBLE_EQ(adapter->score({0.0, 5.0}), 0.0);
    EXPECT_DOUBLE_EQ(adapter->score({-2.0, 3.0}), -6.0);
}

TEST_F(TestCustomLibraryAdapter, LoadFailsMissingLibrary)
{
    auto adapter = CustomLibraryAdapter::load(
        "/nonexistent/path/to/library.so", "some_symbol", 2, TEST_HASH, libraryHash());
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestCustomLibraryAdapter, LoadFailsMissingSymbol)
{
    auto adapter = loadScorer("nonexistent_symbol", 2);
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestCustomLibraryAdapter, LoadFailsEmptyLibraryPath)
{
    auto adapter = CustomLibraryAdapter::load("", "testLinearScorer", 2, TEST_HASH, libraryHash());
    EXPECT_EQ(adapter, nullptr);
}

/// Loading runs the library's initialisers, so a library declaring no digest is never opened.
TEST_F(TestCustomLibraryAdapter, LoadRefusesALibraryDeclaringNoHash)
{
    EXPECT_EQ(
        CustomLibraryAdapter::load(getTestScorerLibPath(), "testLinearScorer", 3, TEST_HASH, ""),
        nullptr);
}

TEST_F(TestCustomLibraryAdapter, LoadFailsEmptySymbolName)
{
    auto adapter = loadScorer("", 2);
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestCustomLibraryAdapter, ScoreThrowsOnFeatureCountMismatch)
{
    auto adapter = loadScorer("testLinearScorer", 3);
    ASSERT_NE(adapter, nullptr);

    EXPECT_THROW(adapter->score({1.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(adapter->score({1.0, 2.0, 3.0, 4.0}), std::invalid_argument);
}

TEST_F(TestCustomLibraryAdapter, ScoreBatch)
{
    auto adapter = loadScorer("testLinearScorer", 2);
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
    auto adapter1 = loadScorer("testLinearScorer", 2);
    auto adapter2 = loadScorer("testConstantScorer", 2);

    ASSERT_NE(adapter1, nullptr);
    ASSERT_NE(adapter2, nullptr);

    EXPECT_DOUBLE_EQ(adapter1->score({1.0, 2.0}), 3.0);
    EXPECT_DOUBLE_EQ(adapter2->score({1.0, 2.0}), 42.0);

    // Each load holds its own dlopen reference.
    adapter1.reset();
    EXPECT_DOUBLE_EQ(adapter2->score({999.0, -100.0}), 42.0);
}
