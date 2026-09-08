// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestCustomLibraryAdapter.cpp
 * @brief Tests for CustomLibraryAdapter (compiled scorer .so) per RFC 0019 §7.2.
 *
 * Tests cover:
 * - Loading .so and resolving symbols
 * - Calling scorer function with feature vectors
 * - Error paths (missing library, missing symbol)
 * - Features hash validation
 * - Lifecycle (dlopen/dlclose)
 */

#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/CustomLibraryAdapter.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_plugin_sdk::ingestor::uhd;

namespace
{
constexpr const char* TEST_HASH = "sha256:test_hash_12345678";

// Helper to get the path to the test scorer library.
// The library is built as hipdnn_test_scorer_lib.so/.dll and placed in the test plugin dir.
std::string getTestScorerLibPath()
{
#ifdef _WIN32
    return std::string(HIPDNN_TEST_PLUGIN_DIR) + "/hipdnn_test_scorer_lib.dll";
#else
    return std::string(HIPDNN_TEST_PLUGIN_DIR) + "/libhipdnn_test_scorer_lib.so";
#endif
}

} // namespace

class TestCustomLibraryAdapter : public ::testing::Test
{
};

TEST_F(TestCustomLibraryAdapter, LoadAndScoreLinear)
{
    const auto libPath = getTestScorerLibPath();
    auto adapter = CustomLibraryAdapter::load(libPath, "test_linear_scorer", 3, TEST_HASH);
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->type(), UhdAdapterType::CUSTOM_LIBRARY);
    EXPECT_EQ(adapter->expectedFeatureCount(), 3U);
    EXPECT_EQ(adapter->getFeaturesHash(), TEST_HASH);

    // test_linear_scorer sums all features
    EXPECT_DOUBLE_EQ(adapter->score({1.0, 2.0, 3.0}), 6.0);
    EXPECT_DOUBLE_EQ(adapter->score({0.0, 0.0, 0.0}), 0.0);
    EXPECT_DOUBLE_EQ(adapter->score({-1.0, 5.0, 2.0}), 6.0);
}

TEST_F(TestCustomLibraryAdapter, LoadAndScoreConstant)
{
    const auto libPath = getTestScorerLibPath();
    auto adapter = CustomLibraryAdapter::load(libPath, "test_constant_scorer", 2, TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    // test_constant_scorer always returns 42.0
    EXPECT_DOUBLE_EQ(adapter->score({1.0, 2.0}), 42.0);
    EXPECT_DOUBLE_EQ(adapter->score({999.0, -100.0}), 42.0);
}

TEST_F(TestCustomLibraryAdapter, LoadAndScoreProduct)
{
    const auto libPath = getTestScorerLibPath();
    auto adapter = CustomLibraryAdapter::load(libPath, "test_product_scorer", 2, TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    // test_product_scorer multiplies first two features
    EXPECT_DOUBLE_EQ(adapter->score({3.0, 4.0}), 12.0);
    EXPECT_DOUBLE_EQ(adapter->score({0.0, 5.0}), 0.0);
    EXPECT_DOUBLE_EQ(adapter->score({-2.0, 3.0}), -6.0);
}

TEST_F(TestCustomLibraryAdapter, LoadFailsMissingLibrary)
{
    auto adapter = CustomLibraryAdapter::load("/nonexistent/path/to/library.so",
                                               "some_symbol",
                                               2,
                                               TEST_HASH);
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
    auto adapter = CustomLibraryAdapter::load("", "test_linear_scorer", 2, TEST_HASH);
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
    auto adapter = CustomLibraryAdapter::load(libPath, "test_linear_scorer", 3, TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    // Adapter expects 3 features, provide 2 -> should throw
    EXPECT_THROW(adapter->score({1.0, 2.0}), std::invalid_argument);

    // Adapter expects 3 features, provide 4 -> should throw
    EXPECT_THROW(adapter->score({1.0, 2.0, 3.0, 4.0}), std::invalid_argument);
}

TEST_F(TestCustomLibraryAdapter, ScoreBatch)
{
    const auto libPath = getTestScorerLibPath();
    auto adapter = CustomLibraryAdapter::load(libPath, "test_linear_scorer", 2, TEST_HASH);
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

    // Load two different symbols from the same library
    auto adapter1 = CustomLibraryAdapter::load(libPath, "test_linear_scorer", 2, TEST_HASH);
    auto adapter2 = CustomLibraryAdapter::load(libPath, "test_constant_scorer", 2, TEST_HASH);

    ASSERT_NE(adapter1, nullptr);
    ASSERT_NE(adapter2, nullptr);

    // Both should work independently
    EXPECT_DOUBLE_EQ(adapter1->score({1.0, 2.0}), 3.0);
    EXPECT_DOUBLE_EQ(adapter2->score({1.0, 2.0}), 42.0);

    // Destroying one shouldn't affect the other (dlclose is independent per load)
    adapter1.reset();
    EXPECT_DOUBLE_EQ(adapter2->score({999.0, -100.0}), 42.0);
}
