// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestUhdAdapters.cpp
 * @brief Tests for UHD model adapters (StaticOrder, TreeData).
 */

#include "heuristics/uhd/adapters/IUhdAdapter.hpp"
#include "heuristics/uhd/adapters/StaticOrderAdapter.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

using hipdnn_backend::heuristics::uhd::StaticOrderAdapter;
using hipdnn_backend::heuristics::uhd::UhdAdapterType;

namespace
{

// ========== StaticOrderAdapter tests ==========

class TestStaticOrderAdapter : public ::testing::Test
{
protected:
    // Signature: [priority, id, tile_m, tile_n]
    std::vector<std::string> _signature = {
        "\"$kernel.priority\"",
        "\"$kernel.id\"",
        "\"$kernel.tile_m\"",
        "\"$kernel.tile_n\"",
    };
};

TEST_F(TestStaticOrderAdapter, CreateFromFieldNames)
{
    std::vector<std::string> orderFields = {"priority", "id"};
    auto adapter = StaticOrderAdapter::create(orderFields, _signature);
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->type(), UhdAdapterType::StaticOrder);
}

TEST_F(TestStaticOrderAdapter, CreateWithKernelPrefixMatch)
{
    // signature has "$kernel.priority" but we pass "priority"
    std::vector<std::string> orderFields = {"priority"};
    auto adapter = StaticOrderAdapter::create(orderFields, _signature);
    ASSERT_NE(adapter, nullptr);
}

TEST_F(TestStaticOrderAdapter, CreateFailsOnUnknownField)
{
    std::vector<std::string> orderFields = {"unknown_field"};
    auto adapter = StaticOrderAdapter::create(orderFields, _signature);
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestStaticOrderAdapter, ReportsCorrectFeatureCount)
{
    std::vector<std::string> orderFields = {"priority"};
    auto adapter = StaticOrderAdapter::create(orderFields, _signature);
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->expectedFeatureCount(), 4u);
}

TEST_F(TestStaticOrderAdapter, ValidatesFeatureCount)
{
    std::vector<std::string> orderFields = {"priority"};
    auto adapter = StaticOrderAdapter::create(orderFields, _signature);
    ASSERT_NE(adapter, nullptr);

    EXPECT_TRUE(adapter->validateFeatureCount(4));
    EXPECT_FALSE(adapter->validateFeatureCount(3));
    EXPECT_FALSE(adapter->validateFeatureCount(5));
}

TEST_F(TestStaticOrderAdapter, EmptyFeaturesHash)
{
    std::vector<std::string> orderFields = {"priority"};
    auto adapter = StaticOrderAdapter::create(orderFields, _signature);
    ASSERT_NE(adapter, nullptr);

    // StaticOrder doesn't use a trained model, so hash is empty
    EXPECT_TRUE(adapter->getFeaturesHash().empty());
}

TEST_F(TestStaticOrderAdapter, ScoresLowerPriorityHigher)
{
    // Order by priority (index 0)
    std::vector<std::string> orderFields = {"priority"};
    auto adapter = StaticOrderAdapter::create(orderFields, _signature);
    ASSERT_NE(adapter, nullptr);

    // Lower priority value should get higher score
    std::vector<double> lowPriority = {1.0, 100, 64, 64};
    std::vector<double> highPriority = {10.0, 200, 64, 64};

    double scoreLow = adapter->score(lowPriority);
    double scoreHigh = adapter->score(highPriority);

    EXPECT_GT(scoreLow, scoreHigh);
}

TEST_F(TestStaticOrderAdapter, OrderByMultipleFields)
{
    // Order by priority then id
    std::vector<std::string> orderFields = {"priority", "id"};
    auto adapter = StaticOrderAdapter::create(orderFields, _signature);
    ASSERT_NE(adapter, nullptr);

    // Same priority, different id - lower id should win
    std::vector<double> samePriorityLowId = {5.0, 1.0, 64, 64};
    std::vector<double> samePriorityHighId = {5.0, 10.0, 64, 64};

    double scoreLowId = adapter->score(samePriorityLowId);
    double scoreHighId = adapter->score(samePriorityHighId);

    EXPECT_GT(scoreLowId, scoreHighId);
}

TEST_F(TestStaticOrderAdapter, PrimaryFieldDominates)
{
    // Order by priority then id
    std::vector<std::string> orderFields = {"priority", "id"};
    auto adapter = StaticOrderAdapter::create(orderFields, _signature);
    ASSERT_NE(adapter, nullptr);

    // Lower priority should win even with higher id
    std::vector<double> lowPriorityHighId = {1.0, 1000.0, 64, 64};
    std::vector<double> highPriorityLowId = {10.0, 1.0, 64, 64};

    double scoreLowPriority = adapter->score(lowPriorityHighId);
    double scoreHighPriority = adapter->score(highPriorityLowId);

    EXPECT_GT(scoreLowPriority, scoreHighPriority);
}

TEST_F(TestStaticOrderAdapter, ConstructorWithIndices)
{
    // Direct construction with indices: order by index 0 (priority)
    std::vector<size_t> indices = {0};
    StaticOrderAdapter adapter(indices, 4);

    EXPECT_EQ(adapter.type(), UhdAdapterType::StaticOrder);
    EXPECT_EQ(adapter.expectedFeatureCount(), 4u);

    std::vector<double> features = {5.0, 100, 64, 64};
    double score = adapter.score(features);
    EXPECT_NE(score, 0.0);
}

TEST_F(TestStaticOrderAdapter, BatchScoring)
{
    std::vector<std::string> orderFields = {"priority"};
    auto adapter = StaticOrderAdapter::create(orderFields, _signature);
    ASSERT_NE(adapter, nullptr);

    std::vector<std::vector<double>> batch = {
        {1.0, 100, 64, 64},
        {5.0, 200, 64, 64},
        {3.0, 300, 64, 64},
    };

    auto scores = adapter->scoreBatch(batch);
    ASSERT_EQ(scores.size(), 3u);

    // Priority 1 should have highest score, then 3, then 5
    EXPECT_GT(scores[0], scores[2]);
    EXPECT_GT(scores[2], scores[1]);
}

} // namespace
