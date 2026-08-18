// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestTreeDataAdapter.cpp
 * @brief Tests for TreeDataAdapter (GBDT tree walker) per RFC 0019 §8.1.
 *
 * Tests cover:
 * - Model loading from buffer
 * - Features hash validation (contract enforcement per RFC §7.3)
 * - Single tree evaluation
 * - Multi-tree ensemble scoring
 * - Missing value handling (NaN with default_left)
 * - Edge cases (empty model, invalid buffer)
 */

#include "heuristics/uhd/adapters/TreeDataAdapter.hpp"
#include "heuristics/uhd/Sha256.hpp"

#include "GbdtModelTestBuilder.hpp"

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/gbdt_model_generated.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

using hipdnn_backend::heuristics::uhd::TreeDataAdapter;
using hipdnn_backend::heuristics::uhd::UhdAdapterType;

namespace
{

/// Shared with TestUhdSelectionFlow so both suites build model artifacts the same way.
using GbdtModelBuilder = hipdnn_backend::heuristics::uhd::testing::GbdtModelTestBuilder;

/// Create a simple single-node tree (just a leaf).
GbdtModelBuilder::TreeSpec makeLeafTree(double leafValue)
{
    GbdtModelBuilder::TreeSpec spec;
    spec.featureIndices = {0};
    spec.thresholds = {0.0};
    spec.leftChildren = {-1}; // leaf
    spec.rightChildren = {-1};
    spec.leafValues = {leafValue};
    spec.defaultLeft = {1};
    return spec;
}

/// Create a simple binary split tree:
///       [0]  (feature 0 < threshold)
///      /   \
///   [1]     [2]
///  leaf    leaf
GbdtModelBuilder::TreeSpec
    makeBinarySplitTree(int32_t featureIdx, double threshold, double leftLeaf, double rightLeaf)
{
    GbdtModelBuilder::TreeSpec spec;
    spec.featureIndices = {featureIdx, 0, 0};
    spec.thresholds = {threshold, 0.0, 0.0};
    spec.leftChildren = {1, -1, -1}; // node 0 -> left=1, nodes 1,2 are leaves
    spec.rightChildren = {2, -1, -1}; // node 0 -> right=2
    spec.leafValues = {0.0, leftLeaf, rightLeaf};
    spec.defaultLeft = {1, 1, 1}; // default left on missing
    return spec;
}

/// Create a deeper tree with multiple splits:
///           [0]  (feature 0 < 5.0)
///          /   \
///       [1]     [2]  (feature 1 < 10.0)
///      leaf    /   \
///           [3]     [4]
///          leaf    leaf
GbdtModelBuilder::TreeSpec makeDeepTree()
{
    GbdtModelBuilder::TreeSpec spec;
    spec.featureIndices = {0, 0, 1, 0, 0};
    spec.thresholds = {5.0, 0.0, 10.0, 0.0, 0.0};
    spec.leftChildren = {1, -1, 3, -1, -1};
    spec.rightChildren = {2, -1, 4, -1, -1};
    spec.leafValues = {0.0, 1.0, 0.0, 2.0, 3.0};
    spec.defaultLeft = {1, 1, 1, 1, 1};
    return spec;
}

// ========== Test Fixture ==========

class TestTreeDataAdapter : public ::testing::Test
{
protected:
    static constexpr const char* TEST_HASH = "sha256:test_features_hash_12345";
};

// ========== Loading Tests ==========

TEST_F(TestTreeDataAdapter, LoadFromBufferSucceeds)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeLeafTree(1.5))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);

    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->type(), UhdAdapterType::TREE_DATA);
}

TEST_F(TestTreeDataAdapter, LoadFromBufferReturnsCorrectFeatureCount)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(5)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeLeafTree(1.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);

    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->expectedFeatureCount(), 5u);
}

TEST_F(TestTreeDataAdapter, LoadFromBufferReturnsCorrectTreeCount)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeLeafTree(1.0))
                      .addTree(makeLeafTree(2.0))
                      .addTree(makeLeafTree(3.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);

    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->treeCount(), 3u);
}

TEST_F(TestTreeDataAdapter, LoadFromBufferStoresFeaturesHash)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeLeafTree(1.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);

    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->getFeaturesHash(), TEST_HASH);
}

// ========== Contract Enforcement Tests (RFC §7.3) ==========

TEST_F(TestTreeDataAdapter, LoadFailsOnHashMismatch)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash("sha256:model_hash")
                      .addTree(makeLeafTree(1.0))
                      .build();

    auto adapter
        = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "sha256:different_hash");

    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestTreeDataAdapter, LoadSucceedsWithEmptyExpectedHash)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash("sha256:any_hash")
                      .addTree(makeLeafTree(1.0))
                      .build();

    // Empty expected hash means skip validation
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "");

    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->getFeaturesHash(), "sha256:any_hash");
}

TEST_F(TestTreeDataAdapter, LoadFailsOnInvalidBuffer)
{
    std::vector<uint8_t> garbage = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05};

    auto adapter = TreeDataAdapter::loadFromBuffer(garbage.data(), garbage.size(), TEST_HASH);

    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestTreeDataAdapter, LoadFailsOnEmptyBuffer)
{
    std::vector<uint8_t> empty;

    auto adapter = TreeDataAdapter::loadFromBuffer(empty.data(), 0, TEST_HASH);

    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestTreeDataAdapter, LoadFailsOnNullBuffer)
{
    auto adapter = TreeDataAdapter::loadFromBuffer(nullptr, 100, TEST_HASH);

    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestTreeDataAdapter, LoadFailsOnTooSmallBuffer)
{
    // Buffer must be at least large enough to have a file identifier
    std::vector<uint8_t> tooSmall = {0x00, 0x00, 0x00, 0x04}; // Just a size prefix

    auto adapter = TreeDataAdapter::loadFromBuffer(tooSmall.data(), tooSmall.size(), TEST_HASH);

    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestTreeDataAdapter, LoadFailsOnWrongFileIdentifier)
{
    // Build a valid buffer but corrupt the file identifier
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeLeafTree(1.0))
                      .build();

    // Corrupt the file identifier (last 4 bytes before root table offset)
    if(buffer.size() >= 8)
    {
        buffer[4] = 'X';
        buffer[5] = 'X';
        buffer[6] = 'X';
        buffer[7] = 'X';
    }

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);

    EXPECT_EQ(adapter, nullptr);
}

// ========== Single Tree Scoring Tests ==========

TEST_F(TestTreeDataAdapter, ScoreLeafOnlyTree)
{
    const double leafValue = 2.5;
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .setBaseScore(0.0)
                      .setLearningRate(1.0)
                      .addTree(makeLeafTree(leafValue))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {1.0, 2.0};
    const double score = adapter->score(features);

    EXPECT_DOUBLE_EQ(score, leafValue);
}

TEST_F(TestTreeDataAdapter, ScoreBinarySplitGoesLeft)
{
    // Tree: if feature[0] < 5.0 then 10.0 else 20.0
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeBinarySplitTree(0, 5.0, 10.0, 20.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {3.0}; // < 5.0, should go left
    const double score = adapter->score(features);

    EXPECT_DOUBLE_EQ(score, 10.0);
}

TEST_F(TestTreeDataAdapter, ScoreBinarySplitGoesRight)
{
    // Tree: if feature[0] < 5.0 then 10.0 else 20.0
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeBinarySplitTree(0, 5.0, 10.0, 20.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {7.0}; // >= 5.0, should go right
    const double score = adapter->score(features);

    EXPECT_DOUBLE_EQ(score, 20.0);
}

TEST_F(TestTreeDataAdapter, ScoreBinarySplitAtThreshold)
{
    // Tree: if feature[0] <= 5.0 then 10.0 else 20.0
    // At exactly threshold, should go LEFT (LightGBM uses <= by default)
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeBinarySplitTree(0, 5.0, 10.0, 20.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {5.0}; // == 5.0, should go left with <= comparison
    const double score = adapter->score(features);

    EXPECT_DOUBLE_EQ(score, 10.0);
}

TEST_F(TestTreeDataAdapter, ScoreDeepTreePath1)
{
    // Deep tree: feature[0] < 5.0 -> leaf(1.0)
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeDeepTree())
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {3.0, 15.0}; // feature[0] < 5.0
    const double score = adapter->score(features);

    EXPECT_DOUBLE_EQ(score, 1.0);
}

TEST_F(TestTreeDataAdapter, ScoreDeepTreePath2)
{
    // Deep tree: feature[0] >= 5.0 && feature[1] < 10.0 -> leaf(2.0)
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeDeepTree())
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {7.0, 5.0}; // feature[0] >= 5.0, feature[1] < 10.0
    const double score = adapter->score(features);

    EXPECT_DOUBLE_EQ(score, 2.0);
}

TEST_F(TestTreeDataAdapter, ScoreDeepTreePath3)
{
    // Deep tree: feature[0] >= 5.0 && feature[1] >= 10.0 -> leaf(3.0)
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeDeepTree())
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {7.0, 15.0}; // feature[0] >= 5.0, feature[1] >= 10.0
    const double score = adapter->score(features);

    EXPECT_DOUBLE_EQ(score, 3.0);
}

// ========== Multi-Tree Ensemble Tests ==========

TEST_F(TestTreeDataAdapter, ScoreMultipleTreesSumsLeaves)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .setBaseScore(0.0)
                      .setLearningRate(1.0)
                      .addTree(makeLeafTree(1.0))
                      .addTree(makeLeafTree(2.0))
                      .addTree(makeLeafTree(3.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {0.0};
    const double score = adapter->score(features);

    // score = base_score + learning_rate * sum(leaf_values)
    // score = 0.0 + 1.0 * (1.0 + 2.0 + 3.0) = 6.0
    EXPECT_DOUBLE_EQ(score, 6.0);
}

TEST_F(TestTreeDataAdapter, ScoreAppliesBaseScore)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .setBaseScore(100.0)
                      .setLearningRate(1.0)
                      .addTree(makeLeafTree(5.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {0.0};
    const double score = adapter->score(features);

    // score = base_score + learning_rate * sum(leaf_values)
    // score = 100.0 + 1.0 * 5.0 = 105.0
    EXPECT_DOUBLE_EQ(score, 105.0);
}

TEST_F(TestTreeDataAdapter, ScoreAppliesLearningRate)
{
    // NOTE: LightGBM's dump_model() returns leaf_values that ALREADY include
    // learning_rate multiplication. We test with pre-scaled values here.
    // If learning_rate=0.1 and "raw" leaves were 100.0 and 200.0,
    // then exported leaf_values would be 10.0 and 20.0.
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .setBaseScore(0.0)
                      .setLearningRate(0.1) // Stored for metadata, not used in score()
                      .addTree(makeLeafTree(10.0)) // Pre-scaled: 0.1 * 100.0
                      .addTree(makeLeafTree(20.0)) // Pre-scaled: 0.1 * 200.0
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {0.0};
    const double score = adapter->score(features);

    // score = base_score + sum(leaf_values)
    // score = 0.0 + (10.0 + 20.0) = 30.0
    // (learning_rate is NOT applied again since leaf_values are pre-scaled)
    EXPECT_DOUBLE_EQ(score, 30.0);
}

TEST_F(TestTreeDataAdapter, ScoreWithBaseScoreAndLearningRate)
{
    // Same principle: leaf_values from LightGBM are pre-scaled by learning_rate.
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .setBaseScore(50.0)
                      .setLearningRate(0.5) // Metadata only
                      .addTree(makeLeafTree(10.0)) // Pre-scaled
                      .addTree(makeLeafTree(20.0)) // Pre-scaled
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {0.0};
    const double score = adapter->score(features);

    // score = base_score + sum(leaf_values)
    // score = 50.0 + (10.0 + 20.0) = 80.0
    EXPECT_DOUBLE_EQ(score, 80.0);
}

// ========== Missing Value Handling Tests ==========

TEST_F(TestTreeDataAdapter, ScoreWithNaNUsesDefaultLeft)
{
    // Tree with default_left = true: NaN should go left
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeBinarySplitTree(0, 5.0, 10.0, 20.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {std::numeric_limits<double>::quiet_NaN()};
    const double score = adapter->score(features);

    // NaN with default_left=true should go left -> 10.0
    EXPECT_DOUBLE_EQ(score, 10.0);
}

TEST_F(TestTreeDataAdapter, ScoreWithInfinity)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeBinarySplitTree(0, 5.0, 10.0, 20.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    // Positive infinity should go right (>= threshold)
    const std::vector<double> featuresPos = {std::numeric_limits<double>::infinity()};
    EXPECT_DOUBLE_EQ(adapter->score(featuresPos), 20.0);

    // Negative infinity should go left (< threshold)
    const std::vector<double> featuresNeg = {-std::numeric_limits<double>::infinity()};
    EXPECT_DOUBLE_EQ(adapter->score(featuresNeg), 10.0);
}

// ========== Edge Cases ==========

TEST_F(TestTreeDataAdapter, ScoreWithNoTrees)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .setBaseScore(42.0)
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {1.0, 2.0};
    const double score = adapter->score(features);

    // With no trees, should return base_score
    EXPECT_DOUBLE_EQ(score, 42.0);
}

TEST_F(TestTreeDataAdapter, ScoreWithEmptyFeatureVector)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(0)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeLeafTree(5.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features;
    const double score = adapter->score(features);

    // Leaf-only tree should still return leaf value
    EXPECT_DOUBLE_EQ(score, 5.0);
}

TEST_F(TestTreeDataAdapter, ValidatesFeatureCount)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(3)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeLeafTree(1.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    EXPECT_TRUE(adapter->validateFeatureCount(3));
    EXPECT_FALSE(adapter->validateFeatureCount(2));
    EXPECT_FALSE(adapter->validateFeatureCount(4));
}

TEST_F(TestTreeDataAdapter, BatchScoring)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeBinarySplitTree(0, 5.0, 10.0, 20.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    // With <= comparison (LightGBM default), values <= 5.0 go left (10.0)
    const std::vector<std::vector<double>> batch = {
        {3.0}, // <= 5.0 -> 10.0
        {7.0}, // > 5.0 -> 20.0
        {5.0}, // <= 5.0 -> 10.0 (at threshold, goes left with <=)
        {-1.0}, // <= 5.0 -> 10.0
    };

    const auto scores = adapter->scoreBatch(batch);

    ASSERT_EQ(scores.size(), 4u);
    EXPECT_DOUBLE_EQ(scores[0], 10.0);
    EXPECT_DOUBLE_EQ(scores[1], 20.0);
    EXPECT_DOUBLE_EQ(scores[2], 10.0); // Now goes left with <= comparison
    EXPECT_DOUBLE_EQ(scores[3], 10.0);
}

// ========== Integration with SelectionEngine ==========

TEST_F(TestTreeDataAdapter, WorksWithFeatureExtractor)
{
    // This tests that TreeDataAdapter integrates properly with the UHD flow
    // by building a model that uses multiple features in realistic ways

    // Tree that prefers higher tile sizes and lower priority:
    // if tile_m (feature 0) >= 128:
    //   if priority (feature 1) < 5: score = 3.0
    //   else: score = 2.0
    // else: score = 1.0
    GbdtModelBuilder::TreeSpec realisticTree;
    realisticTree.featureIndices = {0, 1, 0, 0, 0};
    realisticTree.thresholds = {128.0, 5.0, 0.0, 0.0, 0.0};
    realisticTree.leftChildren = {2, 3, -1, -1, -1}; // node 0: left=2, node 1: left=3
    realisticTree.rightChildren = {1, 4, -1, -1, -1}; // node 0: right=1, node 1: right=4
    realisticTree.leafValues = {0.0, 0.0, 1.0, 3.0, 2.0};
    realisticTree.defaultLeft = {0, 1, 1, 1, 1};

    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(realisticTree)
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    // Test different kernel configurations
    // High tile, low priority -> best score (3.0)
    EXPECT_DOUBLE_EQ(adapter->score({256.0, 1.0}), 3.0);

    // High tile, high priority -> medium score (2.0)
    EXPECT_DOUBLE_EQ(adapter->score({256.0, 10.0}), 2.0);

    // Low tile -> low score (1.0)
    EXPECT_DOUBLE_EQ(adapter->score({64.0, 1.0}), 1.0);
}

// ========== RFC 0019 §9.2: Training arches and model version ==========

TEST_F(TestTreeDataAdapter, ReturnsEmptyTrainingArchesByDefault)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeLeafTree(1.0))
                      .build();
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    EXPECT_TRUE(adapter->getTrainingArches().empty());
}

TEST_F(TestTreeDataAdapter, ReturnsTrainingArchesWhenSet)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .setTrainingArches({"gfx942", "gfx1100"})
                      .addTree(makeLeafTree(1.0))
                      .build();
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    auto arches = adapter->getTrainingArches();
    ASSERT_EQ(arches.size(), 2u);
    EXPECT_EQ(arches[0], "gfx942");
    EXPECT_EQ(arches[1], "gfx1100");
}

TEST_F(TestTreeDataAdapter, IsTrainedForArchReturnsTrueWhenNoArches)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeLeafTree(1.0))
                      .build();
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    // Should return true for any arch when no restriction
    EXPECT_TRUE(adapter->isTrainedForArch("gfx942"));
    EXPECT_TRUE(adapter->isTrainedForArch("gfx950"));
    EXPECT_TRUE(adapter->isTrainedForArch("anything"));
}

TEST_F(TestTreeDataAdapter, IsTrainedForArchReturnsTrueWhenArchInList)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .setTrainingArches({"gfx942", "gfx1100"})
                      .addTree(makeLeafTree(1.0))
                      .build();
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    EXPECT_TRUE(adapter->isTrainedForArch("gfx942"));
    EXPECT_TRUE(adapter->isTrainedForArch("gfx1100"));
}

TEST_F(TestTreeDataAdapter, IsTrainedForArchReturnsFalseWhenArchNotInList)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .setTrainingArches({"gfx942", "gfx1100"})
                      .addTree(makeLeafTree(1.0))
                      .build();
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    EXPECT_FALSE(adapter->isTrainedForArch("gfx950"));
    EXPECT_FALSE(adapter->isTrainedForArch("gfx900"));
}

TEST_F(TestTreeDataAdapter, ReturnsEmptyModelVersionByDefault)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeLeafTree(1.0))
                      .build();
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    EXPECT_TRUE(adapter->getModelVersion().empty());
}

TEST_F(TestTreeDataAdapter, ReturnsModelVersionWhenSet)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .setModelVersion("1.2.3")
                      .addTree(makeLeafTree(1.0))
                      .build();
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    EXPECT_EQ(adapter->getModelVersion(), "1.2.3");
}

// ========== Realistic Multi-Tree Ensemble Test ==========
// This test simulates the structure lgbm_to_flatbuffer.py would produce:
// - Multiple trees (typical LightGBM has 100-500)
// - Each tree uses different feature indices
// - Leaf values represent log1p(TFLOPS) predictions
// - Base score starts at mean of training target

/// Create a tree that splits on multiple features at different depths.
/// This mimics LightGBM's tree structure for kernel selection:
/// Features: 0=M, 1=N, 2=K, 3=tile_m, 4=cu_count
///
/// Tree structure:
///           [0] M < 512
///          /         \
///       [1]           [2] tile_m < 128
///      leaf=0.3      /         \
///                  [3]           [4] cu_count < 60
///                 leaf=0.5      /         \
///                            [5]          [6]
///                          leaf=0.7     leaf=0.9
GbdtModelBuilder::TreeSpec makeRealisticTree1()
{
    GbdtModelBuilder::TreeSpec spec;
    // Node indices: 0=root, 1=left-leaf, 2=right-internal, 3=left-leaf, 4=right-internal, 5=leaf, 6=leaf
    spec.featureIndices = {0, 0, 3, 0, 4, 0, 0}; // M(0), _, tile_m(3), _, cu_count(4)
    spec.thresholds = {512.0, 0.0, 128.0, 0.0, 60.0, 0.0, 0.0};
    spec.leftChildren = {1, -1, 3, -1, 5, -1, -1};
    spec.rightChildren = {2, -1, 4, -1, 6, -1, -1};
    spec.leafValues = {0.0, 0.3, 0.0, 0.5, 0.0, 0.7, 0.9}; // Internal nodes have 0.0
    spec.defaultLeft = {1, 1, 1, 1, 1, 1, 1};
    return spec;
}

/// Second tree focuses on different feature combinations:
/// Features: 0=M, 1=N, 2=K
///
/// Tree structure:
///           [0] N < 1024
///          /         \
///       [1] K < 256    [2]
///      /      \       leaf=0.2
///    [3]       [4]
///  leaf=0.1  leaf=0.15
GbdtModelBuilder::TreeSpec makeRealisticTree2()
{
    GbdtModelBuilder::TreeSpec spec;
    spec.featureIndices = {1, 2, 0, 0, 0}; // N(1), K(2)
    spec.thresholds = {1024.0, 256.0, 0.0, 0.0, 0.0};
    spec.leftChildren = {1, 3, -1, -1, -1};
    spec.rightChildren = {2, 4, -1, -1, -1};
    spec.leafValues = {0.0, 0.0, 0.2, 0.1, 0.15};
    spec.defaultLeft = {1, 1, 1, 1, 1};
    return spec;
}

/// Third tree: simple split on a single feature (tile efficiency):
/// Features: 3=tile_m
///
/// Tree: if tile_m < 64 then -0.05 else 0.05
GbdtModelBuilder::TreeSpec makeRealisticTree3()
{
    GbdtModelBuilder::TreeSpec spec;
    spec.featureIndices = {3, 0, 0};
    spec.thresholds = {64.0, 0.0, 0.0};
    spec.leftChildren = {1, -1, -1};
    spec.rightChildren = {2, -1, -1};
    spec.leafValues = {0.0, -0.05, 0.05};
    spec.defaultLeft = {1, 1, 1};
    return spec;
}

TEST_F(TestTreeDataAdapter, RealisticGbdtEnsembleScoring)
{
    // Build a model with 3 trees, like lgbm_to_flatbuffer.py would produce
    // Features: M(0), N(1), K(2), tile_m(3), cu_count(4)
    const std::string realisticHash = "sha256:realistic_gemm";
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(5)
                      .setFeaturesHash(realisticHash)
                      .setBaseScore(3.5) // Mean of log1p(TFLOPS) in training data
                      .setLearningRate(0.1)
                      .setModelVersion("1.0.0")
                      .setTrainingArches({"gfx942", "gfx950"})
                      .addTree(makeRealisticTree1())
                      .addTree(makeRealisticTree2())
                      .addTree(makeRealisticTree3())
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), realisticHash);
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->expectedFeatureCount(), 5u);
    EXPECT_EQ(adapter->treeCount(), 3u);
    EXPECT_EQ(adapter->getModelVersion(), "1.0.0");
    EXPECT_TRUE(adapter->isTrainedForArch("gfx942"));

    // Test case 1: Small problem (M=256), small tile (tile_m=32), low CU count (cu_count=40)
    // Tree1: M<512 -> leaf=0.3
    // Tree2: N<1024 (assume N=512) -> K<256 (assume K=128) -> leaf=0.1
    // Tree3: tile_m<64 -> leaf=-0.05
    // Total: base_score + (0.3 + 0.1 + (-0.05)) = 3.5 + 0.35 = 3.85
    {
        const std::vector<double> features = {256.0, 512.0, 128.0, 32.0, 40.0};
        const double score = adapter->score(features);
        EXPECT_DOUBLE_EQ(score, 3.5 + 0.3 + 0.1 + (-0.05));
    }

    // Test case 2: Large problem (M=1024), large tile (tile_m=256), high CU count (cu_count=120)
    // Tree1: M>=512 -> tile_m>=128 -> cu_count>=60 -> leaf=0.9
    // Tree2: N>=1024 (assume N=2048) -> leaf=0.2
    // Tree3: tile_m>=64 -> leaf=0.05
    // Total: base_score + (0.9 + 0.2 + 0.05) = 3.5 + 1.15 = 4.65
    {
        const std::vector<double> features = {1024.0, 2048.0, 512.0, 256.0, 120.0};
        const double score = adapter->score(features);
        EXPECT_DOUBLE_EQ(score, 3.5 + 0.9 + 0.2 + 0.05);
    }

    // Test case 3: Medium problem exploring different tree paths
    // M=800 (>512, not <=), tile_m=64 (<=128), N=512 (<=1024), K=512 (>256, not <=)
    // Tree1: M>512 -> tile_m<=128 -> leaf=0.5
    // Tree2: N<=1024 -> K>256 -> leaf=0.15
    // Tree3: tile_m<=64 -> leaf=-0.05 (at threshold, goes left with <=)
    // Total: 3.5 + 0.5 + 0.15 + (-0.05) = 4.1
    {
        const std::vector<double> features = {800.0, 512.0, 512.0, 64.0, 80.0};
        const double score = adapter->score(features);
        EXPECT_DOUBLE_EQ(score, 3.5 + 0.5 + 0.15 + (-0.05));
    }
}

TEST_F(TestTreeDataAdapter, RealisticBatchScoringMatchesSingle)
{
    const std::string realisticHash = "sha256:batch_test";
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(5)
                      .setFeaturesHash(realisticHash)
                      .setBaseScore(3.5)
                      .addTree(makeRealisticTree1())
                      .addTree(makeRealisticTree2())
                      .addTree(makeRealisticTree3())
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), realisticHash);
    ASSERT_NE(adapter, nullptr);

    // Multiple kernel configurations to compare
    const std::vector<std::vector<double>> batch = {
        {256.0, 512.0, 128.0, 32.0, 40.0}, // Small
        {1024.0, 2048.0, 512.0, 256.0, 120.0}, // Large
        {800.0, 512.0, 512.0, 64.0, 80.0}, // Medium
        {512.0, 1024.0, 256.0, 128.0, 60.0}, // Edge case at thresholds
    };

    const auto batchScores = adapter->scoreBatch(batch);
    ASSERT_EQ(batchScores.size(), batch.size());

    // Verify batch scores match individual scores
    for(size_t i = 0; i < batch.size(); ++i)
    {
        const double singleScore = adapter->score(batch[i]);
        EXPECT_DOUBLE_EQ(batchScores[i], singleScore) << "Batch score mismatch at index " << i;
    }
}

TEST_F(TestTreeDataAdapter, RealisticRankingOrdersCorrectly)
{
    // This test verifies that the model correctly ranks kernels
    // by predicted performance (higher score = better)
    const std::string realisticHash = "sha256:ranking_test";
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(5)
                      .setFeaturesHash(realisticHash)
                      .setBaseScore(0.0)
                      .addTree(makeRealisticTree1())
                      .addTree(makeRealisticTree2())
                      .addTree(makeRealisticTree3())
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), realisticHash);
    ASSERT_NE(adapter, nullptr);

    // Score different kernel configurations and verify ordering
    struct KernelConfig
    {
        std::vector<double> features;
        std::string name;
    };

    const std::vector<KernelConfig> configs = {
        {{256.0, 512.0, 128.0, 32.0, 40.0}, "small_tile_low_cu"},
        {{1024.0, 2048.0, 512.0, 256.0, 120.0}, "large_tile_high_cu"},
        {{800.0, 512.0, 512.0, 64.0, 80.0}, "medium"},
    };

    std::vector<std::pair<double, std::string>> scored;
    scored.reserve(configs.size());
    for(const auto& cfg : configs)
    {
        scored.emplace_back(adapter->score(cfg.features), cfg.name);
    }

    // Sort by score descending (higher = better)
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // Verify: large_tile_high_cu should rank highest (uses all "good" paths)
    EXPECT_EQ(scored[0].second, "large_tile_high_cu");

    // medium should rank second
    EXPECT_EQ(scored[1].second, "medium");

    // small_tile_low_cu should rank last
    EXPECT_EQ(scored[2].second, "small_tile_low_cu");
}

// ========== Ownership Transfer Safety Tests (Dangling Pointer Fix) ==========

TEST_F(TestTreeDataAdapter, OwnershipTransferPreservesModel)
{
    // Build a model and load it
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .setBaseScore(10.0)
                      .addTree(makeBinarySplitTree(0, 5.0, 1.0, 2.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    // Clear the original buffer to ensure adapter owns its data
    std::fill(buffer.begin(), buffer.end(), static_cast<uint8_t>(0));

    // Adapter should still work correctly after original buffer is cleared
    const std::vector<double> features = {3.0}; // < 5.0 -> left -> 1.0
    const double score = adapter->score(features);
    EXPECT_DOUBLE_EQ(score, 11.0); // base_score(10.0) + leaf(1.0)
}

TEST_F(TestTreeDataAdapter, LoadFromBufferHandlesLargeBufferCorrectly)
{
    // Build a model with many trees to stress test buffer ownership
    GbdtModelBuilder builder;
    builder.setNumFeatures(5).setFeaturesHash(TEST_HASH).setBaseScore(0.0);

    // Add 100 trees
    for(int i = 0; i < 100; ++i)
    {
        builder.addTree(makeBinarySplitTree(i % 5, static_cast<double>(i), 0.01, 0.02));
    }

    auto buffer = builder.build();
    ASSERT_GT(buffer.size(), 10000u); // Should be a decent size

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->treeCount(), 100u);

    // Clear original buffer
    std::fill(buffer.begin(), buffer.end(), static_cast<uint8_t>(0));

    // Adapter should still work with many trees
    const std::vector<double> features = {50.0, 50.0, 50.0, 50.0, 50.0};
    const double score = adapter->score(features);

    // Score should be non-zero and finite
    EXPECT_FALSE(std::isnan(score));
    EXPECT_FALSE(std::isinf(score));
}

TEST_F(TestTreeDataAdapter, HashMismatchLogsWarning)
{
    // This test verifies that hash mismatch returns nullptr (the warning is logged internally)
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash("sha256:model_hash_abc")
                      .addTree(makeLeafTree(1.0))
                      .build();

    // Expected hash doesn't match model hash
    auto adapter
        = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "sha256:expected_hash_xyz");

    // Should return nullptr due to hash mismatch
    EXPECT_EQ(adapter, nullptr);
}

// ========== Bounded evaluation (RFC 0019 §16) ==========
//
// The model artifact is author-controlled input. FlatBuffers' Verifier validates
// buffer layout, not graph acyclicity, so a cyclic tree is a perfectly well-formed
// buffer. Without a bound the descent spins forever and hangs plan build.

TEST_F(TestTreeDataAdapter, SelfLoopingTreeTerminates)
{
    // Node 0 is an internal node whose left child is itself.
    GbdtModelBuilder::TreeSpec cyclic;
    cyclic.featureIndices = {0, 0};
    cyclic.thresholds = {0.5, 0.5};
    cyclic.leftChildren = {0, -1}; // <-- node 0 points at node 0
    cyclic.rightChildren = {1, -1};
    cyclic.leafValues = {0.0, 1.0};
    cyclic.defaultLeft = {1, 1};

    auto buffer = GbdtModelBuilder().setNumFeatures(1).addTree(cyclic).build();
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "");
    ASSERT_NE(adapter, nullptr);

    // Feature 0.0 <= threshold 0.5 sends the descent left, i.e. back to node 0.
    // This must raise rather than hang.
    EXPECT_THROW(adapter->score({0.0}), std::runtime_error);
}

TEST_F(TestTreeDataAdapter, MutuallyRecursiveTreeTerminates)
{
    // A two-node cycle: 0 -> 1 -> 0. Both indices are in range, so the pre-existing
    // bounds check in the loop condition does not catch it.
    GbdtModelBuilder::TreeSpec cyclic;
    cyclic.featureIndices = {0, 0};
    cyclic.thresholds = {0.5, 0.5};
    cyclic.leftChildren = {1, 0};
    cyclic.rightChildren = {1, 0};
    cyclic.leafValues = {0.0, 0.0};
    cyclic.defaultLeft = {1, 1};

    auto buffer = GbdtModelBuilder().setNumFeatures(1).addTree(cyclic).build();
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "");
    ASSERT_NE(adapter, nullptr);

    EXPECT_THROW(adapter->score({0.0}), std::runtime_error);
}

TEST_F(TestTreeDataAdapter, DeepButAcyclicTreeStillEvaluates)
{
    // The bound is the node count, so a legitimate deep chain must not trip it.
    constexpr int DEPTH = 64;
    GbdtModelBuilder::TreeSpec chain;
    for(int i = 0; i < DEPTH; ++i)
    {
        chain.featureIndices.push_back(0);
        chain.thresholds.push_back(0.5);
        chain.leftChildren.push_back(i + 1); // descend
        chain.rightChildren.push_back(i + 1);
        chain.leafValues.push_back(0.0);
        chain.defaultLeft.push_back(1);
    }
    // Terminal leaf.
    chain.featureIndices.push_back(0);
    chain.thresholds.push_back(0.0);
    chain.leftChildren.push_back(-1);
    chain.rightChildren.push_back(-1);
    chain.leafValues.push_back(7.0);
    chain.defaultLeft.push_back(1);

    auto buffer = GbdtModelBuilder().setNumFeatures(1).addTree(chain).build();
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "");
    ASSERT_NE(adapter, nullptr);

    EXPECT_DOUBLE_EQ(adapter->score({0.0}), 7.0);
}

// ========== Model Hash Field Tests ==========

TEST_F(TestTreeDataAdapter, ModelHashFieldAccepted)
{
    // TODO: Once model hash validation is implemented, this test should verify
    // that the hash is validated. For now, we just verify that the field is
    // accepted and doesn't cause load failures.
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeLeafTree(5.0))
                      .build();

    // Compute actual model hash for validation (RFC 0019 §9.2)
    const std::string modelHash = hipdnn_backend::heuristics::uhd::sha256(buffer.data(), buffer.size());

    // Load with model hash validation enabled
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH, modelHash);
    ASSERT_NE(adapter, nullptr);
    EXPECT_DOUBLE_EQ(adapter->score({0.0, 0.0}), 5.0);
}

} // namespace
