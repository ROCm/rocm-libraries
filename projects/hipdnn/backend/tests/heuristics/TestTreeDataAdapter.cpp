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

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/gbdt_model_generated.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

using hipdnn_backend::heuristics::uhd::TreeDataAdapter;
using hipdnn_backend::heuristics::uhd::UhdAdapterType;

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

/// Helper to build a GBDT model FlatBuffer in memory.
class GbdtModelBuilder
{
public:
    struct TreeSpec
    {
        std::vector<int32_t> featureIndices;
        std::vector<double> thresholds;
        std::vector<int32_t> leftChildren;
        std::vector<int32_t> rightChildren;
        std::vector<double> leafValues;
        std::vector<uint8_t> defaultLeft;
    };

    GbdtModelBuilder& setNumFeatures(int32_t n)
    {
        _numFeatures = n;
        return *this;
    }

    GbdtModelBuilder& setFeaturesHash(const std::string& hash)
    {
        _featuresHash = hash;
        return *this;
    }

    GbdtModelBuilder& setBaseScore(double score)
    {
        _baseScore = score;
        return *this;
    }

    GbdtModelBuilder& setLearningRate(double rate)
    {
        _learningRate = rate;
        return *this;
    }

    GbdtModelBuilder& addTree(const TreeSpec& tree)
    {
        _trees.push_back(tree);
        return *this;
    }

    /// Build and return the FlatBuffer data.
    std::vector<uint8_t> build()
    {
        flatbuffers::FlatBufferBuilder fbb;

        // Build trees
        std::vector<flatbuffers::Offset<fb::GbdtTree>> treeOffsets;
        for(const auto& tree : _trees)
        {
            auto treeOffset = fb::CreateGbdtTreeDirect(fbb,
                                                       &tree.featureIndices,
                                                       &tree.thresholds,
                                                       &tree.leftChildren,
                                                       &tree.rightChildren,
                                                       &tree.leafValues,
                                                       &tree.defaultLeft);
            treeOffsets.push_back(treeOffset);
        }

        auto treesVector = fbb.CreateVector(treeOffsets);
        auto hashOffset = fbb.CreateString(_featuresHash);

        fb::GbdtModelBuilder modelBuilder(fbb);
        modelBuilder.add_trees(treesVector);
        modelBuilder.add_num_features(_numFeatures);
        modelBuilder.add_features_hash(hashOffset);
        modelBuilder.add_base_score(_baseScore);
        modelBuilder.add_learning_rate(_learningRate);

        auto modelOffset = modelBuilder.Finish();
        fbb.FinishSizePrefixed(modelOffset, fb::GbdtModelIdentifier());

        // Copy to vector (skip size prefix for standard buffer)
        // Use FinishSizePrefixed but return buffer without size prefix
        fbb.Finish(modelOffset, fb::GbdtModelIdentifier());

        return std::vector<uint8_t>(fbb.GetBufferPointer(),
                                    fbb.GetBufferPointer() + fbb.GetSize());
    }

private:
    int32_t _numFeatures = 0;
    std::string _featuresHash;
    double _baseScore = 0.0;
    double _learningRate = 1.0;
    std::vector<TreeSpec> _trees;
};

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
GbdtModelBuilder::TreeSpec makeBinarySplitTree(int32_t featureIdx,
                                                double threshold,
                                                double leftLeaf,
                                                double rightLeaf)
{
    GbdtModelBuilder::TreeSpec spec;
    spec.featureIndices = {featureIdx, 0, 0};
    spec.thresholds = {threshold, 0.0, 0.0};
    spec.leftChildren = {1, -1, -1};  // node 0 -> left=1, nodes 1,2 are leaves
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

    auto adapter =
        TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "sha256:different_hash");

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

    std::vector<double> features = {1.0, 2.0};
    double score = adapter->score(features);

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

    std::vector<double> features = {3.0}; // < 5.0, should go left
    double score = adapter->score(features);

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

    std::vector<double> features = {7.0}; // >= 5.0, should go right
    double score = adapter->score(features);

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

    std::vector<double> features = {5.0}; // == 5.0, should go left with <= comparison
    double score = adapter->score(features);

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

    std::vector<double> features = {3.0, 15.0}; // feature[0] < 5.0
    double score = adapter->score(features);

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

    std::vector<double> features = {7.0, 5.0}; // feature[0] >= 5.0, feature[1] < 10.0
    double score = adapter->score(features);

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

    std::vector<double> features = {7.0, 15.0}; // feature[0] >= 5.0, feature[1] >= 10.0
    double score = adapter->score(features);

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

    std::vector<double> features = {0.0};
    double score = adapter->score(features);

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

    std::vector<double> features = {0.0};
    double score = adapter->score(features);

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

    std::vector<double> features = {0.0};
    double score = adapter->score(features);

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

    std::vector<double> features = {0.0};
    double score = adapter->score(features);

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

    std::vector<double> features = {std::numeric_limits<double>::quiet_NaN()};
    double score = adapter->score(features);

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
    std::vector<double> featuresPos = {std::numeric_limits<double>::infinity()};
    EXPECT_DOUBLE_EQ(adapter->score(featuresPos), 20.0);

    // Negative infinity should go left (< threshold)
    std::vector<double> featuresNeg = {-std::numeric_limits<double>::infinity()};
    EXPECT_DOUBLE_EQ(adapter->score(featuresNeg), 10.0);
}

// ========== Edge Cases ==========

TEST_F(TestTreeDataAdapter, ScoreWithNoTrees)
{
    auto buffer =
        GbdtModelBuilder().setNumFeatures(2).setFeaturesHash(TEST_HASH).setBaseScore(42.0).build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    std::vector<double> features = {1.0, 2.0};
    double score = adapter->score(features);

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

    std::vector<double> features;
    double score = adapter->score(features);

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
    std::vector<std::vector<double>> batch = {
        {3.0},  // <= 5.0 -> 10.0
        {7.0},  // > 5.0 -> 20.0
        {5.0},  // <= 5.0 -> 10.0 (at threshold, goes left with <=)
        {-1.0}, // <= 5.0 -> 10.0
    };

    auto scores = adapter->scoreBatch(batch);

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
    realisticTree.leftChildren = {2, 3, -1, -1, -1};  // node 0: left=2, node 1: left=3
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

} // namespace
