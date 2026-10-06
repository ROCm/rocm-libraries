// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestTreeDataAdapter.cpp
 * @brief Tests for TreeDataAdapter (GBDT tree walker) per RFC 0019 §8.1.
 */

#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/TreeDataAdapter.hpp>

#include <hipdnn_test_sdk/utilities/GbdtModelTestBuilder.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/gbdt_model_generated.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

using hipdnn_plugin_sdk::uhd::TreeDataAdapter;

namespace
{

/// Shared with TestUhdSelectionFlow so both suites build model artifacts the same way.
using GbdtModelBuilder = hipdnn_test_sdk::utilities::GbdtModelTestBuilder;

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

/// Node 0 splits @p featureIdx at @p threshold (<=) into leaves @p leftLeaf and @p rightLeaf.
GbdtModelBuilder::TreeSpec
    makeBinarySplitTree(int32_t featureIdx, double threshold, double leftLeaf, double rightLeaf)
{
    GbdtModelBuilder::TreeSpec spec;
    spec.featureIndices = {featureIdx, 0, 0};
    spec.thresholds = {threshold, 0.0, 0.0};
    spec.leftChildren = {1, -1, -1};
    spec.rightChildren = {2, -1, -1};
    spec.leafValues = {0.0, leftLeaf, rightLeaf};
    spec.defaultLeft = {1, 1, 1};
    return spec;
}

/// feature[0] <= 5.0 ? 1.0 : (feature[1] <= 10.0 ? 2.0 : 3.0)
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

class TestTreeDataAdapter : public ::testing::Test
{
protected:
    static constexpr const char* TEST_HASH = "sha256:test_features_hash_12345";
};

// ========== Loading Tests ==========

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

/// RFC 0019 §12: a failed contract check is an error, not a warning. The level is the
/// contract, not the wording.
TEST_F(TestTreeDataAdapter, AContractCheckThatDisablesTheModelReportsAnError)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash("sha256:model_hash")
                      .addTree(makeLeafTree(1.0))
                      .build();

    EXPECT_EQ(
        TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "sha256:different_hash"),
        nullptr);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1U)
        << "the features-hash check must report at ERROR:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_WARN), 0U) << recorder.getRecordedLogsAsString();

    recorder.clearLogs();

    EXPECT_EQ(TreeDataAdapter::loadFromBuffer(
                  buffer.data(), buffer.size(), "sha256:model_hash", "sha256:not-these-bytes"),
              nullptr);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1U)
        << "the model-digest check must report at ERROR:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_WARN), 0U) << recorder.getRecordedLogsAsString();
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
    std::vector<uint8_t> tooSmall = {0x00, 0x00, 0x00, 0x04}; // Just a size prefix

    auto adapter = TreeDataAdapter::loadFromBuffer(tooSmall.data(), tooSmall.size(), TEST_HASH);

    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestTreeDataAdapter, LoadFailsOnWrongFileIdentifier)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeLeafTree(1.0))
                      .build();

    // Bytes 4-7 hold the file identifier.
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
    // Tree: if feature[0] <= 5.0 then 10.0 else 20.0
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeBinarySplitTree(0, 5.0, 10.0, 20.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {3.0};
    const double score = adapter->score(features);

    EXPECT_DOUBLE_EQ(score, 10.0);
}

TEST_F(TestTreeDataAdapter, ScoreBinarySplitGoesRight)
{
    // Tree: if feature[0] <= 5.0 then 10.0 else 20.0
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeBinarySplitTree(0, 5.0, 10.0, 20.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {7.0};
    const double score = adapter->score(features);

    EXPECT_DOUBLE_EQ(score, 20.0);
}

TEST_F(TestTreeDataAdapter, ScoreBinarySplitAtThreshold)
{
    // LightGBM's default comparison is <=, so the threshold itself goes left.
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeBinarySplitTree(0, 5.0, 10.0, 20.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {5.0};
    const double score = adapter->score(features);

    EXPECT_DOUBLE_EQ(score, 10.0);
}

TEST_F(TestTreeDataAdapter, ScoreDeepTreePath1)
{
    // Deep tree: feature[0] <= 5.0 -> leaf(1.0)
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeDeepTree())
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {3.0, 15.0};
    const double score = adapter->score(features);

    EXPECT_DOUBLE_EQ(score, 1.0);
}

TEST_F(TestTreeDataAdapter, ScoreDeepTreePath2)
{
    // Deep tree: feature[0] > 5.0 && feature[1] <= 10.0 -> leaf(2.0)
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeDeepTree())
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {7.0, 5.0};
    const double score = adapter->score(features);

    EXPECT_DOUBLE_EQ(score, 2.0);
}

TEST_F(TestTreeDataAdapter, ScoreDeepTreePath3)
{
    // Deep tree: feature[0] > 5.0 && feature[1] > 10.0 -> leaf(3.0)
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeDeepTree())
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {7.0, 15.0};
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

    EXPECT_DOUBLE_EQ(score, 105.0);
}

TEST_F(TestTreeDataAdapter, ScoreAppliesLearningRate)
{
    // LightGBM's dump_model() exports leaf values already scaled by learning_rate, so
    // score() must not apply it again.
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .setBaseScore(0.0)
                      .setLearningRate(0.1) // Metadata only
                      .addTree(makeLeafTree(10.0))
                      .addTree(makeLeafTree(20.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {0.0};
    const double score = adapter->score(features);

    EXPECT_DOUBLE_EQ(score, 30.0);
}

TEST_F(TestTreeDataAdapter, ScoreWithBaseScoreAndLearningRate)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .setBaseScore(50.0)
                      .setLearningRate(0.5) // Metadata only
                      .addTree(makeLeafTree(10.0))
                      .addTree(makeLeafTree(20.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {0.0};
    const double score = adapter->score(features);

    EXPECT_DOUBLE_EQ(score, 80.0);
}

// ========== Missing Value Handling Tests ==========

TEST_F(TestTreeDataAdapter, ScoreWithNaNUsesDefaultLeft)
{
    // makeBinarySplitTree sets default_left on every node.
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeBinarySplitTree(0, 5.0, 10.0, 20.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    const std::vector<double> features = {std::numeric_limits<double>::quiet_NaN()};
    const double score = adapter->score(features);

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

    const std::vector<double> featuresPos = {std::numeric_limits<double>::infinity()};
    EXPECT_DOUBLE_EQ(adapter->score(featuresPos), 20.0);

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

    EXPECT_DOUBLE_EQ(score, 5.0);
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

    const std::vector<std::vector<double>> batch = {
        {3.0},
        {7.0},
        {5.0}, // At the threshold: <= goes left.
        {-1.0},
    };

    const auto scores = adapter->scoreBatch(batch);

    ASSERT_EQ(scores.size(), 4u);
    EXPECT_DOUBLE_EQ(scores[0], 10.0);
    EXPECT_DOUBLE_EQ(scores[1], 20.0);
    EXPECT_DOUBLE_EQ(scores[2], 10.0);
    EXPECT_DOUBLE_EQ(scores[3], 10.0);
}

// ========== Integration with the feature extractor ==========

TEST_F(TestTreeDataAdapter, WorksWithFeatureExtractor)
{
    // tile_m (feature 0) > 128 ? (priority (feature 1) <= 5 ? 3.0 : 2.0) : 1.0
    GbdtModelBuilder::TreeSpec realisticTree;
    realisticTree.featureIndices = {0, 1, 0, 0, 0};
    realisticTree.thresholds = {128.0, 5.0, 0.0, 0.0, 0.0};
    realisticTree.leftChildren = {2, 3, -1, -1, -1};
    realisticTree.rightChildren = {1, 4, -1, -1, -1};
    realisticTree.leafValues = {0.0, 0.0, 1.0, 3.0, 2.0};
    realisticTree.defaultLeft = {0, 1, 1, 1, 1};

    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(realisticTree)
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    EXPECT_DOUBLE_EQ(adapter->score({256.0, 1.0}), 3.0);
    EXPECT_DOUBLE_EQ(adapter->score({256.0, 10.0}), 2.0);
    EXPECT_DOUBLE_EQ(adapter->score({64.0, 1.0}), 1.0);
}

// ========== RFC 0019 §9.2: Training arches ==========

TEST_F(TestTreeDataAdapter, IsTrainedForArchReturnsTrueWhenNoArches)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeLeafTree(1.0))
                      .build();
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

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
    EXPECT_TRUE(adapter->isTrainedForArch("gfx942:sramecc+:xnack-"));
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
    EXPECT_FALSE(adapter->isTrainedForArch("gfx9420:sramecc+:xnack-"));
}

// ========== Realistic Multi-Tree Ensemble Test ==========
// Features: 0=M, 1=N, 2=K, 3=tile_m, 4=cu_count; leaves are log1p(TFLOPS).

/// M <= 512 ? 0.3 : (tile_m <= 128 ? 0.5 : (cu_count <= 60 ? 0.7 : 0.9))
GbdtModelBuilder::TreeSpec makeRealisticTree1()
{
    GbdtModelBuilder::TreeSpec spec;
    spec.featureIndices = {0, 0, 3, 0, 4, 0, 0};
    spec.thresholds = {512.0, 0.0, 128.0, 0.0, 60.0, 0.0, 0.0};
    spec.leftChildren = {1, -1, 3, -1, 5, -1, -1};
    spec.rightChildren = {2, -1, 4, -1, 6, -1, -1};
    spec.leafValues = {0.0, 0.3, 0.0, 0.5, 0.0, 0.7, 0.9};
    spec.defaultLeft = {1, 1, 1, 1, 1, 1, 1};
    return spec;
}

/// N <= 1024 ? (K <= 256 ? 0.1 : 0.15) : 0.2
GbdtModelBuilder::TreeSpec makeRealisticTree2()
{
    GbdtModelBuilder::TreeSpec spec;
    spec.featureIndices = {1, 2, 0, 0, 0};
    spec.thresholds = {1024.0, 256.0, 0.0, 0.0, 0.0};
    spec.leftChildren = {1, 3, -1, -1, -1};
    spec.rightChildren = {2, 4, -1, -1, -1};
    spec.leafValues = {0.0, 0.0, 0.2, 0.1, 0.15};
    spec.defaultLeft = {1, 1, 1, 1, 1};
    return spec;
}

/// tile_m <= 64 ? -0.05 : 0.05
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
    EXPECT_TRUE(adapter->isTrainedForArch("gfx942"));

    // Small problem: leftmost leaf of every tree.
    {
        const std::vector<double> features = {256.0, 512.0, 128.0, 32.0, 40.0};
        const double score = adapter->score(features);
        EXPECT_DOUBLE_EQ(score, 3.5 + 0.3 + 0.1 + (-0.05));
    }

    // Large problem: rightmost leaf of every tree.
    {
        const std::vector<double> features = {1024.0, 2048.0, 512.0, 256.0, 120.0};
        const double score = adapter->score(features);
        EXPECT_DOUBLE_EQ(score, 3.5 + 0.9 + 0.2 + 0.05);
    }

    // Mixed paths; tile_m == 64 sits on tree 3's threshold and goes left.
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

    const std::vector<std::vector<double>> batch = {
        {256.0, 512.0, 128.0, 32.0, 40.0}, // Small
        {1024.0, 2048.0, 512.0, 256.0, 120.0}, // Large
        {800.0, 512.0, 512.0, 64.0, 80.0}, // Medium
        {512.0, 1024.0, 256.0, 128.0, 60.0}, // Edge case at thresholds
    };

    const auto batchScores = adapter->scoreBatch(batch);
    ASSERT_EQ(batchScores.size(), batch.size());

    for(size_t i = 0; i < batch.size(); ++i)
    {
        const double singleScore = adapter->score(batch[i]);
        EXPECT_DOUBLE_EQ(batchScores[i], singleScore) << "Batch score mismatch at index " << i;
    }
}

TEST_F(TestTreeDataAdapter, RealisticRankingOrdersCorrectly)
{
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

    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    EXPECT_EQ(scored[0].second, "large_tile_high_cu");
    EXPECT_EQ(scored[1].second, "medium");
    EXPECT_EQ(scored[2].second, "small_tile_low_cu");
}

// ========== Ownership Transfer Safety Tests ==========

TEST_F(TestTreeDataAdapter, OwnershipTransferPreservesModel)
{
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .setBaseScore(10.0)
                      .addTree(makeBinarySplitTree(0, 5.0, 1.0, 2.0))
                      .build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    // The adapter must own a copy of the buffer.
    std::fill(buffer.begin(), buffer.end(), static_cast<uint8_t>(0));

    const std::vector<double> features = {3.0};
    const double score = adapter->score(features);
    EXPECT_DOUBLE_EQ(score, 11.0); // base_score(10.0) + leaf(1.0)
}

// ========== Load-time structural validation (RFC 0019 §16) ==========
//
// FlatBuffers validates the buffer layout, not the tree topology. No malformed
// branch may survive loading into the unchecked prepared-node traversal.

TEST_F(TestTreeDataAdapter, SelfLoopingTreeIsRejectedAtLoad)
{
    // Node 0 is an internal node whose left child is itself.
    GbdtModelBuilder::TreeSpec cyclic;
    cyclic.featureIndices = {0, 0};
    cyclic.thresholds = {0.5, 0.5};
    cyclic.leftChildren = {0, -1};
    cyclic.rightChildren = {1, -1};
    cyclic.leafValues = {0.0, 1.0};
    cyclic.defaultLeft = {1, 1};

    auto buffer = GbdtModelBuilder().setNumFeatures(1).addTree(cyclic).build();
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "");
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestTreeDataAdapter, MutuallyRecursiveTreeIsRejectedAtLoad)
{
    // A two-node cycle with both children referring to the same node.
    GbdtModelBuilder::TreeSpec cyclic;
    cyclic.featureIndices = {0, 0};
    cyclic.thresholds = {0.5, 0.5};
    cyclic.leftChildren = {1, 0};
    cyclic.rightChildren = {1, 0};
    cyclic.leafValues = {0.0, 0.0};
    cyclic.defaultLeft = {1, 1};

    auto buffer = GbdtModelBuilder().setNumFeatures(1).addTree(cyclic).build();
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "");
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestTreeDataAdapter, DeepButAcyclicTreeStillEvaluates)
{
    // Validation must not recurse on the host stack or impose a shallow depth cap.
    constexpr int DEPTH = 8192;
    GbdtModelBuilder::TreeSpec chain;
    for(int i = 0; i < DEPTH; ++i)
    {
        chain.featureIndices.push_back(0);
        chain.thresholds.push_back(0.5);
        chain.leftChildren.push_back(i + 1);
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
    auto buffer = GbdtModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addTree(makeLeafTree(5.0))
                      .build();

    const std::string modelHash = hipdnn_plugin_sdk::uhd::sha256(buffer.data(), buffer.size());

    auto adapter
        = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH, modelHash);
    ASSERT_NE(adapter, nullptr);
    EXPECT_DOUBLE_EQ(adapter->score({0.0, 0.0}), 5.0);
}

TEST_F(TestTreeDataAdapter, UnreachableCycleInLaterTreeIsRejected)
{
    auto tree = makeLeafTree(7.0);
    tree.featureIndices.push_back(0);
    tree.thresholds.push_back(5.0);
    tree.leftChildren.push_back(1);
    tree.rightChildren.push_back(1);
    tree.leafValues.push_back(0.0);
    const auto buffer
        = GbdtModelBuilder().setNumFeatures(1).addTree(makeLeafTree(3.0)).addTree(tree).build();

    EXPECT_EQ(TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), ""), nullptr);
}

TEST_F(TestTreeDataAdapter, InvalidChildIndicesAreRejected)
{
    auto tree = makeBinarySplitTree(0, 5.0, 10.0, 20.0);
    tree.leftChildren[0] = 3; // First index beyond the tree.
    auto buffer = GbdtModelBuilder().setNumFeatures(1).addTree(tree).build();
    EXPECT_EQ(TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), ""), nullptr);

    tree.leftChildren[0] = 1;
    tree.rightChildren[0] = -1; // An internal node cannot have an absent child.
    buffer = GbdtModelBuilder().setNumFeatures(1).addTree(tree).build();
    EXPECT_EQ(TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), ""), nullptr);

    tree.leftChildren[0] = -2; // Only -1 is the leaf marker.
    buffer = GbdtModelBuilder().setNumFeatures(1).addTree(tree).build();
    EXPECT_EQ(TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), ""), nullptr);
}

TEST_F(TestTreeDataAdapter, MissingLeafPredictionIsRejected)
{
    auto tree = makeBinarySplitTree(0, 5.0, 10.0, 20.0);
    tree.leafValues.pop_back();
    const auto buffer = GbdtModelBuilder().setNumFeatures(1).addTree(tree).build();
    EXPECT_EQ(TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), ""), nullptr);
}

TEST_F(TestTreeDataAdapter, SplitFeatureMustBelongToDeclaredSignature)
{
    for(const int32_t feature : {-1, 1})
    {
        const auto buffer = GbdtModelBuilder()
                                .setNumFeatures(1)
                                .addTree(makeBinarySplitTree(feature, 5.0, 10.0, 20.0))
                                .build();
        EXPECT_EQ(TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), ""), nullptr);
    }
}

TEST_F(TestTreeDataAdapter, NegativeFeatureCountIsRejected)
{
    const auto buffer = GbdtModelBuilder().setNumFeatures(-1).addTree(makeLeafTree(1.0)).build();
    EXPECT_EQ(TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), ""), nullptr);
}

TEST_F(TestTreeDataAdapter, EmptyTreeIsRejected)
{
    const auto buffer = GbdtModelBuilder().setNumFeatures(1).addTree({}).build();
    EXPECT_EQ(TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), ""), nullptr);
}

TEST_F(TestTreeDataAdapter, OptionalFlagsPreserveComparisonAndMissingValueBehavior)
{
    namespace fb = hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::GbdtTree>> trees;
    const std::vector<uint8_t> empty;
    const std::vector<uint8_t> strict = {0};
    const std::vector<uint8_t> inclusive = {1};
    const std::vector<uint8_t> left = {1};
    const std::vector<uint8_t> right = {0};
    // Each tree pairs one default_left flag set with one decision_lte flag set.
    const std::array<std::pair<const std::vector<uint8_t>*, const std::vector<uint8_t>*>, 4> modes
        = {{{nullptr, nullptr}, {&empty, &empty}, {&left, &strict}, {&right, &inclusive}}};
    double weight = 1.0;
    for(const auto& [defaults, decisions] : modes)
    {
        auto tree = makeBinarySplitTree(0, 5.0, weight, 2.0 * weight);
        trees.push_back(fb::CreateGbdtTreeDirect(builder,
                                                 &tree.featureIndices,
                                                 &tree.thresholds,
                                                 &tree.leftChildren,
                                                 &tree.rightChildren,
                                                 &tree.leafValues,
                                                 defaults,
                                                 decisions));
        weight *= 10.0;
    }
    const auto model = fb::CreateGbdtModel(builder, builder.CreateVector(trees), 1);
    builder.Finish(model, fb::GbdtModelIdentifier());
    auto adapter
        = TreeDataAdapter::loadFromBuffer(builder.GetBufferPointer(), builder.GetSize(), "");
    ASSERT_NE(adapter, nullptr);

    // Absent and empty comparisons default to <=; explicit false remains <.
    EXPECT_DOUBLE_EQ(adapter->score({5.0}), 1211.0);
    EXPECT_DOUBLE_EQ(adapter->score({std::nextafter(5.0, 0.0)}), 1111.0);
    EXPECT_DOUBLE_EQ(adapter->score({std::nextafter(5.0, 10.0)}), 2222.0);
    // Missing default_left is false, unlike missing decision_lte.
    EXPECT_DOUBLE_EQ(adapter->score({std::numeric_limits<double>::quiet_NaN()}), 2122.0);
    EXPECT_DOUBLE_EQ(adapter->score({}), 2122.0);
}

TEST_F(TestTreeDataAdapter, AddsBaseScoreAfterSummingTreesInModelOrder)
{
    const auto buffer = GbdtModelBuilder()
                            .setBaseScore(1.0)
                            .addTree(makeLeafTree(1e16))
                            .addTree(makeLeafTree(-1e16))
                            .addTree(makeLeafTree(0.25))
                            .build();
    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "");
    ASSERT_NE(adapter, nullptr);
    EXPECT_DOUBLE_EQ(adapter->score({}), 1.25);
}

} // namespace

TEST_F(TestTreeDataAdapter, ATreeWhoseParallelArraysDisagreeIsRejected)
{
    // FlatBuffers' Verifier checks each vector alone, not that the node-parallel arrays share
    // a length, so the adapter must (RFC 0019 §16).
    GbdtModelBuilder::TreeSpec ragged;
    ragged.featureIndices = {0, 0, 0};
    ragged.thresholds = {1.0, 2.0, 3.0};
    ragged.leftChildren = {1, -1, -1};
    ragged.rightChildren = {2}; // shorter than left_children: nodes 1 and 2 read OOB
    ragged.leafValues = {0.0, 10.0, 20.0};
    ragged.defaultLeft = {1, 1, 1};

    GbdtModelBuilder builder;
    const auto buffer = builder.setNumFeatures(1).addTree(ragged).build();

    auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "");
    EXPECT_EQ(adapter, nullptr)
        << "a model whose node-parallel arrays have different lengths must be refused "
           "at load, not walked";
}

TEST_F(TestTreeDataAdapter, RepeatedTreeOffsetsCannotClaimMoreNodesThanTheBufferHolds)
{
    // Every `trees` entry may point at one table, so the claimed node count must be checked
    // against the buffer before the adapter reserves storage for it.
    namespace fb = hipdnn_flatbuffers_sdk::data_objects;
    const auto buildRepeated = [](size_t repeats) {
        flatbuffers::FlatBufferBuilder builder;
        const auto spec = makeBinarySplitTree(0, 5.0, 1.0, 2.0);
        const auto tree = fb::CreateGbdtTreeDirect(builder,
                                                   &spec.featureIndices,
                                                   &spec.thresholds,
                                                   &spec.leftChildren,
                                                   &spec.rightChildren,
                                                   &spec.leafValues,
                                                   &spec.defaultLeft);
        const std::vector<flatbuffers::Offset<fb::GbdtTree>> trees(repeats, tree);
        builder.Finish(fb::CreateGbdtModel(builder, builder.CreateVector(trees), 1),
                       fb::GbdtModelIdentifier());
        return std::vector<uint8_t>(builder.GetBufferPointer(),
                                    builder.GetBufferPointer() + builder.GetSize());
    };

    // Sharing a table is legal while the total stays within what the buffer could back.
    const auto shared = buildRepeated(4);
    auto sharedAdapter = TreeDataAdapter::loadFromBuffer(shared.data(), shared.size(), "");
    ASSERT_NE(sharedAdapter, nullptr);
    EXPECT_DOUBLE_EQ(sharedAdapter->score({0.0}), 4.0);

    constexpr size_t REPEATS = 1000;
    const auto inflated = buildRepeated(REPEATS);
    ASSERT_GT(3 * REPEATS, inflated.size() / sizeof(int32_t));
    EXPECT_EQ(TreeDataAdapter::loadFromBuffer(inflated.data(), inflated.size(), ""), nullptr);
}

// ---- Grouped models: solver first, then kernel, inside one artifact -------------------

namespace
{
/// Slot 0 carries the group; slot 1 is what layer 2 splits on.
GbdtModelBuilder groupedBuilder()
{
    GbdtModelBuilder builder;
    builder.setNumFeatures(2).setFeaturesHash("sha256:grouped").setGroupByFeatureIndex(0);
    // Layer 1: a stump on slot 0, so group 1.0 outranks group 0.0.
    GbdtModelBuilder::TreeSpec layerOne;
    layerOne.featureIndices = {0, 0, 0};
    layerOne.thresholds = {0.5, 0.0, 0.0};
    layerOne.leftChildren = {1, -1, -1};
    layerOne.rightChildren = {2, -1, -1};
    layerOne.leafValues = {0.0, 1.0, 9.0};
    layerOne.defaultLeft = {1, 1, 1};
    builder.addTree(layerOne);
    // Layer 2 per group: a constant, so a score identifies which ensemble ran.
    builder.addGroup(0.0, {makeLeafTree(100.0)});
    builder.addGroup(1.0, {makeLeafTree(200.0)});
    return builder;
}
} // namespace

TEST(TestTreeDataAdapterGrouped, RowsOutsideTheChosenGroupAreUnusable)
{
    const auto buffer = groupedBuilder().build();
    const auto adapter
        = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "sha256:grouped");
    ASSERT_NE(adapter, nullptr);

    // Group 1.0 wins layer 1 (leaf 9.0 against 1.0), so only its rows keep a score.
    const auto scores = adapter->scoreBatch({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}});
    ASSERT_EQ(scores.size(), 3u);
    EXPECT_DOUBLE_EQ(scores[1], 200.0);
    EXPECT_EQ(scores[0], -std::numeric_limits<double>::infinity());
    EXPECT_EQ(scores[2], -std::numeric_limits<double>::infinity());
}

TEST(TestTreeDataAdapterGrouped, AMinObjectiveChoosesTheGroupWithTheLowestLayerOneScore)
{
    // Layer 1 of a `min` model predicts a cost, so group 0.0 (leaf 1.0 against 9.0) wins.
    const auto buffer = groupedBuilder().build();
    const auto adapter = TreeDataAdapter::loadFromBuffer(
        buffer.data(), buffer.size(), "sha256:grouped", /*expectedModelHash=*/"", "min");
    ASSERT_NE(adapter, nullptr);

    const auto scores = adapter->scoreBatch({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}});
    ASSERT_EQ(scores.size(), 3u);
    // Raw, unoriented layer-2 scores: orientation is the ranker's job, not the adapter's.
    EXPECT_DOUBLE_EQ(scores[0], 100.0);
    EXPECT_DOUBLE_EQ(scores[2], 100.0);
    EXPECT_EQ(scores[1], -std::numeric_limits<double>::infinity());
}

TEST(TestTreeDataAdapterGrouped, TheSurvivingRowsAreScoredByTheirOwnGroupsTrees)
{
    // Only group 0.0 is present, so layer 1 has one choice and layer 2 must be the group's
    // own ensemble rather than layer 1's score.
    const auto buffer = groupedBuilder().build();
    const auto adapter
        = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "sha256:grouped");
    ASSERT_NE(adapter, nullptr);

    const auto scores = adapter->scoreBatch({{0.0, 0.0}, {0.0, 1.0}});
    ASSERT_EQ(scores.size(), 2u);
    EXPECT_DOUBLE_EQ(scores[0], 100.0);
    EXPECT_DOUBLE_EQ(scores[1], 100.0);
}

TEST(TestTreeDataAdapterGrouped, EqualGroupStandingsResolveIndependentlyOfRowOrder)
{
    // rank()'s tie-break cannot repair this later: the losing group is already discarded.
    GbdtModelBuilder builder;
    builder.setNumFeatures(2)
        .setFeaturesHash("sha256:grouped")
        .setGroupByFeatureIndex(0)
        .addTree(makeLeafTree(5.0));
    builder.addGroup(0.0, {makeLeafTree(100.0)});
    builder.addGroup(1.0, {makeLeafTree(200.0)});
    const auto buffer = builder.build();
    const auto adapter
        = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "sha256:grouped");
    ASSERT_NE(adapter, nullptr);

    const auto forward = adapter->scoreBatch({{1.0, 0.0}, {0.0, 0.0}});
    const auto reversed = adapter->scoreBatch({{0.0, 0.0}, {1.0, 0.0}});
    ASSERT_EQ(forward.size(), 2u);
    ASSERT_EQ(reversed.size(), 2u);
    // The smaller group value wins the tie, whichever row came first.
    EXPECT_DOUBLE_EQ(forward[1], 100.0);
    EXPECT_EQ(forward[0], -std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(reversed[0], 100.0);
    EXPECT_EQ(reversed[1], -std::numeric_limits<double>::infinity());
}

TEST(TestTreeDataAdapterGrouped, ALayerOneScoreOutsideItsTargetCannotChooseTheGroup)
{
    // RFC 0019 §8.3 refuses a negative time for a candidate; it must not decide a group either.
    GbdtModelBuilder builder;
    builder.setNumFeatures(2).setFeaturesHash("sha256:grouped").setGroupByFeatureIndex(0);
    GbdtModelBuilder::TreeSpec layerOne;
    layerOne.featureIndices = {0, 0, 0};
    layerOne.thresholds = {0.5, 0.0, 0.0};
    layerOne.leftChildren = {1, -1, -1};
    layerOne.rightChildren = {2, -1, -1};
    layerOne.leafValues = {0.0, -1.0, 9.0}; // group 0.0: -1 ms; group 1.0: 9 ms
    layerOne.defaultLeft = {1, 1, 1};
    builder.addTree(layerOne);
    builder.addGroup(0.0, {makeLeafTree(100.0)});
    builder.addGroup(1.0, {makeLeafTree(200.0)});
    const auto buffer = builder.build();
    const auto adapter = TreeDataAdapter::loadFromBuffer(buffer.data(),
                                                         buffer.size(),
                                                         "sha256:grouped",
                                                         /*expectedModelHash=*/"",
                                                         "min",
                                                         "identity",
                                                         "time");
    ASSERT_NE(adapter, nullptr);

    const auto scores = adapter->scoreBatch({{0.0, 0.0}, {1.0, 0.0}});
    ASSERT_EQ(scores.size(), 2u);
    EXPECT_EQ(scores[0], -std::numeric_limits<double>::infinity())
        << "a negative time chose the group";
    EXPECT_DOUBLE_EQ(scores[1], 200.0);
}

TEST(TestTreeDataAdapterGrouped, ScoreStillAnswersWithLayerOne)
{
    // One row cannot express a group decision, so score() answers with layer 1.
    const auto buffer = groupedBuilder().build();
    const auto adapter
        = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "sha256:grouped");
    ASSERT_NE(adapter, nullptr);

    EXPECT_DOUBLE_EQ(adapter->score({1.0, 0.0}), 9.0);
    EXPECT_DOUBLE_EQ(adapter->score({0.0, 0.0}), 1.0);
}

TEST(TestTreeDataAdapterGrouped, AnUngroupedModelBatchesExactlyAsItScores)
{
    GbdtModelBuilder builder;
    builder.setNumFeatures(2).setFeaturesHash("sha256:plain").addTree(makeLeafTree(3.5));
    const auto buffer = builder.build();
    const auto adapter
        = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "sha256:plain");
    ASSERT_NE(adapter, nullptr);

    const std::vector<std::vector<double>> rows = {{0.0, 0.0}, {1.0, 2.0}};
    const auto batched = adapter->scoreBatch(rows);
    ASSERT_EQ(batched.size(), rows.size());
    for(size_t i = 0; i < rows.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(batched[i], adapter->score(rows[i]));
    }
}

TEST(TestTreeDataAdapterGrouped, TheGroupingSlotIsReadable)
{
    // Rankers report the slot that actually decided, not one derived from descriptor text.
    const auto buffer = groupedBuilder().build();
    const auto adapter
        = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "sha256:grouped");
    ASSERT_NE(adapter, nullptr);

    EXPECT_EQ(adapter->groupFeatureIndex(), 0);
}

TEST(TestTreeDataAdapterGrouped, AnUngroupedModelReportsNoGroupingSlot)
{
    // -1, so a single-layer model cannot be read as grouping on slot 0.
    GbdtModelBuilder builder;
    builder.setNumFeatures(2).setFeaturesHash("sha256:flat").addTree(makeLeafTree(1.0));
    const auto buffer = builder.build();
    const auto adapter
        = TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), "sha256:flat");
    ASSERT_NE(adapter, nullptr);

    EXPECT_EQ(adapter->groupFeatureIndex(), -1);
}
