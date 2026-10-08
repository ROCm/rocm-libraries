// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestTableAdapter.cpp
 * @brief Tests for TableAdapter (coarse bucket lookup) per RFC 0019 §7 "table".
 */

#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/TableAdapter.hpp>

#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/table_model_generated.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_plugin_sdk::uhd;
namespace fb = hipdnn_flatbuffers_sdk::data_objects;

namespace
{
constexpr const char* TEST_HASH = "sha256:test_hash_12345678";

/// Helper to build a minimal TableModel FlatBuffer for testing.
class TableModelBuilder
{
public:
    TableModelBuilder& setNumFeatures(uint32_t n)
    {
        _numFeatures = n;
        return *this;
    }

    TableModelBuilder& setFeaturesHash(const std::string& hash)
    {
        _featuresHash = hash;
        return *this;
    }

    TableModelBuilder& addBucket(uint32_t featureIdx, std::vector<double> boundaries)
    {
        _buckets.push_back({featureIdx, std::move(boundaries)});
        return *this;
    }

    TableModelBuilder&
        addEntry(std::vector<uint32_t> bucketKey, int64_t kernelId, double score = 1.0)
    {
        _entries.push_back({std::move(bucketKey), kernelId, score});
        return *this;
    }

    TableModelBuilder& setTrainingArches(std::vector<std::string> arches)
    {
        _trainingArches = std::move(arches);
        return *this;
    }

    std::vector<uint8_t> build()
    {
        flatbuffers::FlatBufferBuilder builder;

        std::vector<flatbuffers::Offset<fb::FeatureBucket>> bucketOffsets;
        for(const auto& bucket : _buckets)
        {
            auto boundaries = builder.CreateVector(bucket.boundaries);
            bucketOffsets.push_back(
                fb::CreateFeatureBucket(builder, bucket.featureIdx, boundaries));
        }

        std::vector<flatbuffers::Offset<fb::TableEntry>> entryOffsets;
        for(const auto& entry : _entries)
        {
            auto bucketKey = builder.CreateVector(entry.bucketKey);
            entryOffsets.push_back(
                fb::CreateTableEntry(builder, bucketKey, entry.kernelId, entry.score));
        }

        std::vector<flatbuffers::Offset<flatbuffers::String>> archOffsets;
        archOffsets.reserve(_trainingArches.size());
        for(const auto& arch : _trainingArches)
        {
            archOffsets.push_back(builder.CreateString(arch));
        }

        auto hashOffset = builder.CreateString(_featuresHash);
        auto bucketsVec = builder.CreateVector(bucketOffsets);
        auto entriesVec = builder.CreateVector(entryOffsets);
        auto archesVec = builder.CreateVector(archOffsets);

        auto model = fb::CreateTableModel(
            builder, _numFeatures, hashOffset, bucketsVec, entriesVec, archesVec);

        builder.Finish(model, fb::TableModelIdentifier());

        return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
    }

private:
    struct Bucket
    {
        uint32_t featureIdx;
        std::vector<double> boundaries;
    };
    struct Entry
    {
        std::vector<uint32_t> bucketKey;
        int64_t kernelId;
        double score;
    };

    uint32_t _numFeatures = 0;
    std::string _featuresHash;
    std::vector<Bucket> _buckets;
    std::vector<Entry> _entries;
    std::vector<std::string> _trainingArches;
};

} // namespace

class TestTableAdapter : public ::testing::Test
{
};

TEST_F(TestTableAdapter, LoadFromBufferBasic)
{
    auto buffer = TableModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addBucket(0, {5.0}) // Feature 0: buckets [0-5), [5+)
                      .addBucket(1, {10.0}) // Feature 1: buckets [0-10), [10+)
                      .addEntry({0, 0}, 100, 1.0) // Bucket (0,0) -> kernel 100, score 1.0
                      .addEntry({1, 1}, 200, 2.0) // Bucket (1,1) -> kernel 200, score 2.0
                      .build();

    auto adapter = TableAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->expectedFeatureCount(), 2U);
    EXPECT_EQ(adapter->getFeaturesHash(), TEST_HASH);
}

TEST_F(TestTableAdapter, ScoreExactMatch)
{
    auto buffer = TableModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addBucket(0, {5.0})
                      .addBucket(1, {10.0})
                      .addEntry({0, 0}, 100, 1.5)
                      .addEntry({1, 1}, 200, 3.5)
                      .build();

    auto adapter = TableAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    // Feature vector {3.0, 7.0} -> buckets {0, 0} -> score 1.5
    EXPECT_DOUBLE_EQ(adapter->score({3.0, 7.0}), 1.5);

    // Feature vector {6.0, 12.0} -> buckets {1, 1} -> score 3.5
    EXPECT_DOUBLE_EQ(adapter->score({6.0, 12.0}), 3.5);
}

TEST_F(TestTableAdapter, ScoreFallbackNoMatch)
{
    auto buffer = TableModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addBucket(0, {5.0})
                      .addBucket(1, {10.0})
                      .addEntry({0, 0}, 100, 1.0)
                      .build();

    auto adapter = TableAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    // Bucket (1, 1) has no entry -> declined, never a zero prediction
    EXPECT_EQ(adapter->score({6.0, 12.0}), -std::numeric_limits<double>::infinity());
}

TEST_F(TestTableAdapter, FeaturesHashMismatch)
{
    auto buffer = TableModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash("sha256:wrong_hash")
                      .addBucket(0, {5.0})
                      .addEntry({0}, 100, 1.0)
                      .build();

    auto adapter = TableAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    EXPECT_EQ(adapter, nullptr);
}

/// RFC 0019 §12: a features_hash mismatch is an error, not a warning, as in TreeDataAdapter.
TEST_F(TestTableAdapter, TheFeaturesHashCheckReportsAnError)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    auto buffer = TableModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash("sha256:wrong_hash")
                      .addBucket(0, {5.0})
                      .addEntry({0}, 100, 1.0)
                      .build();

    EXPECT_EQ(TableAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH), nullptr);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1U)
        << recorder.getRecordedLogsAsString();
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_WARN), 0U) << recorder.getRecordedLogsAsString();
}

TEST_F(TestTableAdapter, TrainingArchDetection)
{
    auto buffer = TableModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addBucket(0, {5.0})
                      .addEntry({0}, 100, 1.0)
                      .setTrainingArches({"gfx942", "gfx950"})
                      .build();

    auto adapter = TableAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    EXPECT_TRUE(adapter->isTrainedForArch("gfx942"));
    EXPECT_TRUE(adapter->isTrainedForArch("gfx950"));
    EXPECT_TRUE(adapter->isTrainedForArch("gfx942:sramecc+:xnack-"));
    EXPECT_FALSE(adapter->isTrainedForArch("gfx1100"));
    EXPECT_FALSE(adapter->isTrainedForArch("gfx9420:sramecc+:xnack-"));
}

TEST_F(TestTableAdapter, MultipleBuckets)
{
    // 3 features, 2 boundaries each -> 3 buckets per feature
    auto buffer = TableModelBuilder()
                      .setNumFeatures(3)
                      .setFeaturesHash(TEST_HASH)
                      .addBucket(0, {2.0, 4.0}) // Feature 0: [<2), [2-4), [>=4)
                      .addBucket(1, {8.0, 16.0}) // Feature 1: [<8), [8-16), [>=16)
                      .addBucket(2, {1.0, 10.0}) // Feature 2: [<1), [1-10), [>=10)
                      .addEntry({0, 0, 1}, 111, 10.0) // Low, low, mid -> kernel 111
                      .addEntry({2, 2, 2}, 222, 20.0) // High, high, high -> kernel 222
                      .build();

    auto adapter = TableAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);

    // {1.5, 5.0, 5.0} -> buckets {0, 0, 1} -> score 10.0
    EXPECT_DOUBLE_EQ(adapter->score({1.5, 5.0, 5.0}), 10.0);

    // {5.0, 20.0, 15.0} -> buckets {2, 2, 2} -> score 20.0
    EXPECT_DOUBLE_EQ(adapter->score({5.0, 20.0, 15.0}), 20.0);

    // {1.0, 1.0, 1.0} -> buckets {0, 0, 1} -> score 10.0 (boundary case)
    EXPECT_DOUBLE_EQ(adapter->score({1.0, 1.0, 1.0}), 10.0);

    // {3.0, 9.0, 2.0} -> buckets {1, 1, 1} -> no entry, declined
    EXPECT_EQ(adapter->score({3.0, 9.0, 2.0}), -std::numeric_limits<double>::infinity());
}

TEST_F(TestTableAdapter, LoadFromBufferNullBuffer)
{
    auto adapter = TableAdapter::loadFromBuffer(nullptr, 100, TEST_HASH);
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestTableAdapter, LoadFromBufferTooSmall)
{
    const std::array<uint8_t, 3> buffer = {0, 0, 0};
    auto adapter = TableAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestTableAdapter, LoadFromBufferWrongIdentifier)
{
    flatbuffers::FlatBufferBuilder builder;
    auto hashOffset = builder.CreateString(TEST_HASH);
    auto model = fb::CreateTableModel(builder, 1, hashOffset);
    builder.FinishSizePrefixed(model, "BAAD");

    std::vector<uint8_t> buffer(builder.GetBufferPointer(),
                                builder.GetBufferPointer() + builder.GetSize());

    auto adapter = TableAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    EXPECT_EQ(adapter, nullptr);
}
