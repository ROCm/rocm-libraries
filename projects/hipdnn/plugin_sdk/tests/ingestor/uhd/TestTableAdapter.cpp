// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestTableAdapter.cpp
 * @brief Tests for TableAdapter (coarse bucket lookup) per RFC 0019 §7 "table".
 *
 * Tests cover:
 * - Model loading from buffer
 * - Features hash validation
 * - Bucket quantization and lookup
 * - Training arch detection
 * - Edge cases (no match, invalid buffer)
 */

#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/TableAdapter.hpp>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/table_model_generated.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_plugin_sdk::ingestor::uhd;
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

    TableModelBuilder& addEntry(std::vector<uint32_t> bucketKey, int64_t kernelId, double score = 1.0)
    {
        _entries.push_back({std::move(bucketKey), kernelId, score});
        return *this;
    }

    TableModelBuilder& setTrainingArches(std::vector<std::string> arches)
    {
        _trainingArches = std::move(arches);
        return *this;
    }

    TableModelBuilder& setModelVersion(const std::string& version)
    {
        _modelVersion = version;
        return *this;
    }

    std::vector<uint8_t> build()
    {
        flatbuffers::FlatBufferBuilder builder;

        // Build buckets
        std::vector<flatbuffers::Offset<fb::FeatureBucket>> bucketOffsets;
        for(const auto& bucket : _buckets)
        {
            auto boundaries = builder.CreateVector(bucket.boundaries);
            bucketOffsets.push_back(
                fb::CreateFeatureBucket(builder, bucket.featureIdx, boundaries));
        }

        // Build entries
        std::vector<flatbuffers::Offset<fb::TableEntry>> entryOffsets;
        for(const auto& entry : _entries)
        {
            auto bucketKey = builder.CreateVector(entry.bucketKey);
            entryOffsets.push_back(
                fb::CreateTableEntry(builder, bucketKey, entry.kernelId, entry.score));
        }

        // Build training arches
        std::vector<flatbuffers::Offset<flatbuffers::String>> archOffsets;
        archOffsets.reserve(_trainingArches.size());
        for(const auto& arch : _trainingArches)
        {
            archOffsets.push_back(builder.CreateString(arch));
        }

        auto hashOffset = builder.CreateString(_featuresHash);
        auto versionOffset = builder.CreateString(_modelVersion);
        auto bucketsVec = builder.CreateVector(bucketOffsets);
        auto entriesVec = builder.CreateVector(entryOffsets);
        auto archesVec = builder.CreateVector(archOffsets);

        auto model = fb::CreateTableModel(builder,
                                         _numFeatures,
                                         hashOffset,
                                         bucketsVec,
                                         entriesVec,
                                         archesVec,
                                         versionOffset);

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
    std::string _modelVersion;
};

} // namespace

class TestTableAdapter : public ::testing::Test
{
};

TEST_F(TestTableAdapter, LoadFromBufferBasic)
{
    // Build a simple table: 2 features, 1 bucket each, 2 entries
    auto buffer = TableModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash(TEST_HASH)
                      .addBucket(0, {5.0})       // Feature 0: buckets [0-5), [5+)
                      .addBucket(1, {10.0})      // Feature 1: buckets [0-10), [10+)
                      .addEntry({0, 0}, 100, 1.0) // Bucket (0,0) -> kernel 100, score 1.0
                      .addEntry({1, 1}, 200, 2.0) // Bucket (1,1) -> kernel 200, score 2.0
                      .build();

    auto adapter = TableAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->type(), UhdAdapterType::TABLE);
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

    // Bucket (1, 1) has no entry -> fallback score 0.0
    EXPECT_DOUBLE_EQ(adapter->score({6.0, 12.0}), 0.0);
}

TEST_F(TestTableAdapter, FeaturesHashMismatch)
{
    auto buffer = TableModelBuilder()
                      .setNumFeatures(2)
                      .setFeaturesHash("sha256:wrong_hash")
                      .addBucket(0, {5.0})
                      .addEntry({0}, 100, 1.0)
                      .build();

    // Load should fail due to hash mismatch
    auto adapter = TableAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    EXPECT_EQ(adapter, nullptr);
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
    EXPECT_FALSE(adapter->isTrainedForArch("gfx1100"));

    auto arches = adapter->getTrainingArches();
    EXPECT_EQ(arches.size(), 2U);
    EXPECT_EQ(arches[0], "gfx942");
    EXPECT_EQ(arches[1], "gfx950");
}

TEST_F(TestTableAdapter, ModelVersion)
{
    const std::string version = "v1.0.0-test";
    auto buffer = TableModelBuilder()
                      .setNumFeatures(1)
                      .setFeaturesHash(TEST_HASH)
                      .addBucket(0, {5.0})
                      .addEntry({0}, 100, 1.0)
                      .setModelVersion(version)
                      .build();

    auto adapter = TableAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->getModelVersion(), version);
}

TEST_F(TestTableAdapter, MultipleBuckets)
{
    // 3 features, 2 boundaries each -> 3 buckets per feature
    auto buffer = TableModelBuilder()
                      .setNumFeatures(3)
                      .setFeaturesHash(TEST_HASH)
                      .addBucket(0, {2.0, 4.0})  // Feature 0: [<2), [2-4), [>=4)
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

    // {3.0, 9.0, 2.0} -> buckets {1, 1, 1} -> no entry, fallback 0.0
    EXPECT_DOUBLE_EQ(adapter->score({3.0, 9.0, 2.0}), 0.0);
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
    // Build a buffer with wrong file identifier
    flatbuffers::FlatBufferBuilder builder;
    auto hashOffset = builder.CreateString(TEST_HASH);
    auto model = fb::CreateTableModel(builder, 1, hashOffset);
    builder.FinishSizePrefixed(model, "BAAD"); // Wrong identifier

    std::vector<uint8_t> buffer(builder.GetBufferPointer(),
                                builder.GetBufferPointer() + builder.GetSize());

    auto adapter = TableAdapter::loadFromBuffer(buffer.data(), buffer.size(), TEST_HASH);
    EXPECT_EQ(adapter, nullptr);
}
