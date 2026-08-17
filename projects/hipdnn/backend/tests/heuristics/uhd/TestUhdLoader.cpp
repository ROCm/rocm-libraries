// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <heuristics/uhd/UhdLoader.hpp>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/uhd_generated.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace hipdnn_backend::heuristics::uhd
{

namespace
{

/// Helper to build a minimal valid UHD FlatBuffer in memory.
std::vector<uint8_t> buildMinimalUhd(const std::string& uhdId,
                                     const std::string& featuresHash,
                                     const std::string& objective)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder(1024);

    auto id     = builder.CreateString(uhdId);
    auto hash   = builder.CreateString(featuresHash);
    auto obj    = builder.CreateString(objective);
    auto units  = builder.CreateString("tflops");
    auto xform  = builder.CreateString("identity");

    // Build score metadata
    auto scoreOffset = CreateUhdScoreMetadata(builder, units, false, xform);

    // Build UHD (RFC 0019 §6.4: derived field added between adapter and features_signature)
    auto uhdOffset = CreateUHD(builder,
                               id,          // id
                               0,           // name
                               UhdAdapter::STATIC_ORDER, // adapter
                               0,           // derived (empty for static_order)
                               0,           // features_signature (empty)
                               hash,        // features_hash
                               obj,         // objective
                               scoreOffset);// score

    builder.Finish(uhdOffset, "HUHD");

    uint8_t* buf      = builder.GetBufferPointer();
    const size_t size = builder.GetSize();

    return {buf, buf + size};
}

/// Helper to build a tree_data UHD with model artifact path.
std::vector<uint8_t> buildTreeDataUhd(const std::string& uhdId,
                                      const std::string& featuresHash,
                                      const std::string& modelPath)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder(1024);

    auto id        = builder.CreateString(uhdId);
    auto hash      = builder.CreateString(featuresHash);
    auto obj       = builder.CreateString("max");
    auto units     = builder.CreateString("tflops");
    auto xform     = builder.CreateString("log1p");
    auto modelArt  = builder.CreateString(modelPath);

    std::vector<flatbuffers::Offset<flatbuffers::String>> sigVec;
    sigVec.push_back(builder.CreateString("$q.M"));
    sigVec.push_back(builder.CreateString("$q.N"));
    auto featSig   = builder.CreateVector(sigVec);

    auto scoreOffset = CreateUhdScoreMetadata(builder, units, false, xform);

    // Build UHD with features_signature (tree_data adapter)
    auto uhdOffset = CreateUHD(builder,
                               id,               // id
                               0,                // name
                               UhdAdapter::TREE_DATA, // adapter
                               0,                // derived (empty for this test)
                               featSig,          // features_signature
                               hash,             // features_hash
                               obj,              // objective
                               scoreOffset,      // score
                               modelArt);        // model_artifact_path

    builder.Finish(uhdOffset, "HUHD");

    uint8_t* buf      = builder.GetBufferPointer();
    const size_t size = builder.GetSize();

    return {buf, buf + size};
}

} // namespace

// Unit test suite for UhdLoader
TEST(TestUhdLoader, LoadFromBufferMinimal)
{
    const auto buffer = buildMinimalUhd("test-uhd-id", "sha256:abc123", "max");

    auto config = UhdLoader::loadFromBuffer(buffer.data(), buffer.size());

    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->uhdId, "test-uhd-id");
    EXPECT_EQ(config->featuresHash, "sha256:abc123");
    EXPECT_EQ(config->objective, "max");
    EXPECT_EQ(config->adapterType, "static_order");
}

TEST(TestUhdLoader, LoadFromBufferTreeData)
{
    const auto buffer = buildTreeDataUhd("tree-uhd-id", "sha256:xyz789", "model.bin");

    auto config = UhdLoader::loadFromBuffer(buffer.data(), buffer.size(), "/base/path");

    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->uhdId, "tree-uhd-id");
    EXPECT_EQ(config->featuresHash, "sha256:xyz789");
    EXPECT_EQ(config->objective, "max");
    EXPECT_EQ(config->adapterType, "tree_data");
    EXPECT_EQ(config->scoreUnits, "tflops");
    EXPECT_EQ(config->scoreTransform, "log1p");
    EXPECT_FALSE(config->scoreCalibrated);

    // Relative path should be resolved against basePath
    EXPECT_EQ(config->modelArtifactPath, "/base/path/model.bin");

    // Features signature
    ASSERT_EQ(config->featuresSignature.size(), 2);
    EXPECT_EQ(config->featuresSignature[0], "$q.M");
    EXPECT_EQ(config->featuresSignature[1], "$q.N");
}

TEST(TestUhdLoader, LoadFromBufferInvalidObjective)
{
    const auto buffer = buildMinimalUhd("bad-obj", "sha256:test", "maximize");

    auto config = UhdLoader::loadFromBuffer(buffer.data(), buffer.size());

    EXPECT_FALSE(config.has_value());
}

TEST(TestUhdLoader, LoadFromBufferNullBuffer)
{
    auto config = UhdLoader::loadFromBuffer(nullptr, 0);

    EXPECT_FALSE(config.has_value());
}

TEST(TestUhdLoader, LoadFromBufferCorruptBuffer)
{
    const std::vector<uint8_t> corrupt = {0xDE, 0xAD, 0xBE, 0xEF};

    auto config = UhdLoader::loadFromBuffer(corrupt.data(), corrupt.size());

    EXPECT_FALSE(config.has_value());
}

TEST(TestUhdLoader, LoadFromFileNotFound)
{
    auto config = UhdLoader::load("/nonexistent/path/uhd.fb");

    EXPECT_FALSE(config.has_value());
}

TEST(TestUhdLoader, LoadFromFileRoundTrip)
{
    const auto buffer = buildTreeDataUhd("file-test", "sha256:file", "model.bin");

    // Write to temp file
    const std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "test_uhd.fb";
    {
        std::ofstream file(tempPath, std::ios::binary);
        ASSERT_TRUE(file.is_open());
        file.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    }

    // Load from file
    auto config = UhdLoader::load(tempPath);

    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->uhdId, "file-test");
    EXPECT_EQ(config->featuresHash, "sha256:file");
    EXPECT_EQ(config->adapterType, "tree_data");

    // Cleanup
    std::filesystem::remove(tempPath);
}

} // namespace hipdnn_backend::heuristics::uhd
