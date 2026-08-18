// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestUhdLoaderMultiArch.cpp
 * @brief Tests for loading multi-architecture UHDs (RFC 0019 §3.1, §8.3).
 *
 * Validates that UhdLoader can load UHDs with derived values (§6.4) and that
 * the arch-keyed structure serializes/deserializes correctly.
 */

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

/// Helper: Build a UHD with derived values (RFC 0019 §6.4)
std::vector<uint8_t> buildUhdWithDerived(const std::string& uhdId,
                                          const std::string& arch,
                                          const std::string& scoreTransform)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder(2048);

    auto id    = builder.CreateString(uhdId);
    auto name  = builder.CreateString("Test UHD for " + arch);
    auto hash  = builder.CreateString("sha256:test_hash_" + arch);
    auto obj   = builder.CreateString("max");
    auto units = builder.CreateString("tflops");
    auto xform = builder.CreateString(scoreTransform);

    // Features signature
    std::vector<flatbuffers::Offset<flatbuffers::String>> sigVec;
    sigVec.push_back(builder.CreateString("\"$kernel.tile_m\""));
    sigVec.push_back(builder.CreateString("\"$q.batch\""));
    auto featSig = builder.CreateVector(sigVec);

    // Derived values (RFC 0019 §6.4): num_tiles from problem/kernel params
    std::vector<flatbuffers::Offset<UhdDerivedEntry>> derivedVec;
    auto derivedName = builder.CreateString("num_tiles");
    auto derivedExpr = builder.CreateString(R"({"ceil_div": ["$q.dims[2]", "$kernel.tile_m"]})");
    derivedVec.push_back(CreateUhdDerivedEntry(builder, derivedName, derivedExpr));
    auto derivedOffset = builder.CreateVector(derivedVec);

    // Score metadata
    auto scoreOffset = CreateUhdScoreMetadata(builder, units, true, xform);

    // Build UHD
    auto uhdOffset = CreateUHD(builder,
                               id,               // id
                               name,             // name
                               UhdAdapter::TREE_DATA, // adapter
                               derivedOffset,    // derived (NEW in RFC 0019 §6.4)
                               featSig,          // features_signature
                               hash,             // features_hash
                               obj,              // objective
                               scoreOffset);     // score

    builder.Finish(uhdOffset, "HUHD");

    uint8_t* buf      = builder.GetBufferPointer();
    const size_t size = builder.GetSize();

    return {buf, buf + size};
}

} // namespace

// ========== Multi-Arch UHD Loading Tests ==========

TEST(TestUhdLoaderMultiArch, LoadGfx942UhdWithDerived)
{
    // Build a gfx942 UHD with derived values
    const auto buffer = buildUhdWithDerived("uhd_gfx942_v1", "gfx942", "log1p");

    auto config = UhdLoader::loadFromBuffer(buffer.data(), buffer.size());

    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->uhdId, "uhd_gfx942_v1");
    EXPECT_EQ(config->name, "Test UHD for gfx942");
    EXPECT_EQ(config->featuresHash, "sha256:test_hash_gfx942");
    EXPECT_EQ(config->objective, "max");
    EXPECT_EQ(config->adapterType, "tree_data");
    EXPECT_EQ(config->scoreTransform, "log1p");
    EXPECT_TRUE(config->scoreCalibrated);

    // Validate features signature
    ASSERT_EQ(config->featuresSignature.size(), 2);
    EXPECT_EQ(config->featuresSignature[0], "\"$kernel.tile_m\"");
    EXPECT_EQ(config->featuresSignature[1], "\"$q.batch\"");

    // Validate derived values (RFC 0019 §6.4)
    ASSERT_EQ(config->derived.size(), 1);
    EXPECT_EQ(config->derived[0].first, "num_tiles");
    EXPECT_EQ(config->derived[0].second, R"({"ceil_div": ["$q.dims[2]", "$kernel.tile_m"]})");
}

TEST(TestUhdLoaderMultiArch, LoadGfx950UhdWithDifferentTransform)
{
    // Build a gfx950 UHD with sqrt transform (different from gfx942)
    const auto buffer = buildUhdWithDerived("uhd_gfx950_v1", "gfx950", "sqrt");

    auto config = UhdLoader::loadFromBuffer(buffer.data(), buffer.size());

    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->uhdId, "uhd_gfx950_v1");
    EXPECT_EQ(config->name, "Test UHD for gfx950");
    EXPECT_EQ(config->scoreTransform, "sqrt");
    EXPECT_EQ(config->featuresHash, "sha256:test_hash_gfx950");

    // Derived values should be the same structure (different UHD, same features)
    ASSERT_EQ(config->derived.size(), 1);
    EXPECT_EQ(config->derived[0].first, "num_tiles");
}

TEST(TestUhdLoaderMultiArch, LoadDefaultUhdConservative)
{
    // Build a default fallback UHD with identity transform
    const auto buffer = buildUhdWithDerived("uhd_default_v1", "default", "identity");

    auto config = UhdLoader::loadFromBuffer(buffer.data(), buffer.size());

    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->uhdId, "uhd_default_v1");
    EXPECT_EQ(config->scoreTransform, "identity");
    EXPECT_EQ(config->featuresHash, "sha256:test_hash_default");
    EXPECT_TRUE(config->scoreCalibrated); // Still calibrated for this test

    // Same derived structure
    ASSERT_EQ(config->derived.size(), 1);
}

TEST(TestUhdLoaderMultiArch, RoundTripMultipleArchUhds)
{
    // Simulate saving 3 arch-specific UHDs to separate files
    const std::filesystem::path tempDir = std::filesystem::temp_directory_path();

    // Create gfx942, gfx950, default UHD files
    const auto gfx942Buffer = buildUhdWithDerived("uhd_gfx942", "gfx942", "log1p");
    const auto gfx950Buffer = buildUhdWithDerived("uhd_gfx950", "gfx950", "sqrt");
    const auto defaultBuffer = buildUhdWithDerived("uhd_default", "default", "identity");

    const auto gfx942Path  = tempDir / "test_uhd_gfx942.fb";
    const auto gfx950Path  = tempDir / "test_uhd_gfx950.fb";
    const auto defaultPath = tempDir / "test_uhd_default.fb";

    // Write files
    {
        std::ofstream f942(gfx942Path, std::ios::binary);
        f942.write(reinterpret_cast<const char*>(gfx942Buffer.data()),
                   static_cast<std::streamsize>(gfx942Buffer.size()));

        std::ofstream f950(gfx950Path, std::ios::binary);
        f950.write(reinterpret_cast<const char*>(gfx950Buffer.data()),
                   static_cast<std::streamsize>(gfx950Buffer.size()));

        std::ofstream fDef(defaultPath, std::ios::binary);
        fDef.write(reinterpret_cast<const char*>(defaultBuffer.data()),
                   static_cast<std::streamsize>(defaultBuffer.size()));
    }

    // Load back and validate
    auto cfg942  = UhdLoader::load(gfx942Path);
    auto cfg950  = UhdLoader::load(gfx950Path);
    auto cfgDef  = UhdLoader::load(defaultPath);

    ASSERT_TRUE(cfg942.has_value());
    EXPECT_EQ(cfg942->uhdId, "uhd_gfx942");
    EXPECT_EQ(cfg942->scoreTransform, "log1p");

    ASSERT_TRUE(cfg950.has_value());
    EXPECT_EQ(cfg950->uhdId, "uhd_gfx950");
    EXPECT_EQ(cfg950->scoreTransform, "sqrt");

    ASSERT_TRUE(cfgDef.has_value());
    EXPECT_EQ(cfgDef->uhdId, "uhd_default");
    EXPECT_EQ(cfgDef->scoreTransform, "identity");

    // All should have the same derived structure
    EXPECT_EQ(cfg942->derived.size(), 1);
    EXPECT_EQ(cfg950->derived.size(), 1);
    EXPECT_EQ(cfgDef->derived.size(), 1);

    // Cleanup
    std::filesystem::remove(gfx942Path);
    std::filesystem::remove(gfx950Path);
    std::filesystem::remove(defaultPath);
}

TEST(TestUhdLoaderMultiArch, DerivedValuesRoundTrip)
{
    // Focus test: Ensure derived values serialize/deserialize correctly
    const auto buffer = buildUhdWithDerived("derived_test", "test_arch", "log");

    auto config = UhdLoader::loadFromBuffer(buffer.data(), buffer.size());

    ASSERT_TRUE(config.has_value());

    // Validate derived structure (RFC 0019 §6.4)
    ASSERT_EQ(config->derived.size(), 1);

    const auto& [name, expr] = config->derived[0];
    EXPECT_EQ(name, "num_tiles");

    // Expression should be valid JSON
    EXPECT_EQ(expr, R"({"ceil_div": ["$q.dims[2]", "$kernel.tile_m"]})");

    // Verify it's a JsonLogic expression (has operator as key)
    EXPECT_NE(expr.find("ceil_div"), std::string::npos);
}

} // namespace hipdnn_backend::heuristics::uhd
