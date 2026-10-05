// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/heuristics/uhd/AdapterFactory.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/NativeScorerRegistry.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/UhdConfig.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/NativeAdapter.hpp>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "../../TestResourcePaths.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

/// @file TestUhdAdapters.cpp
/// @brief makeUhdAdapter's dispatch from RFC 0019 §7 kind names to adapters. Each case checks
/// the built adapter scores like the named kind, proving dispatch reached that kind.
namespace hipdnn_plugin_sdk::uhd
{
namespace
{

const std::string FEATURES_HASH = "sha256:test";

double alwaysSeven(const double* /*features*/, size_t /*count*/)
{
    return 7.0;
}

/// Registers @p symbol for the lifetime of one case, so ordering between cases cannot matter.
class ScopedNativeScorer
{
public:
    explicit ScopedNativeScorer(std::string symbol)
        : _symbol(std::move(symbol))
    {
        NativeScorerRegistry::registerSymbol(_symbol, &alwaysSeven);
    }

    ScopedNativeScorer(const ScopedNativeScorer&) = delete;
    ScopedNativeScorer& operator=(const ScopedNativeScorer&) = delete;
    ScopedNativeScorer(ScopedNativeScorer&&) = delete;
    ScopedNativeScorer& operator=(ScopedNativeScorer&&) = delete;

    ~ScopedNativeScorer()
    {
        NativeScorerRegistry::unregisterSymbol(_symbol);
    }

private:
    std::string _symbol;
};

TEST(TestIngestorUhdAdapters, TheFactoryBuildsANativeAdapterFromItsConfig)
{
    const ScopedNativeScorer scorer("test.adapters.factory");

    UhdConfig config;
    config.adapterType = "native";
    config.nativeSymbol = "test.adapters.factory";
    config.featuresHash = FEATURES_HASH;

    const auto adapter = makeUhdAdapter(config);
    ASSERT_NE(adapter, nullptr);
    EXPECT_DOUBLE_EQ(adapter->score({1.0}), 7.0) << "not the registered native scorer";
}

TEST(TestIngestorUhdAdapters, TheFactoryDeclinesAKindItCannotBuild)
{
    // An unknown kind (a newer schema) must not fall through to a default kind.
    UhdConfig config;
    config.adapterType = "onnx";
    config.featuresHash = FEATURES_HASH;

    EXPECT_EQ(makeUhdAdapter(config), nullptr);
}

TEST(TestIngestorUhdAdapters, TheFactoryDeclinesANativeKindWithNoSymbol)
{
    // `native` with an empty payload parses as a UHD and names nothing to call.
    UhdConfig config;
    config.adapterType = "native";
    config.nativeSymbol = "";

    EXPECT_EQ(makeUhdAdapter(config), nullptr);
}

/// The scorer library TestCustomLibraryAdapter dlopen's, as an absolute path.
std::string testScorerLibrary()
{
    return hipdnn_plugin_sdk::test::testScorerLibrary().string();
}

/// The SHA-256 of @p path's bytes. Computed rather than pinned because the library's bytes
/// differ per toolchain.
std::string bytesHashOf(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    EXPECT_TRUE(file) << "the test scorer library is missing: " << path;
    const auto size = file.tellg();
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    file.seekg(0);
    EXPECT_TRUE(file.read(reinterpret_cast<char*>(bytes.data()), size));
    return sha256(bytes.data(), bytes.size());
}

/// RFC 0019 §7.2: the adapter verifies a declared artifact digest before loading, so the check
/// holds for every role that constructs through the factory.
TEST(TestIngestorUhdAdapters, TheFactoryRefusesACustomLibraryWhoseDeclaredHashIsNotItsBytes)
{
    UhdConfig config;
    config.adapterType = "custom_library";
    config.modelArtifactPath = testScorerLibrary();
    config.customLibrarySymbol = "testLinearScorer";
    config.featuresSignature = {"$kernel.tile_m", "$kernel.split_k", "$q.seqlen"};
    config.featuresHash = FEATURES_HASH;

    // A well-formed digest of other bytes: a substituted library, not a malformed field.
    config.modelHash = sha256(std::string("a different library"));
    EXPECT_EQ(makeUhdAdapter(config), nullptr);

    // Control: the correct digest loads, so the refusal above is not unconditional.
    config.modelHash = bytesHashOf(config.modelArtifactPath);
    const auto loaded = makeUhdAdapter(config);
    ASSERT_NE(loaded, nullptr);
    EXPECT_DOUBLE_EQ(loaded->score({1.0, 2.0, 3.0}), 6.0) << "not testLinearScorer";

    // §4.1 makes the artifact hash optional.
    config.modelHash.clear();
    EXPECT_NE(makeUhdAdapter(config), nullptr);
}

/// A one-feature table scoring every value 1.0, written to @p path; returns its bytes' digest.
std::string writeTableModel(const std::filesystem::path& path)
{
    namespace fb = hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<double> boundaries = {5.0};
    const std::vector<uint32_t> key = {0};
    const std::vector<flatbuffers::Offset<fb::FeatureBucket>> buckets
        = {fb::CreateFeatureBucket(builder, 0, builder.CreateVector(boundaries))};
    const std::vector<flatbuffers::Offset<fb::TableEntry>> entries
        = {fb::CreateTableEntry(builder, builder.CreateVector(key), 1.0)};
    const auto model = fb::CreateTableModel(builder,
                                            1,
                                            builder.CreateString(FEATURES_HASH),
                                            builder.CreateVector(buckets),
                                            builder.CreateVector(entries));
    builder.Finish(model, fb::TableModelIdentifier());
    std::ofstream(path, std::ios::binary)
        .write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
               static_cast<std::streamsize>(builder.GetSize()));
    return sha256(builder.GetBufferPointer(), builder.GetSize());
}

TEST(TestIngestorUhdAdapters, TheFactoryRefusesATableWhoseDeclaredHashIsNotItsBytes)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(
        std::filesystem::temp_directory_path()
        / ("uhd_adapters_table_"
           + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())));
    UhdConfig config;
    config.adapterType = "table";
    config.modelArtifactPath = (dir.path() / "table.fb").string();
    config.featuresHash = FEATURES_HASH;
    const auto digest = writeTableModel(config.modelArtifactPath);

    config.modelHash = sha256(std::string("a different table"));
    EXPECT_EQ(makeUhdAdapter(config), nullptr);

    config.modelHash = digest;
    const auto loaded = makeUhdAdapter(config);
    ASSERT_NE(loaded, nullptr);
    EXPECT_DOUBLE_EQ(loaded->score({0.0}), 1.0) << "not the table's one entry";
}

} // namespace
} // namespace hipdnn_plugin_sdk::uhd
