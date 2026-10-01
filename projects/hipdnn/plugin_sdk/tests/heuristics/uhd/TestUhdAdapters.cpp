// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/heuristics/uhd/AdapterFactory.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/NativeScorerRegistry.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/UhdConfig.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/NativeAdapter.hpp>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

/// @file TestUhdAdapters.cpp
/// @brief makeUhdAdapter's dispatch -- RFC 0019 §7's kind names, resolved to an adapter.
///
/// Scoped to the factory. The adapters themselves are covered by TestNativeAdapter,
/// TestCustomLibraryAdapter, TestTableAdapter and TestTreeDataAdapter in this suite, which drive
/// each one directly, including a real dlopen'd scorer built from test_scorer_lib.cpp. Each case
/// here asserts the built adapter scores the way the named kind does, which is what proves the
/// dispatch reached that kind and not some other one.
///
/// That matters because the alternative to declining is not an error. A factory falling through
/// to a default kind would score against a model the descriptor never named, and §5 step 7's
/// degradation to `static_order` -- which is what nullptr triggers -- would never happen.
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
    // An unknown adapter type is a UHD written against a newer schema than this runtime. It has
    // to read as "I cannot rank with this" and not as "rank with the default kind", which would
    // score against a model the descriptor never named.
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
    return (std::filesystem::path(HIPDNN_TEST_PLUGIN_DIR)
            / hipdnn_data_sdk::utilities::getLibraryName("hipdnn_test_scorer_lib"))
        .string();
}

/// The SHA-256 of @p path's bytes -- what a conformant UHD would declare as the artifact's
/// `hash`. Computed rather than pinned: the library is rebuilt from source on every
/// configuration, so a literal digest would pin this suite to one toolchain.
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

/// RFC 0019 §7.2: a body naming an artifact may carry the digest of its bytes, "which the
/// adapter recomputes before parsing and refuses on mismatch".
///
/// The factory dropped `modelHash` on the floor for this arm, so a declared digest bound
/// nothing: the provider dlopen'ed -- and ran the initialisers of -- whatever sat at the
/// declared path. It was invisible because the L1 predictor hashed the same file itself
/// before calling the factory, so only `sort_kernel_catalog`, which has no such pre-check,
/// loaded an unverified library. Both roles construct through here, so verifying in the
/// adapter is what makes the guarantee role-independent.
TEST(TestIngestorUhdAdapters, TheFactoryRefusesACustomLibraryWhoseDeclaredHashIsNotItsBytes)
{
    UhdConfig config;
    config.adapterType = "custom_library";
    config.modelArtifactPath = testScorerLibrary();
    config.customLibrarySymbol = "test_linear_scorer";
    config.featuresSignature = {"$kernel.tile_m", "$kernel.split_k", "$q.seqlen"};
    config.featuresHash = FEATURES_HASH;

    // A digest of the right shape over the wrong bytes: the tamper this check exists for is
    // a substituted library, not a malformed field, so the value has to be a real hash.
    config.modelHash = sha256(std::string("a different library"));
    EXPECT_EQ(makeUhdAdapter(config), nullptr);

    // The control: the same config with the digest the file really has. Without it the case
    // above would also pass against a factory that refused every custom_library outright.
    config.modelHash = bytesHashOf(config.modelArtifactPath);
    const auto loaded = makeUhdAdapter(config);
    ASSERT_NE(loaded, nullptr);
    EXPECT_DOUBLE_EQ(loaded->score({1.0, 2.0, 3.0}), 6.0) << "not test_linear_scorer";

    // And a UHD declaring no digest still loads: §4.1 makes the artifact hash optional.
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
        = {fb::CreateTableEntry(builder, builder.CreateVector(key), 100, 1.0)};
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

/// R8: the table arm dropped `modelHash` exactly as the custom_library arm once did, so a
/// table artifact was scored whatever its bytes -- under an identity (the declared digest)
/// those bytes do not have. It now verifies the digest the way tree_data does.
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
