// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/heuristics/uhd/AdapterFactory.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/NativeScorerRegistry.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/UhdConfig.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/NativeAdapter.hpp>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

/// @file TestUhdAdapters.cpp
/// @brief makeUhdAdapter's dispatch -- RFC 0019 §7's kind names, resolved to an adapter.
///
/// Scoped to the factory. The adapters themselves are covered by the backend suite, which owns
/// TestNativeAdapter, TestCustomLibraryAdapter, TestTableAdapter and TestTreeDataAdapter and
/// drives each one directly, including a real dlopen'd scorer built from test_scorer_lib.cpp.
/// Only the dispatch was untested: nothing outside TestUhdGenArtifact's happy path called
/// makeUhdAdapter, so no case covered what it does with a kind it cannot build.
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
    EXPECT_EQ(adapter->type(), UhdAdapterType::NATIVE);
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
    EXPECT_EQ(loaded->type(), UhdAdapterType::CUSTOM_LIBRARY);

    // And a UHD declaring no digest still loads: §4.1 makes the artifact hash optional.
    config.modelHash.clear();
    EXPECT_NE(makeUhdAdapter(config), nullptr);
}

} // namespace
} // namespace hipdnn_plugin_sdk::uhd
