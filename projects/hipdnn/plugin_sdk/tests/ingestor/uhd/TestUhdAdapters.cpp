// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/ingestor/uhd/AdapterFactory.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/NativeScorerRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/UhdConfig.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/NativeAdapter.hpp>

#include <gtest/gtest.h>

#include <string>

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
namespace hipdnn_plugin_sdk::ingestor::uhd
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

} // namespace
} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
