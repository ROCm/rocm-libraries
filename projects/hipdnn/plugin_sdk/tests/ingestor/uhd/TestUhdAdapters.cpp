// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/ingestor/uhd/AdapterFactory.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/NativeScorerRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/UhdConfig.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/CustomLibraryAdapter.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/NativeAdapter.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

/// @file TestUhdAdapters.cpp
/// @brief RFC 0019 §7's adapter kinds, at the boundary where a bad descriptor arrives.
///
/// A descriptor set is drop-in data from a potentially third-party author, and §5 step 7 requires
/// a malformed one to degrade to `static_order` rather than fail the request. That makes the
/// failure paths the interesting ones: each returns nullptr, and returning a half-built adapter
/// instead would put a scorer with the wrong shape on the live path. The adapters' happy paths
/// are covered through TestUhdKernelHeuristic, which drives them the way ranking does.
namespace hipdnn_plugin_sdk::ingestor::uhd
{
namespace
{

constexpr size_t NUM_FEATURES = 3;
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

TEST(TestIngestorUhdAdapters, ANativeScorerResolvesByTheSymbolItRegistered)
{
    const ScopedNativeScorer scorer("test.adapters.resolves");

    const auto adapter
        = NativeAdapter::resolve("test.adapters.resolves", NUM_FEATURES, FEATURES_HASH);
    ASSERT_NE(adapter, nullptr);

    EXPECT_EQ(adapter->type(), UhdAdapterType::NATIVE);
    EXPECT_EQ(adapter->expectedFeatureCount(), NUM_FEATURES);
    EXPECT_EQ(adapter->getFeaturesHash(), FEATURES_HASH);
    EXPECT_DOUBLE_EQ(adapter->score({1.0, 2.0, 3.0}), 7.0);
}

TEST(TestIngestorUhdAdapters, AnUnregisteredSymbolYieldsNoAdapterRatherThanACallToNothing)
{
    // The engine registers its scorers at load; a UHD naming one that was never registered is
    // an authoring or packaging error. nullptr is what lets makeKernelHeuristic degrade to
    // declared order -- a null function pointer reached at score() time would be a crash inside
    // the caller's process instead.
    EXPECT_EQ(NativeAdapter::resolve("test.adapters.never_registered", NUM_FEATURES, FEATURES_HASH),
              nullptr);
}

TEST(TestIngestorUhdAdapters, AScorerIsNotReachableAfterItIsUnregistered)
{
    // Registration is process-wide and mutable, so "was resolvable once" is not the same claim
    // as "is resolvable now". A stale registry entry would outlive the plugin that owns the
    // function it points at.
    {
        const ScopedNativeScorer scorer("test.adapters.transient");
        ASSERT_NE(NativeAdapter::resolve("test.adapters.transient", NUM_FEATURES, FEATURES_HASH),
                  nullptr);
    }

    EXPECT_EQ(NativeAdapter::resolve("test.adapters.transient", NUM_FEATURES, FEATURES_HASH),
              nullptr);
}

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

TEST(TestIngestorUhdAdapters, ACustomLibraryDeclinesRatherThanThrowingOnABadPath)
{
    // §7.2's escape hatch dlopen's a `.so` the descriptor names. Every failure below is
    // reachable from a descriptor alone -- a typo, a missing file, a renamed export -- so each
    // must return nullptr and let selection degrade, per §5 step 7.
    EXPECT_EQ(CustomLibraryAdapter::load("", "score", NUM_FEATURES, FEATURES_HASH), nullptr)
        << "an empty path";
    EXPECT_EQ(CustomLibraryAdapter::load("/nonexistent/uhd_scorer.so",
                                         "score",
                                         NUM_FEATURES,
                                         FEATURES_HASH),
              nullptr)
        << "a path that does not exist";

    // A real, loadable library that does not export the named symbol. Distinct from the case
    // above: dlopen succeeds and dlsym is what fails, which is the path that leaks the handle
    // if the failure branch forgets to dlclose.
    EXPECT_EQ(CustomLibraryAdapter::load("libm.so.6",
                                         "hipdnn_no_such_scorer_symbol",
                                         NUM_FEATURES,
                                         FEATURES_HASH),
              nullptr)
        << "a library without the named symbol";
}

TEST(TestIngestorUhdAdapters, ACustomLibraryDeclinesAnEmptySymbolWithoutOpeningAnything)
{
    // Checked before dlopen. dlsym("") is not a defined lookup, so reaching it would ask the
    // dynamic linker a question with no answer.
    EXPECT_EQ(CustomLibraryAdapter::load("libm.so.6", "", NUM_FEATURES, FEATURES_HASH), nullptr);
}

TEST(TestIngestorUhdAdapters, AShortFeatureRowIsRefusedRatherThanPassedToTheScorer)
{
    // The native scorer takes a raw pointer and a count and reads that many doubles. Handing it
    // a row shorter than it expects is an out-of-bounds read inside code this process does not
    // own -- and for the custom_library kind, code this repository did not compile. The count is
    // checked on our side of the call for that reason.
    const ScopedNativeScorer scorer("test.adapters.short_row");
    const auto adapter
        = NativeAdapter::resolve("test.adapters.short_row", NUM_FEATURES, FEATURES_HASH);
    ASSERT_NE(adapter, nullptr);

    EXPECT_THROW(adapter->score({1.0}), std::invalid_argument);
    EXPECT_NO_THROW(adapter->score({1.0, 2.0, 3.0}));
}

} // namespace
} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
