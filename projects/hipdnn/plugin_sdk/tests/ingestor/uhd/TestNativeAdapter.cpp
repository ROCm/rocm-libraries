// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/uhd/NativeScorerRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/uhd/adapters/NativeAdapter.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor::uhd
{
namespace
{

/// Sums the feature row, so a test can predict the score exactly.
double sumScorer(const double* features, size_t numFeatures)
{
    double total = 0.0;
    for(size_t i = 0; i < numFeatures; ++i)
    {
        total += features[i];
    }
    return total;
}

/// Ignores the row entirely, standing in for a scorer that featurizes from
/// bindings itself (RFC 0019 §7.1).
double constantScorer(const double* /*features*/, size_t /*numFeatures*/)
{
    return 42.0;
}

constexpr const char* SUM_SYMBOL = "test_native_sum_scorer";
constexpr const char* CONSTANT_SYMBOL = "test_native_constant_scorer";

} // namespace

TEST(TestNativeScorerRegistry, ResolveReturnsRegisteredScorer)
{
    const ScopedNativeScorer scope(SUM_SYMBOL, &sumScorer);

    EXPECT_EQ(NativeScorerRegistry::resolve(SUM_SYMBOL), &sumScorer);
    EXPECT_EQ(NativeScorerRegistry::tryResolve(SUM_SYMBOL), &sumScorer);
}

TEST(TestNativeScorerRegistry, UnregisteredSymbolFailsClosed)
{
    EXPECT_EQ(NativeScorerRegistry::tryResolve("no_such_symbol"), nullptr);
    EXPECT_THROW(NativeScorerRegistry::resolve("no_such_symbol"), std::runtime_error);
}

TEST(TestNativeScorerRegistry, DuplicateRegistrationRejected)
{
    const ScopedNativeScorer scope(SUM_SYMBOL, &sumScorer);

    EXPECT_THROW(NativeScorerRegistry::registerSymbol(SUM_SYMBOL, &constantScorer),
                 std::runtime_error);
}

TEST(TestNativeScorerRegistry, NullRegistrationDegradesRatherThanCrashes)
{
    // The shared ingestor registry accepts any T, including a null function
    // pointer. NativeAdapter resolves through tryResolve, so a null entry reads
    // as unresolved and selection degrades to static_order instead of calling
    // through a null pointer.
    const ScopedNativeScorer scope("null_scorer", nullptr);

    EXPECT_EQ(NativeAdapter::resolve("null_scorer", 3, "sha256:abc"), nullptr);
}

TEST(TestNativeScorerRegistry, ScopeUnregistersOnDestruction)
{
    {
        const ScopedNativeScorer scope(SUM_SYMBOL, &sumScorer);
        EXPECT_NE(NativeScorerRegistry::tryResolve(SUM_SYMBOL), nullptr);
    }

    EXPECT_EQ(NativeScorerRegistry::tryResolve(SUM_SYMBOL), nullptr);
}

TEST(TestNativeAdapter, ResolvesAndScoresFeatureRow)
{
    const ScopedNativeScorer scope(SUM_SYMBOL, &sumScorer);

    auto adapter = NativeAdapter::resolve(SUM_SYMBOL, 3, "sha256:abc");
    ASSERT_NE(adapter, nullptr);

    EXPECT_EQ(adapter->type(), UhdAdapterType::NATIVE);
    EXPECT_EQ(adapter->expectedFeatureCount(), 3U);
    EXPECT_EQ(adapter->getFeaturesHash(), "sha256:abc");
    EXPECT_DOUBLE_EQ(adapter->score({1.0, 2.0, 3.0}), 6.0);
}

TEST(TestNativeAdapter, UnresolvedSymbolReturnsNullForStaticOrderFallback)
{
    // Selection must degrade to static_order rather than fail the request
    // (RFC 0019 §5), so an unresolved symbol is a null adapter, not a throw.
    EXPECT_EQ(NativeAdapter::resolve("never_registered", 3, "sha256:abc"), nullptr);
}

TEST(TestNativeAdapter, EmptySymbolReturnsNull)
{
    EXPECT_EQ(NativeAdapter::resolve("", 3, "sha256:abc"), nullptr);
}

TEST(TestNativeAdapter, FeatureCountMismatchThrows)
{
    const ScopedNativeScorer scope(SUM_SYMBOL, &sumScorer);

    auto adapter = NativeAdapter::resolve(SUM_SYMBOL, 3, "sha256:abc");
    ASSERT_NE(adapter, nullptr);

    EXPECT_THROW(adapter->score({1.0, 2.0}), std::invalid_argument);
}

TEST(TestNativeAdapter, ZeroFeatureCountSkipsRowValidation)
{
    // RFC 0019 §7.1 makes features_signature optional for this adapter: a
    // scorer may featurize from the bindings instead of taking the row.
    const ScopedNativeScorer scope(CONSTANT_SYMBOL, &constantScorer);

    auto adapter = NativeAdapter::resolve(CONSTANT_SYMBOL, 0, "");
    ASSERT_NE(adapter, nullptr);

    EXPECT_DOUBLE_EQ(adapter->score({}), 42.0);
    EXPECT_DOUBLE_EQ(adapter->score({1.0, 2.0, 3.0, 4.0}), 42.0);
}

} // namespace hipdnn_plugin_sdk::ingestor::uhd
