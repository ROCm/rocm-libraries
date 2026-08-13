// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestKernelIngestorStateManager.cpp
 * @brief Unit tests for KernelIngestorStateManager: matching, pruning, caching, ranking,
 *        knob discovery, metadata completion, and construction-time validation.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

/// Binds "test.bound_token" to a value that disagrees with acceptGraph's.
inline bool bindConflictingTokenValue(const MatchContext& /*context*/, BoundTokens& bound)
{
    bound["test.bound_token"] = BOUND_TOKEN_VALUE + 1;
    return true;
}

/// Binds "test.bound_token" to acceptGraph's value, via a different matcher id.
inline bool bindAgreeingTokenValue(const MatchContext& /*context*/, BoundTokens& bound)
{
    bound["test.bound_token"] = BOUND_TOKEN_VALUE;
    return true;
}

// ---------------------------------------------------------------------------
// Matching and pruning
// ---------------------------------------------------------------------------

TEST(TestKernelIngestorStateManager, KernelLevelMatcherPrunesTheCatalog)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto definitions = manager->unsortedDefinitions(context);

    // Three kernels in the pack; the HALF one does not survive.
    ASSERT_EQ(definitions.size(), 2U);
    for(const auto& definition : definitions)
    {
        EXPECT_EQ(definition.getStringMetadata(DTYPE), "FLOAT");
    }
}

TEST(TestKernelIngestorStateManager, GraphLevelMatcherFailurePrunesTheWholePack)
{
    const ScopedSymbols symbols("test.graph", rejectGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(2));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    EXPECT_TRUE(manager->unsortedDefinitions(context).empty());
    // A rejected graph costs one matcher call, not one per kernel.
    EXPECT_EQ(counters().graphCalls, 1);
    EXPECT_EQ(counters().kernelCalls, 0);
}

TEST(TestKernelIngestorStateManager, MatchesOncePerGraphAndDevice)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(3));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    manager->unsortedDefinitions(context);
    manager->unsortedDefinitions(context);
    manager->unsortedDefinitions(context);

    EXPECT_EQ(counters().graphCalls, 1);
    // One call per kernel in the pack, on the single uncached pass.
    EXPECT_EQ(counters().kernelCalls, 3);
}

TEST(TestKernelIngestorStateManager, EvaluatesASharedGraphMatcherOncePerGraphNotOncePerPack)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    // Matchers are shared by id, so the graph-scoped check runs once, not once per pack.
    // The first pack also fails a second, unshared matcher, so only the second pack
    // survives, isolating binding reuse (read from the memo) from re-evaluation.
    constexpr const char* UNSHARED_FAIL_SYMBOL = "test.cross_pack_binding_reuse.fail";
    GraphMatcherRegistry::registerSymbol(UNSHARED_FAIL_SYMBOL, rejectGraph);
    const auto unsharedFailMatcherId = testId(0x84);

    const KernelDescriptorPack first = makePack({GRAPH_MATCHER_ID, unsharedFailMatcherId});
    KernelDescriptorPack second = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    second.id = testId(0x80);
    // Distinct metadata tuple: uniqueness is engine-wide, not per pack.
    second.kernels = {makeKernel(testId(0x81), "second_pack_kernel", 512, "FLOAT")};

    const StateManager manager(
        makeSchema(),
        {{GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, "test.graph"},
         {unsharedFailMatcherId, "always fails", MatchScope::GRAPH, UNSHARED_FAIL_SYMBOL},
         {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, "test.kernel"}},
        makeTestDispatches(),
        {first, second},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const TestGraph graph(makeGraphId(20));
    const auto properties = testDeviceProperties();
    const auto catalog = manager.unsortedCatalog(MatchContext{graph, 0, properties});

    // Two calls: the shared matcher once, plus the first pack's unshared failing matcher.
    EXPECT_EQ(counters().graphCalls, 2);
    // Only the second pack's kernel survives; the first pack was pruned before its
    // kernels ran.
    ASSERT_EQ(catalog.entries.size(), 1U);
    EXPECT_EQ(catalog.entries.front().packId, second.id);
    // The binding reaches catalog.bound via the second pack's memo read, not the
    // pruned first pack that actually ran the matcher.
    EXPECT_EQ(catalog.bound.count("test.bound_token"), 1U);

    GraphMatcherRegistry::unregisterSymbol(UNSHARED_FAIL_SYMBOL);
}

TEST(TestKernelIngestorStateManager, ASharedGraphMatcherFailurePrunesEveryPackListingIt)
{
    const ScopedSymbols symbols("test.graph", rejectGraph, "test.kernel", countingFloatKernels);

    auto first = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    auto second = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    second.id = testId(0x82);
    second.kernels = {makeKernel(testId(0x83), "second_pack_kernel", 512, "FLOAT")};

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {first, second},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const TestGraph graph(makeGraphId(21));
    const auto properties = testDeviceProperties();

    EXPECT_TRUE(manager.unsortedDefinitions(MatchContext{graph, 0, properties}).empty());
    // One failing evaluation disqualifies both packs, and no per-kernel work runs at all.
    EXPECT_EQ(counters().graphCalls, 1);
    EXPECT_EQ(counters().kernelCalls, 0);
}

TEST(TestKernelIngestorStateManager, PrunedPackBindingsAreNotVisibleToSurvivingPackKernels)
{
    // Two packs share an engine. The pruned pack's own matcher binds a token before a
    // second matcher in the same pack fails and prunes it; the surviving pack lists
    // neither matcher, so it must not see that binding.
    constexpr const char* PASS_NO_BIND_SYMBOL = "test.pruned_bound_isolation.pass_no_bind";
    constexpr const char* LEAK_THEN_PRUNE_SYMBOL = "test.pruned_bound_isolation.leak";
    constexpr const char* ALWAYS_FAILS_SYMBOL = "test.pruned_bound_isolation.fail";

    // acceptAnyGraph passes and binds nothing; acceptGraph passes and binds
    // "test.bound_token"; rejectGraph always fails. Registered before the manager,
    // which resolves every matcher symbol at construction.
    const ScopedBlockSizeScore scorer;
    GraphMatcherRegistry::registerSymbol(PASS_NO_BIND_SYMBOL, &acceptAnyGraph);
    GraphMatcherRegistry::registerSymbol(LEAK_THEN_PRUNE_SYMBOL, &acceptGraph);
    GraphMatcherRegistry::registerSymbol(ALWAYS_FAILS_SYMBOL, &rejectGraph);
    counters().reset();

    const auto passNoBindMatcherId = testId(0xA4);
    const auto leakMatcherId = testId(0xA5);
    const auto failMatcherId = testId(0xA6);

    KernelDescriptorPack survivingPack;
    survivingPack.id = testId(0xA0);
    survivingPack.name = "surviving pack";
    survivingPack.matcherIds = {passNoBindMatcherId};
    survivingPack.engineId = ENGINE_ID;
    survivingPack.dispatchId = DISPATCH_ID;
    survivingPack.kernels = {makeKernel(testId(0xA1), "surviving_kernel", 64, "FLOAT")};

    KernelDescriptorPack prunedPack;
    prunedPack.id = testId(0xA2);
    prunedPack.name = "pruned pack";
    // Order matters: the leaking matcher must run before the failing one prunes the
    // pack (matchers run in listed order).
    prunedPack.matcherIds = {leakMatcherId, failMatcherId};
    prunedPack.engineId = ENGINE_ID;
    prunedPack.dispatchId = DISPATCH_ID;
    prunedPack.kernels = {makeKernel(testId(0xA3), "pruned_kernel", 128, "FLOAT")};

    const StateManager manager(
        makeSchema(),
        {{passNoBindMatcherId, "passes, binds nothing", MatchScope::GRAPH, PASS_NO_BIND_SYMBOL},
         {leakMatcherId, "passes, binds a token", MatchScope::GRAPH, LEAK_THEN_PRUNE_SYMBOL},
         {failMatcherId, "always fails", MatchScope::GRAPH, ALWAYS_FAILS_SYMBOL}},
        makeTestDispatches(),
        {survivingPack, prunedPack},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const TestGraph graph(makeGraphId(50));
    const auto properties = testDeviceProperties();
    const auto catalog = manager.unsortedCatalog(MatchContext{graph, 0, properties});

    // Only the surviving pack's kernel enters the catalog.
    ASSERT_EQ(catalog.entries.size(), 1U);
    EXPECT_EQ(catalog.entries.front().packId, survivingPack.id);

    // The pruned pack's bound token must not reach the catalog the surviving pack
    // reads, even though its matcher returned true before the pack was pruned.
    EXPECT_EQ(catalog.bound.count("test.bound_token"), 0U);

    GraphMatcherRegistry::unregisterSymbol(PASS_NO_BIND_SYMBOL);
    GraphMatcherRegistry::unregisterSymbol(LEAK_THEN_PRUNE_SYMBOL);
    GraphMatcherRegistry::unregisterSymbol(ALWAYS_FAILS_SYMBOL);
}

TEST(TestKernelIngestorStateManager, TwoPacksBindingOneTokenToDifferentValuesThrows)
{
    // Conflicting values under one token name is an authoring error; a silent merge
    // would hide it.
    constexpr const char* CONFLICTING_SYMBOL = "test.bound_conflict.conflicting";
    // Ahead of the manager: matcher symbols resolve at construction.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    GraphMatcherRegistry::registerSymbol(CONFLICTING_SYMBOL, &bindConflictingTokenValue);

    const auto conflictingMatcherId = testId(0xB0);
    const KernelDescriptorPack first = makePack({GRAPH_MATCHER_ID});
    KernelDescriptorPack second = makePack({conflictingMatcherId});
    second.id = testId(0xB1);
    second.name = "the conflicting pack";
    second.kernels = {makeKernel(testId(0xB2), "second_pack_kernel", 512, "FLOAT")};

    const StateManager manager(makeSchema(),
                               {{GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, "test.graph"},
                                {conflictingMatcherId,
                                 "binds a conflicting value",
                                 MatchScope::GRAPH,
                                 CONFLICTING_SYMBOL}},
                               makeTestDispatches(),
                               {first, second},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const TestGraph graph(makeGraphId(51));
    const auto properties = testDeviceProperties();

    try
    {
        manager.unsortedCatalog(MatchContext{graph, 0, properties});
        ADD_FAILURE() << "expected a cross-pack token conflict to be reported";
    }
    catch(const std::runtime_error& error)
    {
        // The message has to identify which pack to go and fix, not only that some
        // pack disagreed.
        const std::string message = error.what();
        EXPECT_NE(message.find("the conflicting pack"), std::string::npos);
        EXPECT_NE(message.find(toString(second.id)), std::string::npos);
        EXPECT_NE(message.find("test.bound_token"), std::string::npos);
    }

    GraphMatcherRegistry::unregisterSymbol(CONFLICTING_SYMBOL);
}

TEST(TestKernelIngestorStateManager, TwoPacksBindingOneTokenToTheSameValueMergeCleanly)
{
    // Packs agreeing on a token's value must still merge cleanly.
    constexpr const char* AGREEING_SYMBOL = "test.bound_conflict.agreeing";
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    GraphMatcherRegistry::registerSymbol(AGREEING_SYMBOL, &bindAgreeingTokenValue);

    const auto agreeingMatcherId = testId(0xB3);
    const KernelDescriptorPack first = makePack({GRAPH_MATCHER_ID});
    KernelDescriptorPack second = makePack({agreeingMatcherId});
    second.id = testId(0xB4);
    second.kernels = {makeKernel(testId(0xB5), "second_pack_kernel", 512, "FLOAT")};

    const StateManager manager(
        makeSchema(),
        {{GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, "test.graph"},
         {agreeingMatcherId, "binds the same value", MatchScope::GRAPH, AGREEING_SYMBOL}},
        makeTestDispatches(),
        {first, second},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const TestGraph graph(makeGraphId(52));
    const auto properties = testDeviceProperties();

    const auto catalog = manager.unsortedCatalog(MatchContext{graph, 0, properties});

    EXPECT_EQ(tryGetBoundInt(catalog.bound, "test.bound_token"), BOUND_TOKEN_VALUE);
    // Agreement never prunes anything: all 4 kernels across both packs survive.
    EXPECT_EQ(catalog.entries.size(), 4U);
}

TEST(TestKernelIngestorStateManager, MatchesSeparatelyPerDevice)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(4));
    const auto properties = testDeviceProperties();

    manager->unsortedDefinitions(MatchContext{graph, 0, properties});
    manager->unsortedDefinitions(MatchContext{graph, 1, properties});

    // A kernel applicable on one device need not be applicable on another.
    EXPECT_EQ(counters().graphCalls, 2);
}

TEST(TestKernelIngestorStateManager, RematchesEveryCallWhenTheGraphHasNoIdentity)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    // A graph with no identity has no key to memoize under; inventing one would alias
    // unrelated graphs.
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto first = manager->unsortedDefinitions(context);
    const auto second = manager->unsortedDefinitions(context);

    // Correct both times; only the caching is lost.
    EXPECT_EQ(first.size(), 2U);
    EXPECT_EQ(second.size(), 2U);
    EXPECT_EQ(counters().graphCalls, 2);
}

TEST(TestKernelIngestorStateManager, ServesACachedRankingWithoutRematching)
{
    // The single-threaded half of D3: once a key is ranked, neither accessor rematches
    // or re-ranks it. The concurrent half needs two threads and lives in the concurrency
    // suite as ARankingSurvivesAConcurrentUnsortedAccess.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(0x5D));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    ASSERT_TRUE(manager->sortedCatalog(context).isSorted);

    static_cast<void>(manager->unsortedCatalog(context));

    EXPECT_TRUE(manager->sortedCatalog(context).isSorted);
    // One match total: everything after the first call was a cache hit.
    EXPECT_EQ(counters().graphCalls, 1);
}

TEST(TestKernelIngestorStateManager, DistinctGraphsCarryingANilUuidDoNotShareACatalogEntry)
{
    // Two graphs sharing the nil id must not share a cache entry, or the second would
    // get the first's catalog.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph first(makeNilGraphId());
    const TestGraph second(makeNilGraphId());
    const auto properties = testDeviceProperties();

    const auto firstDefinitions = manager->unsortedDefinitions(MatchContext{first, 0, properties});
    const auto secondDefinitions
        = manager->unsortedDefinitions(MatchContext{second, 0, properties});

    EXPECT_EQ(counters().graphCalls, 2);
    EXPECT_EQ(firstDefinitions.size(), 2U);
    EXPECT_EQ(secondDefinitions.size(), 2U);
}

// ---------------------------------------------------------------------------
// Bound state alongside the catalog
// ---------------------------------------------------------------------------

TEST(TestKernelIngestorStateManager, CarriesWhatMatchingBoundThroughToDispatch)
{
    // RFC 0017 section 8.5: matching also binds the fields dispatch reads, instead of
    // re-deriving them from the graph.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto properties = testDeviceProperties();
    const TestGraph graph;
    const MatchContext context{graph, 0, properties};

    const auto bound = manager->unsortedCatalog(context).bound;

    ASSERT_EQ(bound.count("test.bound_token"), 1U);
    // tryGetBoundInt also pins that the bound value is the integer alternative, not
    // another.
    EXPECT_EQ(tryGetBoundInt(bound, "test.bound_token"), BOUND_TOKEN_VALUE);
}

TEST(TestKernelIngestorStateManager, ReadingBoundStateAfterMatchingDoesNotRematch)
{
    // Section 8.1: bound state is cached with the catalog, so recovering it never
    // re-matches. The graph carries an identity, so a cache entry serves the second
    // and third reads.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto properties = testDeviceProperties();
    const TestGraph graph(makeGraphId(11));
    const MatchContext context{graph, 0, properties};

    static_cast<void>(manager->unsortedCatalog(context));
    const auto afterMatching = counters().graphCalls;

    // Asserting only graphCalls would miss a cache that emptied Catalog::bound; the
    // cached reads must carry it too.
    const auto secondRead = manager->unsortedCatalog(context).bound;
    const auto thirdRead = manager->sortedCatalog(context).bound;

    EXPECT_EQ(counters().graphCalls, afterMatching);
    EXPECT_EQ(tryGetBoundInt(secondRead, "test.bound_token"), BOUND_TOKEN_VALUE);
    EXPECT_EQ(tryGetBoundInt(thirdRead, "test.bound_token"), BOUND_TOKEN_VALUE);
}

// ---------------------------------------------------------------------------
// Caching
// ---------------------------------------------------------------------------

TEST(TestKernelIngestorStateManager, RematchesAfterCacheEviction)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager(SCORE_SYMBOL, 1);
    const auto properties = testDeviceProperties();
    const TestGraph first(makeGraphId(5));
    const TestGraph second(makeGraphId(6));

    manager->unsortedDefinitions(MatchContext{first, 0, properties});
    manager->unsortedDefinitions(MatchContext{second, 0, properties});
    manager->unsortedDefinitions(MatchContext{first, 0, properties});

    // Capacity 1 evicts the first graph, forcing a rematch; eviction costs work, not
    // correctness.
    EXPECT_EQ(counters().graphCalls, 3);
}

// ---------------------------------------------------------------------------
// Ranking reuse and knob value discovery
// ---------------------------------------------------------------------------

TEST(TestKernelIngestorStateManager, SortedDefinitionsAreRankedBestFirst)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(7));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto sorted = manager->sortedDefinitions(context);

    ASSERT_EQ(sorted.size(), 2U);
    EXPECT_EQ(sorted.front().getIntMetadata(BLOCK_SIZE), 256);
}

TEST(TestKernelIngestorStateManager, RankingReusesTheAlreadyMatchedCatalog)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(8));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    manager->unsortedDefinitions(context);
    manager->sortedDefinitions(context);
    manager->sortedDefinitions(context);

    // Ranking is a read of the cached catalog, never a rematch.
    EXPECT_EQ(counters().graphCalls, 1);
    EXPECT_EQ(counters().kernelCalls, 3);
}

TEST(TestKernelIngestorStateManager, KnobValuesComeFromTheCatalogInRankedOrder)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(9));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto values = StateManager::knobValues(manager->sortedDefinitions(context), BLOCK_SIZE);

    // The pruned HALF kernel also has block_size 64 but contributes no value; a knob
    // reflects the surviving catalog, not the schema range.
    ASSERT_EQ(values.size(), 2U);
    EXPECT_EQ(std::get<int64_t>(values[0]), 256);
    EXPECT_EQ(std::get<int64_t>(values[1]), 64);
}

// ---------------------------------------------------------------------------
// getDispatchDetails
// ---------------------------------------------------------------------------

TEST(TestKernelIngestorStateManager, GetDispatchDetailsThrowsOnADanglingDispatchId)
{
    // A missing dispatch after a graph was accepted is a hard error, not a silent
    // decline. Built directly, bypassing pack validation, since validateAndIndexPacks()
    // cannot see a definition that never went through a pack.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto kernel = makeDefinition(testId(0x01), 64);

    EXPECT_THROW(manager->getDispatchDetails(kernel), std::runtime_error);
}

// ---------------------------------------------------------------------------
// completeMetadata (exercised through the constructor/matching path)
// ---------------------------------------------------------------------------

TEST(TestKernelIngestorStateManager, CompletesAnOmittedFieldFromItsSchemaDefault)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    KernelDescriptorPack pack = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    // Omits block_size, which defaults, and states dtype, which does not.
    KernelDescriptor sparse;
    sparse.id = testId(0x70);
    sparse.name = "kernel_defaults";
    sparse.metadata = {{DTYPE, MetadataValue{std::string{"FLOAT"}}}};
    pack.kernels = {sparse};

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {pack},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const TestGraph graph(makeGraphId(10));
    const auto properties = testDeviceProperties();
    const auto definitions = manager.unsortedDefinitions(MatchContext{graph, 0, properties});

    ASSERT_EQ(definitions.size(), 1U);
    EXPECT_EQ(definitions.front().getIntMetadata(BLOCK_SIZE), 64);
    EXPECT_EQ(definitions.front().getStringMetadata(DTYPE), "FLOAT");
}

// ---------------------------------------------------------------------------
// Construction-time validation (TEST_P): each case must fail construction with
// std::invalid_argument, carrying a message that identifies what to fix.
// ---------------------------------------------------------------------------

struct StateManagerConstructionThrowCase
{
    std::string name;
    /// Substring the failure must contain. Asserting the type alone cannot tell a
    /// rejected descriptor apart from an unrelated bug that happens to throw the same
    /// type. After ALMIOPEN-2401 these messages are what an operator gets instead of a
    /// compiler error, so they are part of the contract.
    std::string expectedMessageSubstring;
    std::function<std::unique_ptr<StateManager>()> construct;
};

class TestKernelIngestorStateManagerConstructionThrows
    : public ::testing::TestWithParam<StateManagerConstructionThrowCase>
{
};

TEST_P(TestKernelIngestorStateManagerConstructionThrows, RejectsAtConstruction)
{
    // Every case below names the fixture's symbols, and the constructor resolves them
    // eagerly, so without these registered each case would throw runtime_error for an
    // unresolved symbol before reaching the invalid_argument it exists to assert.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    try
    {
        GetParam().construct();
        FAIL() << "expected std::invalid_argument";
    }
    catch(const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find(GetParam().expectedMessageSubstring),
                  std::string::npos)
            << "message did not explain the rejection: " << error.what();
    }
}

INSTANTIATE_TEST_SUITE_P(
    EagerValidationChecks,
    TestKernelIngestorStateManagerConstructionThrows,
    ::testing::Values(
        StateManagerConstructionThrowCase{
            "RejectsAKernelOmittingAFieldWithNoDefault",
            "which declares no default",
            [] {
                // dtype has no schema default, so every kernel must state it explicitly.
                KernelDescriptorPack pack = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
                KernelDescriptor missingDtype;
                missingDtype.id = testId(0x71);
                missingDtype.name = "kernel_missing_dtype";
                missingDtype.metadata = {{BLOCK_SIZE, MetadataValue{int64_t{64}}}};
                pack.kernels = {missingDtype};
                return std::make_unique<StateManager>(
                    makeSchema(),
                    makeTestMatchers(),
                    makeTestDispatches(),
                    std::vector<KernelDescriptorPack>{pack},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
            }},
        StateManagerConstructionThrowCase{
            "RejectsAKernelSupplyingAFieldTheSchemaDoesNotDeclare",
            "does not declare",
            [] {
                // An undeclared field (e.g. a misspelling) silently takes its default
                // while the stray value joins the catalog key unread.
                KernelDescriptorPack pack = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
                KernelDescriptor undeclared;
                undeclared.id = testId(0x74);
                undeclared.name = "kernel_undeclared_field";
                undeclared.metadata = {{BLOCK_SIZE, MetadataValue{int64_t{64}}},
                                       {DTYPE, MetadataValue{std::string{"FLOAT"}}},
                                       {"blocksize", MetadataValue{int64_t{128}}}};
                pack.kernels = {undeclared};
                return std::make_unique<StateManager>(
                    makeSchema(),
                    makeTestMatchers(),
                    makeTestDispatches(),
                    std::vector<KernelDescriptorPack>{pack},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
            }},
        StateManagerConstructionThrowCase{
            "RejectsAKernelSupplyingAFieldOfTheWrongType",
            "a value of the wrong type",
            [] {
                // A wrong-typed field would otherwise surface as a bad_variant_access far
                // away, inside a matcher or scorer.
                KernelDescriptorPack pack = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
                KernelDescriptor wrongType;
                wrongType.id = testId(0x72);
                wrongType.name = "kernel_wrong_type";
                wrongType.metadata = {{BLOCK_SIZE, MetadataValue{std::string{"64"}}},
                                      {DTYPE, MetadataValue{std::string{"FLOAT"}}}};
                pack.kernels = {wrongType};
                return std::make_unique<StateManager>(
                    makeSchema(),
                    makeTestMatchers(),
                    makeTestDispatches(),
                    std::vector<KernelDescriptorPack>{pack},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
            }},
        StateManagerConstructionThrowCase{
            "RejectsAPackNamingAnUnknownMatcher",
            "names unknown matcher",
            [] {
                // A dangling matcher reference is caught at construction, not when a
                // graph first arrives.
                return std::make_unique<StateManager>(
                    makeSchema(),
                    std::vector<MatchDescriptor>{},
                    makeTestDispatches(),
                    std::vector<KernelDescriptorPack>{makePack({testId(0xFF)})},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
            }},
        StateManagerConstructionThrowCase{
            "RejectsAPackNamingAnUnknownDispatchDescriptor",
            "names unknown dispatch descriptor",
            [] {
                return std::make_unique<StateManager>(
                    makeSchema(),
                    makeTestMatchers(),
                    std::vector<DispatchDescriptor>{},
                    std::vector<KernelDescriptorPack>{makePack({GRAPH_MATCHER_ID})},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
            }},
        StateManagerConstructionThrowCase{
            "RejectsTwoKernelsSharingAMetadataTuple",
            "duplicates the metadata tuple",
            [] {
                // Same completed tuple as kernel_64_float: selection would have two
                // indistinguishable candidates and no basis to prefer either.
                KernelDescriptorPack pack = makePack({GRAPH_MATCHER_ID});
                pack.kernels.push_back(makeKernel(testId(0x73), "kernel_duplicate", 64, "FLOAT"));
                return std::make_unique<StateManager>(
                    makeSchema(),
                    makeTestMatchers(),
                    makeTestDispatches(),
                    std::vector<KernelDescriptorPack>{pack},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
            }},
        StateManagerConstructionThrowCase{
            "RejectsADuplicateMatchDescriptorId",
            "duplicate match descriptor id",
            [] {
                // Silent first-wins under a duplicate id would run whichever matcher
                // loaded first.
                std::vector<MatchDescriptor> matchers = makeTestMatchers();
                matchers.push_back({GRAPH_MATCHER_ID,
                                    "a different matcher entirely",
                                    MatchScope::KERNEL,
                                    "test.kernel"});
                return std::make_unique<StateManager>(
                    makeSchema(),
                    matchers,
                    makeTestDispatches(),
                    std::vector<KernelDescriptorPack>{makePack({GRAPH_MATCHER_ID})},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
            }},
        StateManagerConstructionThrowCase{
            "RejectsADuplicateDispatchDescriptorId",
            "duplicate dispatch descriptor id",
            [] {
                std::vector<DispatchDescriptor> dispatches = makeTestDispatches();
                dispatches.push_back(
                    {DISPATCH_ID, "a different dispatch entirely", "test.dispatch.other"});
                return std::make_unique<StateManager>(
                    makeSchema(),
                    makeTestMatchers(),
                    dispatches,
                    std::vector<KernelDescriptorPack>{makePack({GRAPH_MATCHER_ID})},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
            }},
        StateManagerConstructionThrowCase{"RejectsAMissingHeuristic",
                                          "requires a heuristic",
                                          [] {
                                              return std::make_unique<StateManager>(
                                                  makeSchema(),
                                                  std::vector<MatchDescriptor>{},
                                                  std::vector<DispatchDescriptor>{},
                                                  std::vector<KernelDescriptorPack>{},
                                                  nullptr);
                                          }}),
    [](const ::testing::TestParamInfo<StateManagerConstructionThrowCase>& info) {
        return info.param.name;
    });

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
