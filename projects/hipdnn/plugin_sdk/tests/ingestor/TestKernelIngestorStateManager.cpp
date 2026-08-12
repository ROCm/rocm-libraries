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
 * @brief Unit tests for KernelIngestorStateManager.hpp: matching, pruning, caching,
 *        ranking-reuse, knob value discovery, metadata completion, and the eager
 *        construction-time validation the constructor's doc calls out.
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
    // The whole point of the graph/kernel split: a pack that cannot serve the graph
    // costs one matcher call, not one per kernel.
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

    // Matchers are shared by id across packs, so a shared graph-scoped check runs once,
    // not once per pack.
    //
    // The first pack also fails a second, unshared matcher, so only the second pack
    // survives. That isolates BINDING REUSE from memoization: if both packs survived,
    // the first pack's own merge could put the token in catalog.bound, hiding a
    // mutation that only merges a memo entry's bindings on the cache-MISS pack and
    // skips it on every cache-HIT pack -- exactly the second pack's read here.
    constexpr const char* UNSHARED_FAIL_SYMBOL = "test.cross_pack_binding_reuse.fail";
    GraphMatcherRegistry::registerSymbol(UNSHARED_FAIL_SYMBOL, rejectGraph);
    const auto unsharedFailMatcherId = testId(0x84);

    const KernelDescriptorPack first = makePack({GRAPH_MATCHER_ID, unsharedFailMatcherId});
    KernelDescriptorPack second = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    second.id = testId(0x80);
    // A distinct metadata tuple: the uniqueness rule is engine-wide, not per pack, so
    // the second pack cannot repeat a block size the first already uses.
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

    // Three native calls would mean the shared matcher ran twice; it must not. One call
    // for GRAPH_MATCHER_ID (shared, evaluated once for both packs) plus one for the
    // first pack's own unshared, failing matcher is the whole budget.
    EXPECT_EQ(counters().graphCalls, 2);
    // Only the second pack's kernel: the first pack's graph-scoped matchers did not all
    // pass, so none of its kernels were ever considered.
    ASSERT_EQ(catalog.entries.size(), 1U);
    EXPECT_EQ(catalog.entries.front().packId, second.id);
    // The point of this test: the shared matcher's binding reaches catalog.bound through
    // the second pack's read of the memo, not through the (pruned) first pack that
    // actually ran the matcher.
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
    // Two packs sharing an engine, the normal case since matchers are shared by id.
    // The pruned pack's own graph-scoped matcher passes and binds a token before a
    // second matcher in the SAME pack fails and prunes it; the surviving pack lists
    // neither of those matchers. If the pruned pack's binding reaches catalog.bound
    // anyway, a kernel in the surviving pack -- which never asked for that token -- can
    // read it. That is the correctness bug item 5 fixes: the header's claim that
    // "nothing a failing matcher wrote can be read by a kernel that survives" held only
    // by accident with one pack, and breaks the moment a second pack shares the engine.
    constexpr const char* PASS_NO_BIND_SYMBOL = "test.pruned_bound_isolation.pass_no_bind";
    constexpr const char* LEAK_THEN_PRUNE_SYMBOL = "test.pruned_bound_isolation.leak";
    constexpr const char* ALWAYS_FAILS_SYMBOL = "test.pruned_bound_isolation.fail";

    // Reused rather than reinvented: acceptAnyGraph (fixtures) passes and binds nothing;
    // acceptGraph (this file's fixtures) passes and binds "test.bound_token"; rejectGraph
    // always fails. Exactly the three behaviors this test needs.
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
    // Order matters: the leaking matcher must run (and bind) before the failing one
    // prunes the pack, exactly as graphLevelMatchersPass evaluates them in listed order.
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

    // The surviving pack's kernel is the only entry: the pruned pack's kernel never
    // enters the catalog, which was already correct before this fix.
    ASSERT_EQ(catalog.entries.size(), 1U);
    EXPECT_EQ(catalog.entries.front().packId, survivingPack.id);

    // What this test actually regresses: the token only the pruned pack's own matcher
    // bound must not survive into the catalog the surviving pack's kernel reads from,
    // even though that matcher itself returned true before its pack's second matcher
    // failed.
    EXPECT_EQ(catalog.bound.count("test.bound_token"), 0U);

    GraphMatcherRegistry::unregisterSymbol(PASS_NO_BIND_SYMBOL);
    GraphMatcherRegistry::unregisterSymbol(LEAK_THEN_PRUNE_SYMBOL);
    GraphMatcherRegistry::unregisterSymbol(ALWAYS_FAILS_SYMBOL);
}

TEST(TestKernelIngestorStateManager, TwoPacksBindingOneTokenToDifferentValuesThrows)
{
    // Two packs writing different values under one token name is an authoring error
    // (Catalog::bound's doc); a silent first-wins merge would hide it.
    constexpr const char* CONFLICTING_SYMBOL = "test.bound_conflict.conflicting";
    GraphMatcherRegistry::registerSymbol(CONFLICTING_SYMBOL, &bindConflictingTokenValue);

    const auto conflictingMatcherId = testId(0xB0);
    const KernelDescriptorPack first = makePack({GRAPH_MATCHER_ID});
    KernelDescriptorPack second = makePack({conflictingMatcherId});
    second.id = testId(0xB1);
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

    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const TestGraph graph(makeGraphId(51));
    const auto properties = testDeviceProperties();

    EXPECT_THROW(manager.unsortedCatalog(MatchContext{graph, 0, properties}), std::runtime_error);

    GraphMatcherRegistry::unregisterSymbol(CONFLICTING_SYMBOL);
}

TEST(TestKernelIngestorStateManager, TwoPacksBindingOneTokenToTheSameValueMergeCleanly)
{
    // Packs agreeing on a token's value (the normal shared-matcher case) must still
    // merge cleanly.
    constexpr const char* AGREEING_SYMBOL = "test.bound_conflict.agreeing";
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

    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
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

    // The same graph on a different device is a different problem: a kernel applicable
    // on one device need not be applicable on another.
    EXPECT_EQ(counters().graphCalls, 2);
}

TEST(TestKernelIngestorStateManager, RematchesEveryCallWhenTheGraphHasNoIdentity)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    // A legacy or unfinalized graph: there is no key to memoize under, and inventing one
    // would alias unrelated graphs.
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

TEST(TestKernelIngestorStateManager, DistinctGraphsCarryingANilUuidDoNotShareACatalogEntry)
{
    // Two unrelated graphs sharing the nil id must not share a cache entry: a regression
    // would run the matcher once, not twice, and serve the second graph the first's catalog.
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
    // RFC 0017 section 8.5: matching does double duty, deciding a kernel applies and
    // binding the fields the launch will use. A dispatch handler must read those values
    // rather than re-deriving them from the graph with a second notion of its shape.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto properties = testDeviceProperties();
    const TestGraph graph;
    const MatchContext context{graph, 0, properties};

    const auto bound = manager->unsortedCatalog(context).bound;

    ASSERT_EQ(bound.count("test.bound_token"), 1U);
    // Compared as the integer alternative rather than against the variant as a whole:
    // a bare int64 on the right would not convert, and asserting through the accessor
    // also pins that the matcher bound an integer and not some other alternative.
    EXPECT_EQ(tryGetBoundInt(bound, "test.bound_token"), BOUND_TOKEN_VALUE);
}

TEST(TestKernelIngestorStateManager, ReadingBoundStateAfterMatchingDoesNotRematch)
{
    // Section 8.1 keeps the bound token state alongside the catalog precisely so that
    // "nothing is re-matched" once a graph has been matched. Recovering these values by
    // re-running the matcher would be correct and quietly quadratic.
    //
    // The graph carries an identity, so there is a cache entry to serve the second and
    // third reads. A graph without one is matched fresh every call by design; that is why
    // the entries and the bound state come back from a single call rather than two.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto properties = testDeviceProperties();
    const TestGraph graph(makeGraphId(11));
    const MatchContext context{graph, 0, properties};

    static_cast<void>(manager->unsortedCatalog(context));
    const auto afterMatching = counters().graphCalls;

    // The cached reads must still carry the bound state, not merely avoid re-matching.
    // Asserting only graphCalls would pass against a cache that served an entry with
    // Catalog::bound emptied -- which destroys the section 8.1 property this test is
    // named for while leaving its counter untouched.
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

    // Capacity 1, so the second graph evicted the first and it had to rematch. Eviction
    // costs work, never a wrong answer.
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

    // The pruned HALF kernel also carried block_size 64, but it contributes no value:
    // a knob offers what the surviving catalog implements, not the schema's range.
    ASSERT_EQ(values.size(), 2U);
    EXPECT_EQ(std::get<int64_t>(values[0]), 256);
    EXPECT_EQ(std::get<int64_t>(values[1]), 64);
}

// ---------------------------------------------------------------------------
// getDispatchDetails
// ---------------------------------------------------------------------------

TEST(TestKernelIngestorStateManager, GetDispatchDetailsThrowsOnADanglingDispatchId)
{
    // After applicability accepted a graph, hipDNN has already chosen this engine on
    // that promise, so a missing dispatch is a hard error rather than a silent decline.
    // Constructed directly rather than through the pack/matcher path: the point here is
    // a KernelDefinition whose dispatchId the state manager's own descriptor set never
    // registered, which validateAndIndexPacks() cannot catch because this definition
    // never went through a pack.
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
// Construction-time validation: every eager check validateAndIndexPacks() and the
// constructor itself run, TEST_P'd over one shared shape (a pack, invalid in one way,
// must fail construction with std::invalid_argument).
// ---------------------------------------------------------------------------

struct StateManagerConstructionThrowCase
{
    std::string name;
    std::function<std::unique_ptr<StateManager>()> construct;
};

class TestKernelIngestorStateManagerConstructionThrows
    : public ::testing::TestWithParam<StateManagerConstructionThrowCase>
{
};

TEST_P(TestKernelIngestorStateManagerConstructionThrows, RejectsAtConstruction)
{
    EXPECT_THROW(GetParam().construct(), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    EagerValidationChecks,
    TestKernelIngestorStateManagerConstructionThrows,
    ::testing::Values(
        StateManagerConstructionThrowCase{
            "RejectsAKernelOmittingAFieldWithNoDefault",
            [] {
                // dtype declares no default, so every kernel must state it. Omitting it
                // would otherwise produce a catalog key the author never wrote.
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
            [] {
                // A misspelled field name is the common case, and left in place it fails
                // twice over: the real field silently takes its default, and the stray
                // value joins the catalog key, where nothing downstream can read it.
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
            [] {
                // block_size is declared INT. A string here would otherwise surface far
                // away, as a bad_variant_access inside a matcher or a scorer.
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
            [] {
                // A dangling cross-reference cannot be evaluated, so it is caught when
                // the descriptor set is assembled rather than when a graph first arrives.
                return std::make_unique<StateManager>(
                    makeSchema(),
                    std::vector<MatchDescriptor>{},
                    makeTestDispatches(),
                    std::vector<KernelDescriptorPack>{makePack({testId(0xFF)})},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
            }},
        StateManagerConstructionThrowCase{
            "RejectsAPackNamingAnUnknownDispatchDescriptor",
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
            [] {
                // Two distinct matchers under one id: silent first-wins would let a pack
                // run whichever matcher loaded first.
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
