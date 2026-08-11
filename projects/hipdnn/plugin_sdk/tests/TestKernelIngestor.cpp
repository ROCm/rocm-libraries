// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlanBuilder.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/LruCache.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestKernelIngestor.cpp
 * @brief Unit tests for the descriptor-driven kernel ingestor's SDK-side machinery.
 *
 * Covers the pieces every descriptor-backed engine depends on, independent of any
 * provider: the bounded catalog cache, the native symbol registry's fail-closed
 * behavior, the heuristic's ranking and tie-break order, and the state manager's
 * matching, pruning, caching, and validation.
 *
 * Provider-specific behavior (the pointwise-add matchers and the real GPU dispatch)
 * is covered by the hip-kernel-provider tests.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;

using namespace hipdnn_plugin_sdk::ingestor::testing;

/// Counts matcher invocations, so a test can assert that a graph-scoped matcher runs
/// once per (graph, device) while a kernel-scoped one runs once per surviving kernel.
struct MatcherCounters
{
    int graphCalls = 0;
    int kernelCalls = 0;

    void reset()
    {
        graphCalls = 0;
        kernelCalls = 0;
    }
};

MatcherCounters& counters()
{
    static MatcherCounters s_counters;
    return s_counters;
}

/// An arbitrary value acceptGraph binds, so a test can prove it reached dispatch intact.
constexpr int64_t BOUND_TOKEN_VALUE = 4242;

bool acceptGraph(const MatchContext& /*context*/, BoundTokens& bound)
{
    ++counters().graphCalls;
    // Stands in for the tensor uids and dimensions a real matcher resolves: the point
    // under test is that whatever matching bound reaches dispatch without a re-match.
    bound["test.bound_token"] = BOUND_TOKEN_VALUE;
    return true;
}

bool rejectGraph(const MatchContext& /*context*/, BoundTokens& /*bound*/)
{
    ++counters().graphCalls;
    return false;
}

bool countingFloatKernels(const MatchContext& context, const KernelDefinition& kernel)
{
    ++counters().kernelCalls;
    return acceptFloatKernels(context, kernel);
}

/// Every kernel scores the same, so ranking falls through to the tie-break.
constexpr const char* CONSTANT_SCORE_SYMBOL = "hipdnn.kernel_ingestor.test.constant_score";

double scoreConstant(const KernelDefinition& /*kernel*/, const MatchContext& /*context*/)
{
    return 1.0;
}

/// Registers the constant scorer for a test's duration. The heuristic resolves its
/// symbol on first use, so the registration has to outlive the ranking call.
class ScopedConstantScore
{
public:
    ScopedConstantScore()
    {
        ScoreRegistry::registerSymbol(CONSTANT_SCORE_SYMBOL, &scoreConstant);
    }

    ~ScopedConstantScore()
    {
        ScoreRegistry::unregisterSymbol(CONSTANT_SCORE_SYMBOL);
    }

    ScopedConstantScore(const ScopedConstantScore&) = delete;
    ScopedConstantScore& operator=(const ScopedConstantScore&) = delete;
};

MetadataSchema makeSchema()
{
    return {SCHEMA_ID,
            "test schema",
            {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
             {DTYPE, MetadataType::STRING, std::nullopt}}};
}

KernelDescriptor makeKernel(const DescriptorId& id,
                            const std::string& name,
                            int64_t blockSize,
                            const std::string& dtype,
                            int64_t priority = 0)
{
    auto kernel = makeTestKernel(id, name, blockSize, dtype);
    kernel.priority = priority;
    return kernel;
}

/// The pack shape a real engine ships, wired to the counting matchers above.
KernelDescriptorPack makePack(const std::vector<DescriptorId>& matcherIds)
{
    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "test pack";
    pack.matcherIds = matcherIds;
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeKernel(testId(0x64), "kernel_64_float", 64, "FLOAT"),
                    makeKernel(testId(0x65), "kernel_256_float", 256, "FLOAT"),
                    makeKernel(testId(0x66), "kernel_64_half", 64, "HALF")};
    return pack;
}

std::vector<MatchDescriptor> makeTestMatchers()
{
    return {{GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, "test.graph"},
            {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, "test.kernel"}};
}

std::vector<DispatchDescriptor> makeTestDispatches()
{
    return {{DISPATCH_ID, "test dispatch", "test.dispatch"}};
}

/// A catalog entry, for the ranking tests that build one directly.
KernelDefinition makeDefinition(const DescriptorId& id, int64_t blockSize, int64_t priority = 0)
{
    return {id,
            PACK_ID,
            DISPATCH_ID,
            KernelSource{KernelSourceKind::EMBEDDED_SOURCE, "Test.cpp", "TestKernel"},
            {{BLOCK_SIZE, MetadataValue{blockSize}}},
            priority};
}

/// Registers counting matchers under this file's own symbol names, so these tests
/// observe invocation counts without disturbing the shared fixture's registrations.
class ScopedSymbols
{
public:
    ScopedSymbols(std::string graphSymbol,
                  GraphMatcherFn graphFn,
                  std::string kernelSymbol,
                  KernelMatcherFn kernelFn)
        : _graphSymbol(std::move(graphSymbol))
        , _kernelSymbol(std::move(kernelSymbol))
    {
        GraphMatcherRegistry::registerSymbol(_graphSymbol, graphFn);
        KernelMatcherRegistry::registerSymbol(_kernelSymbol, kernelFn);
        // The heuristic resolves its symbol on the first call that needs an order, not
        // at construction, so ranking tests need it registered for their duration too.
        ScoreRegistry::registerSymbol(SCORE_SYMBOL, &scoreByBlockSize);
        counters().reset();
    }

    ~ScopedSymbols()
    {
        GraphMatcherRegistry::unregisterSymbol(_graphSymbol);
        KernelMatcherRegistry::unregisterSymbol(_kernelSymbol);
        ScoreRegistry::unregisterSymbol(SCORE_SYMBOL);
    }

    ScopedSymbols(const ScopedSymbols&) = delete;
    ScopedSymbols& operator=(const ScopedSymbols&) = delete;

private:
    std::string _graphSymbol;
    std::string _kernelSymbol;
};

using TestHandle = int;
using StateManager = KernelIngestorStateManager<TestHandle>;

std::unique_ptr<StateManager> makeStateManager(const std::string& scoreSymbol = SCORE_SYMBOL,
                                               size_t cacheCapacity
                                               = StateManager::DEFAULT_CATALOG_CACHE_CAPACITY)
{
    std::vector<MatchDescriptor> matchers{
        {GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, "test.graph"},
        {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, "test.kernel"}};
    std::vector<DispatchDescriptor> dispatches{{DISPATCH_ID, "test dispatch", "test.dispatch"}};

    return std::make_unique<StateManager>(
        makeSchema(),
        std::move(matchers),
        std::move(dispatches),
        std::vector<KernelDescriptorPack>{makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID})},
        std::make_shared<NativeKernelHeuristic>(scoreSymbol),
        cacheCapacity);
}

// ---------------------------------------------------------------------------
// LruCache
// ---------------------------------------------------------------------------

TEST(TestIngestorLruCache, ReturnsCachedValueWithinCapacity)
{
    LruCache<int, std::string> cache(2);
    cache.put(1, "one");

    const auto found = cache.get(1);

    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, "one");
}

TEST(TestIngestorLruCache, ReportsMissForAbsentKey)
{
    LruCache<int, std::string> cache(2);

    EXPECT_FALSE(cache.get(42).has_value());
}

TEST(TestIngestorLruCache, EvictsLeastRecentlyUsedPastCapacity)
{
    LruCache<int, std::string> cache(2);
    cache.put(1, "one");
    cache.put(2, "two");
    cache.put(3, "three");

    EXPECT_FALSE(cache.get(1).has_value());
    EXPECT_TRUE(cache.get(2).has_value());
    EXPECT_TRUE(cache.get(3).has_value());
    EXPECT_EQ(cache.size(), 2U);
}

TEST(TestIngestorLruCache, ReadingAnEntryProtectsItFromEviction)
{
    LruCache<int, std::string> cache(2);
    cache.put(1, "one");
    cache.put(2, "two");

    // Touching 1 makes 2 the least recently used, so inserting 3 must evict 2, not 1.
    ASSERT_TRUE(cache.get(1).has_value());
    cache.put(3, "three");

    EXPECT_TRUE(cache.get(1).has_value());
    EXPECT_FALSE(cache.get(2).has_value());
}

TEST(TestIngestorLruCache, OverwritingAKeyDoesNotGrowTheCache)
{
    LruCache<int, std::string> cache(2);
    cache.put(1, "one");
    cache.put(1, "uno");

    const auto found = cache.get(1);

    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, "uno");
    EXPECT_EQ(cache.size(), 1U);
}

TEST(TestIngestorLruCache, RejectsZeroCapacity)
{
    // A zero-capacity cache would evict every entry as it was inserted, which is always
    // a caller bug rather than a way to disable caching.
    using IntCache = LruCache<int, int>;
    EXPECT_THROW(IntCache(0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// NativeRegistry
// ---------------------------------------------------------------------------

TEST(TestIngestorNativeRegistry, ResolvesARegisteredSymbol)
{
    GraphMatcherRegistry::registerSymbol("registry.resolves", acceptGraph);

    EXPECT_EQ(GraphMatcherRegistry::resolve("registry.resolves"), acceptGraph);

    GraphMatcherRegistry::unregisterSymbol("registry.resolves");
}

TEST(TestIngestorNativeRegistry, RejectsDuplicateRegistration)
{
    GraphMatcherRegistry::registerSymbol("registry.duplicate", acceptGraph);

    // Two implementations behind one name leaves one silently unreachable, and which one
    // wins would depend on static-init order.
    EXPECT_THROW(GraphMatcherRegistry::registerSymbol("registry.duplicate", rejectGraph),
                 std::runtime_error);

    GraphMatcherRegistry::unregisterSymbol("registry.duplicate");
}

TEST(TestIngestorNativeRegistry, FailsClosedOnUnknownSymbol)
{
    // A descriptor naming a symbol the provider does not ship must surface as an error,
    // never as an engine that quietly matches nothing.
    EXPECT_THROW(GraphMatcherRegistry::resolve("registry.never_registered"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// IKernelHeuristic
// ---------------------------------------------------------------------------

TEST(TestIngestorHeuristic, ResolvesItsSymbolOnFirstUseRatherThanAtConstruction)
{
    // RFC 0017 section 8.1 admits the heuristic at applicability only as a name, and
    // section 3 generalizes it: a heuristic model is not read until something needs the
    // catalog ranked. An engine whose matchers reject a graph must never pay for its
    // selector, so constructing one against a symbol that is not registered yet has to
    // be legal, and only scoring may fail.
    const NativeKernelHeuristic heuristic("hipdnn.kernel_ingestor.test.not_yet_registered");

    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    const auto kernel = makeDefinition(testId(0x01), 64);

    EXPECT_THROW(heuristic.score(kernel, context), std::runtime_error);
}

TEST(TestIngestorHeuristic, RanksHigherScoringKernelsFirst)
{
    const ScopedTestSymbols symbols;
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    const auto lowId = testId(0x01);
    const auto highId = testId(0x02);
    catalog.entries = {makeDefinition(lowId, 64), makeDefinition(highId, 256)};

    const NativeKernelHeuristic heuristic(SCORE_SYMBOL);
    const auto ranked = heuristic.rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, highId);
}

TEST(TestIngestorHeuristic, BreaksScoreTiesOnPriority)
{
    const ScopedConstantScore constantScore;
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    const auto lowPriorityId = testId(0x01);
    const auto highPriorityId = testId(0x02);
    catalog.entries = {makeDefinition(lowPriorityId, 64, 1), makeDefinition(highPriorityId, 64, 5)};

    const NativeKernelHeuristic heuristic(CONSTANT_SCORE_SYMBOL);
    const auto ranked = heuristic.rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, highPriorityId);
}

TEST(TestIngestorHeuristic, BreaksRemainingTiesOnKernelIdForStabilityAcrossRuns)
{
    const ScopedConstantScore constantScore;
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    // Deliberately inserted out of id order: the result must not depend on load order.
    // Inserted highest-id first: the result must not depend on load order.
    const auto lowerId = testId(0x01);
    const auto higherId = testId(0x02);
    catalog.entries = {makeDefinition(higherId, 64), makeDefinition(lowerId, 64)};

    const NativeKernelHeuristic heuristic(CONSTANT_SCORE_SYMBOL);
    const auto ranked = heuristic.rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, lowerId);
}

// ---------------------------------------------------------------------------
// KernelIngestorStateManager
// ---------------------------------------------------------------------------

TEST(TestIngestorStateManager, KernelLevelMatcherPrunesTheCatalog)
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

TEST(TestIngestorStateManager, GraphLevelMatcherFailurePrunesTheWholePack)
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

TEST(TestIngestorStateManager, MatchesOncePerGraphAndDevice)
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

TEST(TestIngestorStateManager, EvaluatesASharedGraphMatcherOncePerGraphNotOncePerPack)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    // Matchers are shared by id across packs, so the broadly shared checks are what prune
    // the candidate set fast. Re-running one per pack that lists it would make the most
    // shared check the most expensive, inverting the property the graph/kernel split
    // exists to provide.
    auto first = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    auto second = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    second.id = testId(0x80);
    // A distinct metadata tuple: the uniqueness rule is engine-wide, not per pack, so
    // the second pack cannot repeat a block size the first already uses.
    second.kernels = {makeKernel(testId(0x81), "second_pack_kernel", 512, "FLOAT")};

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {first, second},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const TestGraph graph(makeGraphId(20));
    const auto properties = testDeviceProperties();
    const auto definitions = manager.unsortedDefinitions(MatchContext{graph, 0, properties});

    // Two packs, one shared graph-scoped matcher: evaluated once.
    EXPECT_EQ(counters().graphCalls, 1);
    // Two FLOAT survivors from the first pack plus one from the second.
    EXPECT_EQ(definitions.size(), 3U);
}

TEST(TestIngestorStateManager, ASharedGraphMatcherFailurePrunesEveryPackListingIt)
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

TEST(TestIngestorStateManager, PrunedPackBindingsAreNotVisibleToSurvivingPackKernels)
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
    // acceptGraph (this file) passes and binds "test.bound_token"; rejectGraph (this
    // file) always fails. Exactly the three behaviors this test needs.
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

TEST(TestIngestorStateManager, MatchesSeparatelyPerDevice)
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

TEST(TestIngestorStateManager, RematchesEveryCallWhenTheGraphHasNoIdentity)
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

TEST(TestIngestorStateManager, CarriesWhatMatchingBoundThroughToDispatch)
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
    EXPECT_EQ(bound.at("test.bound_token"), BOUND_TOKEN_VALUE);
}

TEST(TestIngestorStateManager, ReadingBoundStateAfterMatchingDoesNotRematch)
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

    static_cast<void>(manager->unsortedCatalog(context).bound);
    static_cast<void>(manager->sortedCatalog(context).bound);

    EXPECT_EQ(counters().graphCalls, afterMatching);
}

TEST(TestIngestorStateManager, RematchesAfterCacheEviction)
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

TEST(TestIngestorStateManager, SortedDefinitionsAreRankedBestFirst)
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

TEST(TestIngestorStateManager, RankingReusesTheAlreadyMatchedCatalog)
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

TEST(TestIngestorStateManager, KnobValuesComeFromTheCatalogInRankedOrder)
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

TEST(TestIngestorStateManager, CompletesAnOmittedFieldFromItsSchemaDefault)
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

TEST(TestIngestorStateManager, RejectsAKernelOmittingAFieldWithNoDefault)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    KernelDescriptorPack pack = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    // dtype declares no default, so every kernel must state it. Omitting it would
    // otherwise produce a catalog key the author never wrote.
    KernelDescriptor missingDtype;
    missingDtype.id = testId(0x71);
    missingDtype.name = "kernel_missing_dtype";
    missingDtype.metadata = {{BLOCK_SIZE, MetadataValue{int64_t{64}}}};
    pack.kernels = {missingDtype};

    // Caught when the descriptor set is assembled, not when a graph first arrives: the
    // completed tuple is the kernel's catalog key, so it has to exist before matching.
    EXPECT_THROW(StateManager(makeSchema(),
                              makeTestMatchers(),
                              makeTestDispatches(),
                              {pack},
                              std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL)),
                 std::invalid_argument);
}

TEST(TestIngestorStateManager, RejectsAKernelSupplyingAFieldTheSchemaDoesNotDeclare)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    KernelDescriptorPack pack = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    // A misspelled field name is the common case, and left in place it fails twice over:
    // the real field silently takes its default, and the stray value joins the catalog
    // key, where nothing downstream can read it. Two kernels differing only in such a
    // value would both enter the catalog with selection unable to tell them apart.
    KernelDescriptor undeclared;
    undeclared.id = testId(0x74);
    undeclared.name = "kernel_undeclared_field";
    undeclared.metadata = {{BLOCK_SIZE, MetadataValue{int64_t{64}}},
                           {DTYPE, MetadataValue{std::string{"FLOAT"}}},
                           {"blocksize", MetadataValue{int64_t{128}}}};
    pack.kernels = {undeclared};

    EXPECT_THROW(StateManager(makeSchema(),
                              makeTestMatchers(),
                              makeTestDispatches(),
                              {pack},
                              std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL)),
                 std::invalid_argument);
}

TEST(TestIngestorStateManager, RejectsAKernelSupplyingAFieldOfTheWrongType)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    KernelDescriptorPack pack = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    // block_size is declared INT. A string here would otherwise surface far away, as a
    // bad_variant_access inside a matcher or a scorer.
    KernelDescriptor wrongType;
    wrongType.id = testId(0x72);
    wrongType.name = "kernel_wrong_type";
    wrongType.metadata = {{BLOCK_SIZE, MetadataValue{std::string{"64"}}},
                          {DTYPE, MetadataValue{std::string{"FLOAT"}}}};
    pack.kernels = {wrongType};

    EXPECT_THROW(StateManager(makeSchema(),
                              makeTestMatchers(),
                              makeTestDispatches(),
                              {pack},
                              std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL)),
                 std::invalid_argument);
}

TEST(TestIngestorStateManager, RejectsAPackNamingAnUnknownMatcher)
{
    // A dangling cross-reference cannot be evaluated, so it is caught when the
    // descriptor set is assembled rather than when a graph first arrives.
    EXPECT_THROW(StateManager(makeSchema(),
                              {},
                              makeTestDispatches(),
                              {makePack({testId(0xFF)})},
                              std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL)),
                 std::invalid_argument);
}

TEST(TestIngestorStateManager, RejectsAPackNamingAnUnknownDispatchDescriptor)
{
    EXPECT_THROW(StateManager(makeSchema(),
                              makeTestMatchers(),
                              {},
                              {makePack({GRAPH_MATCHER_ID})},
                              std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL)),
                 std::invalid_argument);
}

TEST(TestIngestorStateManager, RejectsTwoKernelsSharingAMetadataTuple)
{
    KernelDescriptorPack pack = makePack({GRAPH_MATCHER_ID});
    // Same completed tuple as kernel_64_float: selection would have two indistinguishable
    // candidates and no basis to prefer either.
    pack.kernels.push_back(makeKernel(testId(0x73), "kernel_duplicate", 64, "FLOAT"));

    EXPECT_THROW(StateManager(makeSchema(),
                              makeTestMatchers(),
                              makeTestDispatches(),
                              {pack},
                              std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL)),
                 std::invalid_argument);
}

TEST(TestIngestorStateManager, RejectsAMissingHeuristic)
{
    EXPECT_THROW(StateManager(makeSchema(), {}, {}, {}, nullptr), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// GenericEngine: engine-level descriptor validation
// ---------------------------------------------------------------------------

/// Minimal stand-ins for the provider types GenericEngine is parameterized on. Only the
/// members the engine actually touches are present, so a change that starts depending on
/// more of a real handle or context fails here rather than compiling silently.
struct StubHandle
{
    void storeEngineDetailsDetachedBuffer(const void* /*ptr*/,
                                          std::unique_ptr<flatbuffers::DetachedBuffer> buffer)
    {
        _buffers.push_back(std::move(buffer));
    }

private:
    std::vector<std::unique_ptr<flatbuffers::DetachedBuffer>> _buffers;
};

struct StubSettings
{
    KnobFilter ingestorKnobFilter;
};

struct StubContext
{
    void setExecutionSettings(const StubSettings& /*settings*/) {}

    void setPlan(std::unique_ptr<hipdnn_plugin_sdk::IPlan<StubHandle>> plan)
    {
        _plan = std::move(plan);
    }

    bool hasPlan() const
    {
        return _plan != nullptr;
    }

private:
    std::unique_ptr<hipdnn_plugin_sdk::IPlan<StubHandle>> _plan;
};

/// A device resolver over StubHandle, for the engine-level tests.
class StubDeviceResolver : public IDeviceResolver<StubHandle>
{
public:
    DeviceId deviceId(const StubHandle& /*handle*/) const override
    {
        return 0;
    }

    const hipDeviceProp_t& deviceProperties(DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    hipDeviceProp_t _properties = testDeviceProperties();
};

using StubEngine = GenericEngine<StubHandle, StubSettings, StubContext>;

/// The engine-level tests need a state manager over StubHandle, which the shared fixture
/// builds over int, so this mirrors it for the stub handle type.
std::unique_ptr<KernelIngestorStateManager<StubHandle>> makeStubStateManager()
{
    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.name = "test schema";
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};

    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "test pack";
    pack.matcherIds = {GRAPH_MATCHER_ID};
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT")};

    return std::make_unique<KernelIngestorStateManager<StubHandle>>(
        std::move(schema),
        std::vector<MatchDescriptor>{
            {GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, GRAPH_MATCH_SYMBOL}},
        std::vector<DispatchDescriptor>{
            {DISPATCH_ID, "test dispatch", "hipdnn.kernel_ingestor.test.dispatch"}},
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
}

EngineDescriptor makeEngineWithKnobs(std::vector<std::string> knobs)
{
    EngineDescriptor engine;
    engine.id = ENGINE_ID;
    engine.name = "test:engine";
    engine.heuristicId = HEURISTIC_ID;
    engine.metadataSchemaId = SCHEMA_ID;
    engine.knobs = std::move(knobs);
    return engine;
}

TEST(TestIngestorGenericEngine, AcceptsAKnobNamingADeclaredMetadataField)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;

    EXPECT_NO_THROW(
        (StubEngine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver)));
}

TEST(TestIngestorGenericEngine, RejectsAKnobNamingNoMetadataField)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;

    // A knob is only a name: the field it points at supplies the type, the default and
    // the legal values, so a knob matching no field can never be reported or honoured.
    // Left unchecked it is silently dropped, which reads to a caller exactly like a knob
    // the engine chose not to expose.
    EXPECT_THROW(
        (StubEngine(makeEngineWithKnobs({"no_such_field"}), makeStubStateManager(), resolver)),
        std::invalid_argument);
}

// ---------------------------------------------------------------------------
// GenericPlanBuilder: knob filtering (item 6)
// ---------------------------------------------------------------------------

/// A no-op dispatch handler, sufficient to let buildPlan() construct a GenericPlan:
/// these tests assert which kernel selection chose, not how it launches.
class NoopDispatchHandler : public IKernelDispatchHandler<TestHandle>
{
public:
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& /*context*/,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& /*kernel*/) const override
    {
        return std::make_unique<PreparedDispatch>();
    }

    void launch(const TestHandle& /*handle*/,
                const PreparedDispatch& /*prepared*/,
                const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                uint32_t /*numDeviceBuffers*/,
                void* /*workspace*/) const override
    {
    }
};

/// Registers the no-op dispatch handler under this test suite's dispatch symbol for a
/// test's duration.
class ScopedTestDispatch
{
public:
    ScopedTestDispatch()
    {
        DispatchRegistry<TestHandle>::registerSymbol("test.dispatch", &_handler);
    }

    ~ScopedTestDispatch()
    {
        DispatchRegistry<TestHandle>::unregisterSymbol("test.dispatch");
    }

    ScopedTestDispatch(const ScopedTestDispatch&) = delete;
    ScopedTestDispatch& operator=(const ScopedTestDispatch&) = delete;

private:
    NoopDispatchHandler _handler;
};

struct KnobFilterSettings
{
    KnobFilter ingestorKnobFilter;
};

struct KnobFilterContext
{
    void setExecutionSettings(const KnobFilterSettings& /*settings*/) {}

    void setPlan(std::unique_ptr<hipdnn_plugin_sdk::IPlan<TestHandle>> plan)
    {
        _plan = std::move(plan);
    }

    /// Safe without RTTI: GenericPlanBuilder<TestHandle, ...> only ever hands setPlan()
    /// a GenericPlan<TestHandle>, so the concrete type is known at every call site here.
    const GenericPlan<TestHandle>& plan() const
    {
        return static_cast<const GenericPlan<TestHandle>&>(*_plan);
    }

private:
    std::unique_ptr<hipdnn_plugin_sdk::IPlan<TestHandle>> _plan;
};

using TestPlanBuilder = GenericPlanBuilder<TestHandle, KnobFilterSettings, KnobFilterContext>;

/// An IEngineConfig setting `knobName` to `value`, built the same way
/// TestEngineConfigWrapper.cpp builds one: a real flatbuffer, not a mock, so these tests
/// exercise the same parsing path a real caller's setAttribute() eventually produces.
hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper makeIntKnobEngineConfig(
    flatbuffers::FlatBufferBuilder& builder, const std::string& knobName, int64_t value)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    std::vector<flatbuffers::Offset<KnobSetting>> knobSettings;
    knobSettings.push_back(CreateKnobSettingDirect(
        builder, knobName.c_str(), KnobValue::IntValue, CreateIntValue(builder, value).Union()));
    auto knobsVector = builder.CreateVector(knobSettings);
    builder.Finish(CreateEngineConfig(builder, ENGINE_ID.front(), knobsVector));

    return hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper(
        builder.GetBufferPointer(), builder.GetSize());
}

TEST(TestIngestorPlanBuilderKnobFilter, HonorsAnExplicitKnobSettingOverTheHeuristicDefault)
{
    // The heuristic (scoreByBlockSize) ranks 256 first, so this is the assertion that
    // proves the knob is honored rather than merely advertised: leaving it alone would
    // select 256, but setting it must select 64 instead.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedTestDispatch dispatch;
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeIntKnobEngineConfig(fbb, BLOCK_SIZE, 64);

    const TestGraph graph(makeGraphId(30));
    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    KnobFilterContext context;
    builder.buildPlan(0, graph, engineConfig, context);

    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 64);
}

TEST(TestIngestorPlanBuilderKnobFilter, UnsatisfiableKnobValueThrowsInvalidValueNamingItAndTheValue)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedTestDispatch dispatch;
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    // No surviving kernel carries block_size 999: the catalog matches (two FLOAT
    // kernels), but knob filtering excludes both.
    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeIntKnobEngineConfig(fbb, BLOCK_SIZE, 999);

    const TestGraph graph(makeGraphId(31));
    KnobFilterContext context;

    try
    {
        builder.buildPlan(0, graph, engineConfig, context);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(ex.getMessage().find(BLOCK_SIZE), std::string::npos);
        EXPECT_NE(ex.getMessage().find("999"), std::string::npos);
    }
}

TEST(TestIngestorPlanBuilderKnobFilter,
     EmptyCatalogBeforeFilteringStillThrowsInternalErrorWithItsOriginalMessage)
{
    // The pre-existing, unrelated failure: applicability accepted a graph this engine
    // cannot serve at all (matching itself found nothing), independent of any knob.
    // Item 6 must not collapse this into the same status or message as the
    // knob-filtering case above -- a caller needs to tell "your knobs excluded
    // everything" apart from "this engine cannot serve this graph regardless of knobs".
    const ScopedSymbols symbols("test.graph", rejectGraph, "test.kernel", countingFloatKernels);
    const ScopedTestDispatch dispatch;
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    // No knob set: an empty engine config is as close to "no request" as this helper
    // allows, and the point under test is the no-kernels-at-all path, not filtering.
    fbb.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(fbb, ENGINE_ID.front()));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper engineConfig(
        fbb.GetBufferPointer(), fbb.GetSize());

    const TestGraph graph(makeGraphId(32));
    KnobFilterContext context;

    try
    {
        builder.buildPlan(0, graph, engineConfig, context);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
        EXPECT_EQ(ex.getMessage(),
                  "engine '" + engine.name + "' accepted this graph but has no applicable kernel");
    }
}

} // namespace
