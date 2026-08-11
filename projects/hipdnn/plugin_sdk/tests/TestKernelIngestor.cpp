// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
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

bool acceptGraph(const MatchContext& /*context*/)
{
    ++counters().graphCalls;
    return true;
}

bool rejectGraph(const MatchContext& /*context*/)
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
double scoreConstant(const KernelDefinition& /*kernel*/, const MatchContext& /*context*/)
{
    return 1.0;
}

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
            "Test.cpp",
            "TestKernel",
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
        counters().reset();
    }

    ~ScopedSymbols()
    {
        GraphMatcherRegistry::unregisterSymbol(_graphSymbol);
        KernelMatcherRegistry::unregisterSymbol(_kernelSymbol);
    }

    ScopedSymbols(const ScopedSymbols&) = delete;
    ScopedSymbols& operator=(const ScopedSymbols&) = delete;

private:
    std::string _graphSymbol;
    std::string _kernelSymbol;
};

using TestHandle = int;
using StateManager = KernelIngestorStateManager<TestHandle>;

std::unique_ptr<StateManager> makeStateManager(ScoreFn scoreFn = scoreByBlockSize,
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
        std::make_shared<NativeKernelHeuristic>(scoreFn),
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

TEST(TestIngestorHeuristic, RanksHigherScoringKernelsFirst)
{
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    const auto lowId = testId(0x01);
    const auto highId = testId(0x02);
    catalog.entries = {makeDefinition(lowId, 64), makeDefinition(highId, 256)};

    const NativeKernelHeuristic heuristic(scoreByBlockSize);
    const auto ranked = heuristic.rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, highId);
}

TEST(TestIngestorHeuristic, BreaksScoreTiesOnPriority)
{
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    const auto lowPriorityId = testId(0x01);
    const auto highPriorityId = testId(0x02);
    catalog.entries = {makeDefinition(lowPriorityId, 64, 1), makeDefinition(highPriorityId, 64, 5)};

    const NativeKernelHeuristic heuristic(scoreConstant);
    const auto ranked = heuristic.rank(catalog, context);

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().kernelId, highPriorityId);
}

TEST(TestIngestorHeuristic, BreaksRemainingTiesOnKernelIdForStabilityAcrossRuns)
{
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    Catalog catalog;
    // Deliberately inserted out of id order: the result must not depend on load order.
    // Inserted highest-id first: the result must not depend on load order.
    const auto lowerId = testId(0x01);
    const auto higherId = testId(0x02);
    catalog.entries = {makeDefinition(higherId, 64), makeDefinition(lowerId, 64)};

    const NativeKernelHeuristic heuristic(scoreConstant);
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

    const auto definitions = manager->unsortedDefinitions(0, context);

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

    EXPECT_TRUE(manager->unsortedDefinitions(0, context).empty());
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

    manager->unsortedDefinitions(0, context);
    manager->unsortedDefinitions(0, context);
    manager->unsortedDefinitions(0, context);

    EXPECT_EQ(counters().graphCalls, 1);
    // One call per kernel in the pack, on the single uncached pass.
    EXPECT_EQ(counters().kernelCalls, 3);
}

TEST(TestIngestorStateManager, MatchesSeparatelyPerDevice)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(4));
    const auto properties = testDeviceProperties();

    manager->unsortedDefinitions(0, MatchContext{graph, 0, properties});
    manager->unsortedDefinitions(1, MatchContext{graph, 1, properties});

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

    const auto first = manager->unsortedDefinitions(0, context);
    const auto second = manager->unsortedDefinitions(0, context);

    // Correct both times; only the caching is lost.
    EXPECT_EQ(first.size(), 2U);
    EXPECT_EQ(second.size(), 2U);
    EXPECT_EQ(counters().graphCalls, 2);
}

TEST(TestIngestorStateManager, RematchesAfterCacheEviction)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager(scoreByBlockSize, 1);
    const auto properties = testDeviceProperties();
    const TestGraph first(makeGraphId(5));
    const TestGraph second(makeGraphId(6));

    manager->unsortedDefinitions(0, MatchContext{first, 0, properties});
    manager->unsortedDefinitions(0, MatchContext{second, 0, properties});
    manager->unsortedDefinitions(0, MatchContext{first, 0, properties});

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

    const auto sorted = manager->sortedDefinitions(0, context);

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

    manager->unsortedDefinitions(0, context);
    manager->sortedDefinitions(0, context);
    manager->sortedDefinitions(0, context);

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

    const auto values
        = StateManager::knobValues(manager->sortedDefinitions(0, context), BLOCK_SIZE);

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
                               std::make_shared<NativeKernelHeuristic>(scoreByBlockSize));

    const TestGraph graph(makeGraphId(10));
    const auto properties = testDeviceProperties();
    const auto definitions = manager.unsortedDefinitions(0, MatchContext{graph, 0, properties});

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
                              std::make_shared<NativeKernelHeuristic>(scoreByBlockSize)),
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
                              std::make_shared<NativeKernelHeuristic>(scoreByBlockSize)),
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
                              std::make_shared<NativeKernelHeuristic>(scoreByBlockSize)),
                 std::invalid_argument);
}

TEST(TestIngestorStateManager, RejectsAPackNamingAnUnknownDispatchDescriptor)
{
    EXPECT_THROW(StateManager(makeSchema(),
                              makeTestMatchers(),
                              {},
                              {makePack({GRAPH_MATCHER_ID})},
                              std::make_shared<NativeKernelHeuristic>(scoreByBlockSize)),
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
                              std::make_shared<NativeKernelHeuristic>(scoreByBlockSize)),
                 std::invalid_argument);
}

TEST(TestIngestorStateManager, RejectsAMissingHeuristic)
{
    EXPECT_THROW(StateManager(makeSchema(), {}, {}, {}, nullptr), std::invalid_argument);
}

} // namespace
