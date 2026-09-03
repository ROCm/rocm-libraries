// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <filesystem>
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

#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

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

inline bool rejectSecondCriterion(const MatchContext& /*context*/, const BoundTokens& /*bound*/)
{
    ++counters().graphCalls;
    return false;
}

inline bool rejectEveryKernel(const MatchContext& /*context*/,
                              const BoundTokens& /*bound*/,
                              const KernelDefinition& /*kernel*/)
{
    ++counters().kernelCalls;
    return false;
}

TEST(TestKernelIngestorStateManager, KernelLevelMatcherPrunesTheCatalog)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto definitions = manager->unsortedDefinitions(context);

    ASSERT_EQ(definitions.size(), 2U);
    for(const auto& definition : definitions)
    {
        EXPECT_EQ(definition.getStringMetadata(DTYPE), "FLOAT");
    }
}

TEST(TestKernelIngestorStateManager, GraphLevelMatcherFailurePrunesTheWholePack)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &rejectCriterion);

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID})},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               "test.graph");

    const TestGraph graph(makeGraphId(2));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    EXPECT_TRUE(manager.unsortedDefinitions(context).empty());
    EXPECT_EQ(counters().graphCalls, 1);
    EXPECT_EQ(counters().kernelCalls, 0);
}

TEST(TestKernelIngestorStateManager, MatchesOncePerGraphAndDevice)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID})},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               "test.graph");

    const TestGraph graph(makeGraphId(3));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    manager.unsortedDefinitions(context);
    manager.unsortedDefinitions(context);
    manager.unsortedDefinitions(context);

    // The engine's graph match and its criterion each run once; the catalog cache serves
    // the other two queries.
    EXPECT_EQ(counters().graphMatchCalls, 1);
    EXPECT_EQ(counters().graphCalls, 1);
    EXPECT_EQ(counters().kernelCalls, 3);
}

TEST(TestKernelIngestorStateManager, EvaluatesASharedGraphMatcherOncePerGraphNotOncePerPack)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto shared = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);
    const auto unsharedFail
        = scopedGraphMatcher("test.cross_pack_criterion_reuse.fail", &rejectSecondCriterion);
    const auto unsharedFailMatcherId = testId(0x84);

    const KernelDescriptorPack first = makePack({GRAPH_MATCHER_ID, unsharedFailMatcherId});
    KernelDescriptorPack second = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    second.id = testId(0x80);
    second.kernels = {makeKernel(testId(0x81), "second_pack_kernel", 512, "FLOAT")};

    const StateManager manager(
        makeSchema(),
        {{GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, "test.graph_criterion"},
         {unsharedFailMatcherId,
          "always fails",
          MatchScope::GRAPH,
          "test.cross_pack_criterion_reuse.fail"},
         {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, "test.kernel"}},
        makeTestDispatches(),
        {first, second},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
        "test.graph");

    const TestGraph graph(makeGraphId(20));
    const auto properties = testDeviceProperties();
    const auto catalog = manager.unsortedCatalog(MatchContext{graph, 0, properties});

    // The shared criterion is memoized by descriptor id, so two packs listing it cost one
    // evaluation; the unshared one adds the second.
    EXPECT_EQ(counters().graphCalls, 2);
    ASSERT_EQ(catalog.entries.size(), 1U);
    EXPECT_EQ(catalog.entries.front().packId, second.id);
}

TEST(TestKernelIngestorStateManager, ASharedGraphMatcherFailurePrunesEveryPackListingIt)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &rejectCriterion);

    auto first = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    auto second = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    second.id = testId(0x82);
    second.kernels = {makeKernel(testId(0x83), "second_pack_kernel", 512, "FLOAT")};

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {first, second},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               "test.graph");

    const TestGraph graph(makeGraphId(21));
    const auto properties = testDeviceProperties();

    EXPECT_TRUE(manager.unsortedDefinitions(MatchContext{graph, 0, properties}).empty());
    // One failing evaluation disqualifies both packs, and no per-kernel work runs at all.
    EXPECT_EQ(counters().graphCalls, 1);
    EXPECT_EQ(counters().kernelCalls, 0);
}

/// Per-arch shards ship the same logical kernel built for different targets, so their
/// completed tuples are identical by construction. Tuple uniqueness is per
/// overlapping-arch group, not per engine: no device sees both, so neither is ambiguous.
/// Asserted per device rather than as "construction succeeded", which cannot tell an
/// admitted pair from a silently deduplicated one.
TEST(TestKernelIngestorStateManager, AdmitsTwoPacksSharingATupleUnderDisjointArch)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    auto first = makePack({GRAPH_MATCHER_ID}, {"gfx90a"});
    first.kernels = {makeKernel(testId(0x90), "kernel_gfx90a", 64, "FLOAT")};
    auto second = makePack({GRAPH_MATCHER_ID}, {"gfx942"});
    second.id = testId(0x91);
    second.kernels = {makeKernel(testId(0x92), "kernel_gfx942", 64, "FLOAT")};

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {first, second},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               "test.graph");

    const TestGraph graph(makeGraphId(22));
    // Distinct device ids as well as arch strings: the catalog cache is keyed by
    // (graph, device id), so two devices sharing an id would answer from one catalog.
    const auto definitionsFor = [&](int deviceId, const char* deviceArch) {
        auto properties = testDeviceProperties();
        properties.gcnArchName = deviceArch;
        return manager.unsortedDefinitions(MatchContext{graph, deviceId, properties});
    };

    const auto onGfx90a = definitionsFor(0, "gfx90a:sramecc+:xnack-");
    ASSERT_EQ(onGfx90a.size(), 1u);
    EXPECT_EQ(onGfx90a.front().kernelId, testId(0x90));

    const auto onGfx942 = definitionsFor(1, "gfx942:sramecc+");
    ASSERT_EQ(onGfx942.size(), 1u);
    EXPECT_EQ(onGfx942.front().kernelId, testId(0x92));
}

/// The same thing one level down, and the reason a kernel carries an arch at all: two
/// implementations of ONE problem shape in ONE pack, each built for what it can run on.
/// Their completed tuples are identical -- same block_size, same dtype -- so claiming by
/// the pack would throw, and both would have to be split into separate packs to coexist.
TEST(TestKernelIngestorStateManager, AdmitsTwoKernelsOfOnePackSharingATupleUnderDisjointArch)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    auto pack = makePack({GRAPH_MATCHER_ID}, {"gfx90a", "gfx942"});
    auto portable = makeKernel(testId(0x90), "portable", 64, "FLOAT");
    portable.arch = {"gfx90a"};
    auto accelerated = makeKernel(testId(0x92), "mfma", 64, "FLOAT");
    accelerated.arch = {"gfx942"};
    pack.kernels = {portable, accelerated};

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {pack},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               "test.graph");

    const TestGraph graph(makeGraphId(23));
    const auto definitionsFor = [&](int deviceId, const char* deviceArch) {
        auto properties = testDeviceProperties();
        properties.gcnArchName = deviceArch;
        return manager.unsortedDefinitions(MatchContext{graph, deviceId, properties});
    };

    // The pack passes the arch gate on both devices, so only the per-kernel gate can be
    // what separates these -- a pack-level filter alone would hand both to both.
    const auto onGfx90a = definitionsFor(0, "gfx90a:sramecc+:xnack-");
    ASSERT_EQ(onGfx90a.size(), 1u);
    EXPECT_EQ(onGfx90a.front().kernelId, testId(0x90));

    const auto onGfx942 = definitionsFor(1, "gfx942:sramecc+");
    ASSERT_EQ(onGfx942.size(), 1u);
    EXPECT_EQ(onGfx942.front().kernelId, testId(0x92));
}

/// Narrowing does not buy an escape from uniqueness: two kernels a gfx942 device would
/// both see still name one catalog key, whether they narrowed to reach it or inherited
/// the pack. Overlapping is enough -- the two lists need not be equal.
TEST(TestKernelIngestorStateManager, RejectsTwoKernelsOfOnePackNarrowedToOverlappingArch)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    auto pack = makePack({GRAPH_MATCHER_ID}, {"gfx942", "gfx950"});
    auto broad = makeKernel(testId(0x90), "broad", 64, "FLOAT");
    broad.arch = {"gfx942", "gfx950"};
    auto narrow = makeKernel(testId(0x92), "narrow", 64, "FLOAT");
    narrow.arch = {"gfx942"};
    pack.kernels = {broad, narrow};

    EXPECT_THROW(StateManager(makeSchema(),
                              makeTestMatchers(),
                              makeTestDispatches(),
                              {pack},
                              std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                              "test.graph"),
                 std::invalid_argument);
}

/// A kernel that declares nothing runs wherever its pack does. Pinned because the gate
/// reads the kernel's list, so an empty one must stay the widest value rather than
/// becoming a claim on nothing and filtering every unstamped kernel out.
TEST(TestKernelIngestorStateManager, OffersAnUnstampedKernelEverywhereItsPackReaches)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    auto pack = makePack({GRAPH_MATCHER_ID}, {"gfx90a", "gfx942"});
    pack.kernels = {makeKernel(testId(0x90), "unstamped", 64, "FLOAT")};

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {pack},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               "test.graph");

    const TestGraph graph(makeGraphId(24));
    int deviceId = 0;
    for(const auto* deviceArch : {"gfx90a:sramecc+:xnack-", "gfx942:sramecc+"})
    {
        // A fresh device id per arch: the catalog is cached by (graph, device id), so
        // reusing one would answer the second arch from the first one's catalog.
        auto properties = testDeviceProperties();
        properties.gcnArchName = deviceArch;
        const auto definitions
            = manager.unsortedDefinitions(MatchContext{graph, deviceId++, properties});
        EXPECT_EQ(definitions.size(), 1u) << "unstamped kernel missing on " << deviceArch;
    }
}

/// An arch-independent pack claims every device, so it overlaps a per-arch one and the
/// two cannot share a tuple: on a gfx942 device both would apply and the catalog key
/// would name two kernels. This is the empty-list arm of archOverlaps, which the disjoint
/// case above never reaches.
TEST(TestKernelIngestorStateManager, RejectsATupleSharedByAnArchIndependentAndAPerArchPack)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    auto anywhere = makePack({GRAPH_MATCHER_ID});
    anywhere.kernels = {makeKernel(testId(0x93), "kernel_anywhere", 64, "FLOAT")};
    auto pinned = makePack({GRAPH_MATCHER_ID}, {"gfx942"});
    pinned.id = testId(0x94);
    pinned.kernels = {makeKernel(testId(0x95), "kernel_gfx942", 64, "FLOAT")};

    EXPECT_THROW(StateManager(makeSchema(),
                              makeTestMatchers(),
                              makeTestDispatches(),
                              {anywhere, pinned},
                              std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                              "test.graph"),
                 std::invalid_argument);
}

TEST(TestKernelIngestorStateManager, APackIsPrunedWhenAnyOfItsCriteriaFails)
{
    // A pack passes only if every criterion it lists passes: the first admits it, the
    // second rejects, and the pack does not reach its kernels.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto passes = scopedGraphMatcher("test.conjunction.pass", &acceptCriterion);
    const auto fails = scopedGraphMatcher("test.conjunction.fail", &rejectSecondCriterion);

    const auto passMatcherId = testId(0xA4);
    const auto failMatcherId = testId(0xA6);

    KernelDescriptorPack survivingPack;
    survivingPack.id = testId(0xA0);
    survivingPack.name = "surviving pack";
    survivingPack.matcherIds = {passMatcherId};
    survivingPack.engineId = ENGINE_ID;
    survivingPack.dispatchId = DISPATCH_ID;
    survivingPack.kernels = {makeKernel(testId(0xA1), "surviving_kernel", 64, "FLOAT")};

    KernelDescriptorPack prunedPack;
    prunedPack.id = testId(0xA2);
    prunedPack.name = "pruned pack";
    prunedPack.matcherIds = {passMatcherId, failMatcherId};
    prunedPack.engineId = ENGINE_ID;
    prunedPack.dispatchId = DISPATCH_ID;
    prunedPack.kernels = {makeKernel(testId(0xA3), "pruned_kernel", 128, "FLOAT")};

    const StateManager manager(
        makeSchema(),
        {{passMatcherId, "passes", MatchScope::GRAPH, "test.conjunction.pass"},
         {failMatcherId, "always fails", MatchScope::GRAPH, "test.conjunction.fail"}},
        makeTestDispatches(),
        {survivingPack, prunedPack},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
        "test.graph");

    const TestGraph graph(makeGraphId(50));
    const auto properties = testDeviceProperties();
    const auto catalog = manager.unsortedCatalog(MatchContext{graph, 0, properties});

    ASSERT_EQ(catalog.entries.size(), 1U);
    EXPECT_EQ(catalog.entries.front().packId, survivingPack.id);
}

TEST(TestKernelIngestorStateManager, EveryPackOfOneEngineSharesWhatTheGraphMatchBound)
{
    // Two packs, one engine: the engine's graph match runs once and both packs' kernels
    // land in a catalog carrying its tokens.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    auto first = makePack({});
    KernelDescriptorPack second = makePack({});
    second.id = testId(0xB4);
    second.kernels = {makeKernel(testId(0xB5), "second_pack_kernel", 512, "FLOAT")};

    const StateManager manager(
        makeSchema(),
        std::vector<MatchDescriptor>{
            {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, "test.kernel"}},
        makeTestDispatches(),
        {first, second},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
        "test.graph");

    const TestGraph graph(makeGraphId(22));
    const auto properties = testDeviceProperties();
    const auto catalog = manager.unsortedCatalog(MatchContext{graph, 0, properties});

    EXPECT_EQ(counters().graphMatchCalls, 1);
    EXPECT_EQ(catalog.entries.size(), 4U);
    EXPECT_EQ(tryGetBoundInt(catalog.bound, "test.bound_token"), BOUND_TOKEN_VALUE);
}

TEST(TestKernelIngestorStateManager, AGraphTheMatchDeclinesEmptiesTheCatalogWithoutCriteria)
{
    // nullopt from the engine's graph match is the whole answer: no criterion and no
    // kernel matcher runs, because there is nothing left to narrow.
    const ScopedSymbols symbols("test.graph", rejectGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID})},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               "test.graph");

    const TestGraph graph(makeGraphId(23));
    const auto properties = testDeviceProperties();
    const auto catalog = manager.unsortedCatalog(MatchContext{graph, 0, properties});

    EXPECT_TRUE(catalog.entries.empty());
    EXPECT_TRUE(catalog.bound.empty());
    EXPECT_EQ(counters().graphMatchCalls, 1);
    EXPECT_EQ(counters().graphCalls, 0);
    EXPECT_EQ(counters().kernelCalls, 0);
}

TEST(TestKernelIngestorStateManager, AnEngineWithNoGraphMatchStillMatchesOnItsCriteria)
{
    // A graph_match is optional: an engine that declares none is admitted or declined by
    // its criteria alone, and presents an empty token map to dispatch.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID})},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               std::string{});

    const TestGraph graph(makeGraphId(24));
    const auto properties = testDeviceProperties();
    const auto catalog = manager.unsortedCatalog(MatchContext{graph, 0, properties});

    EXPECT_EQ(catalog.entries.size(), 2U);
    EXPECT_TRUE(catalog.bound.empty());
    EXPECT_EQ(counters().graphMatchCalls, 0);
    EXPECT_EQ(counters().graphCalls, 1);
}

TEST(TestKernelIngestorStateManager, RefusesToConstructAgainstAnUnregisteredGraphMatchSymbol)
{
    // Same eager-resolution contract the matcher and dispatch symbols get: a misspelled
    // graph_match excludes the engine at construction, not at the first query. And the
    // failure names the engine, as the matcher and dispatch failures name their own
    // descriptors -- graph_match has no descriptor of its own, so the caller-supplied
    // description is the only thing that can point at the file to fix.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    try
    {
        const StateManager manager(
            makeSchema(),
            std::vector<MatchDescriptor>{
                {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, "test.kernel"}},
            makeTestDispatches(),
            {makePack({KERNEL_MATCHER_ID})},
            std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
            "test.graph.not_registered",
            "engine 'test:misspelled_graph_match'");
        FAIL() << "expected an unresolved-symbol failure";
    }
    catch(const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("test.graph.not_registered"), std::string::npos) << message;
        EXPECT_NE(message.find("test:misspelled_graph_match"), std::string::npos) << message;
    }
}

TEST(TestKernelIngestorStateManager, MatchesSeparatelyPerDevice)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(4));
    const auto properties = testDeviceProperties();

    manager->unsortedDefinitions(MatchContext{graph, 0, properties});
    manager->unsortedDefinitions(MatchContext{graph, 1, properties});

    EXPECT_EQ(counters().graphMatchCalls, 2);
}

TEST(TestKernelIngestorStateManager, NoDeviceYieldsAnEmptyCatalogEvenWhenMatchersWouldAccept)
{
    const ScopedSymbols symbols("test.graph", acceptAnyGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(0x61));
    const auto properties = testDeviceProperties();

    // Positive half first: same manager and graph, a real device, matchers that accept.
    ASSERT_FALSE(manager->unsortedCatalog(MatchContext{graph, 0, properties}).entries.empty());

    // Only the device id changes; the catalog must go empty before any matcher runs.
    EXPECT_TRUE(
        manager->unsortedCatalog(MatchContext{graph, NO_DEVICE, properties}).entries.empty());
}

TEST(TestKernelIngestorStateManager, RematchesEveryCallWhenTheGraphHasNoIdentity)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph;
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto first = manager->unsortedDefinitions(context);
    const auto second = manager->unsortedDefinitions(context);

    EXPECT_EQ(first.size(), 2U);
    EXPECT_EQ(second.size(), 2U);
    EXPECT_EQ(counters().graphMatchCalls, 2);
}

TEST(TestKernelIngestorStateManager, ServesACachedRankingWithoutRematching)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(0x5D));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    ASSERT_TRUE(manager->sortedCatalog(context).isSorted);

    static_cast<void>(manager->unsortedCatalog(context));

    EXPECT_TRUE(manager->sortedCatalog(context).isSorted);
    EXPECT_EQ(counters().graphMatchCalls, 1);
}

TEST(TestKernelIngestorStateManager, DistinctGraphsCarryingANilUuidDoNotShareACatalogEntry)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph first(makeNilGraphId());
    const TestGraph second(makeNilGraphId());
    const auto properties = testDeviceProperties();

    const auto firstDefinitions = manager->unsortedDefinitions(MatchContext{first, 0, properties});
    const auto secondDefinitions
        = manager->unsortedDefinitions(MatchContext{second, 0, properties});

    EXPECT_EQ(counters().graphMatchCalls, 2);
    EXPECT_EQ(firstDefinitions.size(), 2U);
    EXPECT_EQ(secondDefinitions.size(), 2U);
}

TEST(TestKernelIngestorStateManager, CarriesWhatTheGraphMatchBoundThroughToDispatch)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto properties = testDeviceProperties();
    const TestGraph graph;
    const MatchContext context{graph, 0, properties};

    const auto bound = manager->unsortedCatalog(context).bound;

    ASSERT_EQ(bound.count("test.bound_token"), 1U);
    EXPECT_EQ(tryGetBoundInt(bound, "test.bound_token"), BOUND_TOKEN_VALUE);
}

TEST(TestKernelIngestorStateManager, ReadingBoundStateAfterMatchingDoesNotRematch)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto properties = testDeviceProperties();
    const TestGraph graph(makeGraphId(11));
    const MatchContext context{graph, 0, properties};

    static_cast<void>(manager->unsortedCatalog(context));
    const auto afterMatching = counters().graphMatchCalls;

    const auto secondRead = manager->unsortedCatalog(context).bound;
    const auto thirdRead = manager->sortedCatalog(context).bound;

    EXPECT_EQ(counters().graphMatchCalls, afterMatching);
    EXPECT_EQ(tryGetBoundInt(secondRead, "test.bound_token"), BOUND_TOKEN_VALUE);
    EXPECT_EQ(tryGetBoundInt(thirdRead, "test.bound_token"), BOUND_TOKEN_VALUE);
}

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

    EXPECT_EQ(counters().graphMatchCalls, 3);
}

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

    EXPECT_EQ(counters().graphMatchCalls, 1);
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

    ASSERT_EQ(values.size(), 2U);
    EXPECT_EQ(std::get<int64_t>(values[0]), 256);
    EXPECT_EQ(std::get<int64_t>(values[1]), 64);
}

// A pack whose kernel matchers reject everything contributes nothing, so it must read
// as excluded rather than as a participant that happened to score zero. The two declines
// ahead of it, arch and graph-scoped, both say plainly that the pack is out.
TEST(TestKernelIngestorStateManager, APackAdmittingNoKernelSaysSoRatherThanReportingZero)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", rejectEveryKernel);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(0x5E));
    const auto properties = testDeviceProperties();

    EXPECT_TRUE(manager->unsortedDefinitions(MatchContext{graph, 0, properties}).empty());

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO,
                                          toString(PACK_ID)
                                              + " admitted no kernel of 3 at the arch gate or a "
                                                "kernel-scoped matcher"))
        << "a pack that contributed nothing must not read as one that scored zero:\n"
        << recorder.getRecordedLogsAsString();

    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "admitted 0 of"))
        << "the zero-admit case must not fall through to the contributor message:\n"
        << recorder.getRecordedLogsAsString();
}

TEST(TestKernelIngestorStateManager, APackAdmittingSomeKernelsStillReportsTheCount)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(0x5F));
    const auto properties = testDeviceProperties();

    EXPECT_EQ(manager->unsortedDefinitions(MatchContext{graph, 0, properties}).size(), 2U);

    EXPECT_TRUE(recorder.hasLogContaining(
        HIPDNN_SEV_INFO,
        toString(PACK_ID) + " admitted 2 of 3 kernel(s) after kernel-scoped matching"))
        << recorder.getRecordedLogsAsString();
}

TEST(TestKernelIngestorStateManager, RefusesToConstructAgainstAnUnregisteredDispatchSymbol)
{
    // Descriptor and native halves agree on dispatch symbols by string with no
    // compile-time check; eager resolution must catch a misspelled one.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    try
    {
        const StateManager manager(
            makeSchema(),
            makeTestMatchers(),
            std::vector<DispatchDescriptor>{
                {DISPATCH_ID, "misspelled dispatch", "test.dispatch.not_registered"}},
            std::vector<KernelDescriptorPack>{makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID})},
            std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
            std::string{});
        FAIL() << "expected an unresolved-symbol failure";
    }
    catch(const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("test.dispatch.not_registered"), std::string::npos);
        EXPECT_NE(message.find("misspelled dispatch"), std::string::npos);
    }
}

TEST(TestKernelIngestorStateManager, GetDispatchDetailsThrowsOnADanglingDispatchId)
{
    // Built directly since validation cannot see a definition never in a pack. The
    // dangling id must not be one the manager registered, or this fails at resolve
    // before reaching the branch under test.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    auto kernel = makeDefinition(testId(0x01), 64);
    kernel.dispatchId = testId(0xDD);

    try
    {
        manager->getDispatchDetails(kernel);
        FAIL() << "expected an unknown-dispatch-descriptor failure";
    }
    catch(const std::runtime_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("names unknown dispatch descriptor"),
                  std::string::npos)
            << "threw for the wrong reason: " << error.what();
    }
}

TEST(TestKernelIngestorStateManager, CompletesAnOmittedFieldFromItsSchemaDefault)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    KernelDescriptorPack pack = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    KernelDescriptor sparse;
    sparse.id = testId(0x70);
    sparse.name = "kernel_defaults";
    sparse.metadata = {{DTYPE, MetadataValue{std::string{"FLOAT"}}}};
    pack.kernels = {sparse};

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {pack},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               std::string{});

    const TestGraph graph(makeGraphId(10));
    const auto properties = testDeviceProperties();
    const auto definitions = manager.unsortedDefinitions(MatchContext{graph, 0, properties});

    ASSERT_EQ(definitions.size(), 1U);
    EXPECT_EQ(definitions.front().getIntMetadata(BLOCK_SIZE), 64);
    EXPECT_EQ(definitions.front().getStringMetadata(DTYPE), "FLOAT");
}

// The source-kind gate: KPACK indexes, the two kinds with no adapter are dropped, and a
// drop costs only its own kernel.

/// A hand-built pack reaches only this gate -- the loader's own is upstream of it -- so
/// this is where a KPACK kernel built in memory proves it indexes like an embedded one.
TEST(TestKernelIngestorStateManager, IndexesAKpackKernel)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    KernelDescriptorPack pack = makePack({KERNEL_MATCHER_ID});
    KernelDescriptor kernel = makeKernel(testId(0x76), "kernel_kpack", 64, "FLOAT");
    kernel.source = makeKpackSource();
    pack.kernels = {kernel};

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {pack},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               std::string{});

    const TestGraph graph(makeGraphId(0x76));
    const auto properties = testDeviceProperties();
    const auto definitions = manager.unsortedDefinitions(MatchContext{graph, 0, properties});

    ASSERT_EQ(definitions.size(), 1U);
    const auto& source = definitions.front().source;
    EXPECT_EQ(source.kind, KernelSourceKind::KPACK);
    EXPECT_EQ(source.library, "kpack/hip_kernel_provider_gfx942.kpack");
    EXPECT_EQ(source.tocKey, "test-toc-key");
    EXPECT_EQ(source.symbol, "TestKernel");
    EXPECT_EQ(source.sha256, std::string(64, 'a'));
    EXPECT_EQ(definitions.front().name, "kernel_kpack");
}

/// Admitting KPACK must not admit the two kinds nothing can dispatch. They are dropped
/// rather than thrown on, so the evidence is their absence from the catalog plus an ERROR
/// naming each.
TEST(TestKernelIngestorStateManager, RejectsAnUnadaptedSourceKind)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    KernelDescriptorPack pack = makePack({KERNEL_MATCHER_ID});
    KernelDescriptor hsaco = makeKernel(testId(0x77), "kernel_hsaco", 64, "FLOAT");
    hsaco.source.kind = KernelSourceKind::HSACO_FILE;
    KernelDescriptor rocke = makeKernel(testId(0x78), "kernel_rocke", 256, "FLOAT");
    rocke.source.kind = KernelSourceKind::ROCKE_BUILDER;
    pack.kernels = {hsaco, rocke};

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {pack},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               std::string{});

    const TestGraph graph(makeGraphId(0x77));
    const auto properties = testDeviceProperties();

    EXPECT_TRUE(manager.unsortedDefinitions(MatchContext{graph, 0, properties}).empty());

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "kernel 'kernel_hsaco'"))
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "kernel 'kernel_rocke'"))
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR,
                                          "declares a source kind this build has no "
                                          "adapter for"))
        << recorder.getRecordedLogsAsString();
}

/// Failure granularity is the kernel, not the pack: an artifact this build cannot load
/// must not cost the siblings it ships beside.
TEST(TestKernelIngestorStateManager, DropsOnlyTheUnadaptedKernelAndKeepsItsPack)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    KernelDescriptorPack pack = makePack({KERNEL_MATCHER_ID});
    KernelDescriptor unadapted = makeKernel(testId(0x79), "kernel_hsaco", 64, "FLOAT");
    unadapted.source.kind = KernelSourceKind::HSACO_FILE;
    pack.kernels = {unadapted, makeKernel(testId(0x7A), "kernel_sibling", 256, "FLOAT")};

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {pack},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               std::string{});

    const TestGraph graph(makeGraphId(0x79));
    const auto properties = testDeviceProperties();
    const auto definitions = manager.unsortedDefinitions(MatchContext{graph, 0, properties});

    ASSERT_EQ(definitions.size(), 1U);
    EXPECT_EQ(definitions.front().getIntMetadata(BLOCK_SIZE), 256);
    EXPECT_EQ(definitions.front().name, "kernel_sibling");
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "kernel 'kernel_hsaco'"))
        << recorder.getRecordedLogsAsString();
}

/// `source.library` is relative, so a definition is only usable together with the
/// directory of the descriptor that declared it. The state manager carries the anchor
/// through unchanged; the loader is what fills it.
TEST(TestKernelIngestorStateManager, CarriesTheOriginDirectoryIntoTheDefinition)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);
    const std::filesystem::path origin("/descriptors/gfx942");

    KernelDescriptorPack pack = makePack({KERNEL_MATCHER_ID});
    KernelDescriptor kernel = makeKernel(testId(0x7B), "kernel_kpack", 64, "FLOAT");
    kernel.source = makeKpackSource();
    kernel.originDirectory = origin;
    pack.kernels = {kernel};

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {pack},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               std::string{});

    const TestGraph graph(makeGraphId(0x7B));
    const auto properties = testDeviceProperties();
    const auto definitions = manager.unsortedDefinitions(MatchContext{graph, 0, properties});

    ASSERT_EQ(definitions.size(), 1U);
    EXPECT_EQ(definitions.front().originDirectory, origin);
}

struct StateManagerConstructionThrowCase
{
    std::string name;
    std::string expectedMessageSubstring;
    std::function<std::unique_ptr<StateManager>()> construct;
};

class TestKernelIngestorStateManagerConstructionThrows
    : public ::testing::TestWithParam<StateManagerConstructionThrowCase>
{
};

TEST_P(TestKernelIngestorStateManagerConstructionThrows, RejectsAtConstruction)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

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
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                    std::string{});
            }},
        StateManagerConstructionThrowCase{
            "RejectsAKernelSupplyingAFieldTheSchemaDoesNotDeclare",
            "does not declare",
            [] {
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
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                    std::string{});
            }},
        StateManagerConstructionThrowCase{
            "RejectsAKernelSupplyingAFieldOfTheWrongType",
            "a value of the wrong type",
            [] {
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
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                    std::string{});
            }},
        StateManagerConstructionThrowCase{
            "RejectsAPackNamingAnUnknownMatcher",
            "names unknown matcher",
            [] {
                return std::make_unique<StateManager>(
                    makeSchema(),
                    std::vector<MatchDescriptor>{},
                    makeTestDispatches(),
                    std::vector<KernelDescriptorPack>{makePack({testId(0xFF)})},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                    std::string{});
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
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                    std::string{});
            }},
        StateManagerConstructionThrowCase{
            "RejectsTwoKernelsSharingAMetadataTuple",
            "duplicates the metadata tuple",
            [] {
                KernelDescriptorPack pack = makePack({GRAPH_MATCHER_ID});
                pack.kernels.push_back(makeKernel(testId(0x73), "kernel_duplicate", 64, "FLOAT"));
                return std::make_unique<StateManager>(
                    makeSchema(),
                    makeTestMatchers(),
                    makeTestDispatches(),
                    std::vector<KernelDescriptorPack>{pack},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                    std::string{});
            }},
        // Overlapping lists need not be equal: a gfx942 device satisfies both, so the
        // tuple really is ambiguous. Plain string equality would let this construct.
        StateManagerConstructionThrowCase{
            "RejectsTwoPacksSharingATupleUnderOverlappingArch",
            "duplicates the metadata tuple",
            [] {
                auto first = makePack({GRAPH_MATCHER_ID}, {"gfx942"});
                first.kernels = {makeKernel(testId(0x93), "kernel_narrow", 64, "FLOAT")};
                auto second = makePack({GRAPH_MATCHER_ID}, {"gfx942", "gfx950"});
                second.id = testId(0x94);
                second.kernels = {makeKernel(testId(0x95), "kernel_broad", 64, "FLOAT")};
                return std::make_unique<StateManager>(
                    makeSchema(),
                    makeTestMatchers(),
                    makeTestDispatches(),
                    std::vector<KernelDescriptorPack>{first, second},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                    std::string{});
            }},
        StateManagerConstructionThrowCase{
            "RejectsADuplicateMatchDescriptorId",
            "duplicate match descriptor id",
            [] {
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
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                    std::string{});
            }},
        StateManagerConstructionThrowCase{
            "RejectsADuplicateDispatchDescriptorId",
            "duplicate dispatch descriptor id",
            [] {
                std::vector<DispatchDescriptor> dispatches = makeTestDispatches();
                dispatches.push_back(
                    {DISPATCH_ID, "a different dispatch entirely", "test.dispatch"});
                return std::make_unique<StateManager>(
                    makeSchema(),
                    makeTestMatchers(),
                    dispatches,
                    std::vector<KernelDescriptorPack>{makePack({GRAPH_MATCHER_ID})},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                    std::string{});
            }},
        StateManagerConstructionThrowCase{"RejectsAMissingHeuristic",
                                          "requires a heuristic",
                                          [] {
                                              return std::make_unique<StateManager>(
                                                  makeSchema(),
                                                  std::vector<MatchDescriptor>{},
                                                  std::vector<DispatchDescriptor>{},
                                                  std::vector<KernelDescriptorPack>{},
                                                  nullptr,
                                                  std::string{});
                                          }}),
    [](const ::testing::TestParamInfo<StateManagerConstructionThrowCase>& info) {
        return info.param.name;
    });

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
