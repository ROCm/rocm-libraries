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

inline bool bindConflictingTokenValue(const MatchContext& /*context*/, BoundTokens& bound)
{
    bound["test.bound_token"] = BOUND_TOKEN_VALUE + 1;
    return true;
}

inline bool bindAgreeingTokenValue(const MatchContext& /*context*/, BoundTokens& bound)
{
    bound["test.bound_token"] = BOUND_TOKEN_VALUE;
    return true;
}

inline bool rejectEveryKernel(const MatchContext& /*context*/, const KernelDefinition& /*kernel*/)
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
    const ScopedSymbols symbols("test.graph", rejectGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(2));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    EXPECT_TRUE(manager->unsortedDefinitions(context).empty());
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
    EXPECT_EQ(counters().kernelCalls, 3);
}

TEST(TestKernelIngestorStateManager, EvaluatesASharedGraphMatcherOncePerGraphNotOncePerPack)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    constexpr const char* UNSHARED_FAIL_SYMBOL = "test.cross_pack_binding_reuse.fail";
    GraphMatcherRegistry::registerSymbol(UNSHARED_FAIL_SYMBOL, rejectGraph);
    const auto unsharedFailMatcherId = testId(0x84);

    const KernelDescriptorPack first = makePack({GRAPH_MATCHER_ID, unsharedFailMatcherId});
    KernelDescriptorPack second = makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID});
    second.id = testId(0x80);
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

    EXPECT_EQ(counters().graphCalls, 2);
    ASSERT_EQ(catalog.entries.size(), 1U);
    EXPECT_EQ(catalog.entries.front().packId, second.id);
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

/// Per-arch shards ship the same logical kernel built for different targets, so their
/// completed tuples are identical by construction. Tuple uniqueness is per
/// overlapping-arch group, not per engine: no device sees both, so neither is ambiguous.
/// Asserted per device rather than as "construction succeeded", which cannot tell an
/// admitted pair from a silently deduplicated one.
TEST(TestKernelIngestorStateManager, AdmitsTwoPacksSharingATupleUnderDisjointArch)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    auto first = makePack({GRAPH_MATCHER_ID}, {"gfx90a"});
    first.kernels = {makeKernel(testId(0x90), "kernel_gfx90a", 64, "FLOAT")};
    auto second = makePack({GRAPH_MATCHER_ID}, {"gfx942"});
    second.id = testId(0x91);
    second.kernels = {makeKernel(testId(0x92), "kernel_gfx942", 64, "FLOAT")};

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {first, second},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

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

/// An arch-independent pack claims every device, so it overlaps a per-arch one and the
/// two cannot share a tuple: on a gfx942 device both would apply and the catalog key
/// would name two kernels. This is the empty-list arm of archOverlaps, which the disjoint
/// case above never reaches.
TEST(TestKernelIngestorStateManager, RejectsATupleSharedByAnArchIndependentAndAPerArchPack)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    auto anywhere = makePack({GRAPH_MATCHER_ID});
    anywhere.kernels = {makeKernel(testId(0x93), "kernel_anywhere", 64, "FLOAT")};
    auto pinned = makePack({GRAPH_MATCHER_ID}, {"gfx942"});
    pinned.id = testId(0x94);
    pinned.kernels = {makeKernel(testId(0x95), "kernel_gfx942", 64, "FLOAT")};

    EXPECT_THROW(StateManager(makeSchema(),
                              makeTestMatchers(),
                              makeTestDispatches(),
                              {anywhere, pinned},
                              std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL)),
                 std::invalid_argument);
}

TEST(TestKernelIngestorStateManager, PrunedPackBindingsAreNotVisibleToSurvivingPackKernels)
{
    // Pruned pack's matcher binds a token before a second fails and prunes it; the
    // surviving pack lists neither matcher and must not see the binding.
    constexpr const char* PASS_NO_BIND_SYMBOL = "test.pruned_bound_isolation.pass_no_bind";
    constexpr const char* LEAK_THEN_PRUNE_SYMBOL = "test.pruned_bound_isolation.leak";
    constexpr const char* ALWAYS_FAILS_SYMBOL = "test.pruned_bound_isolation.fail";

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

    ASSERT_EQ(catalog.entries.size(), 1U);
    EXPECT_EQ(catalog.entries.front().packId, survivingPack.id);

    EXPECT_EQ(catalog.bound.count("test.bound_token"), 0U);

    GraphMatcherRegistry::unregisterSymbol(PASS_NO_BIND_SYMBOL);
    GraphMatcherRegistry::unregisterSymbol(LEAK_THEN_PRUNE_SYMBOL);
    GraphMatcherRegistry::unregisterSymbol(ALWAYS_FAILS_SYMBOL);
}

TEST(TestKernelIngestorStateManager, TwoMatchersInOnePackBindingOneTokenDifferentlyThrows)
{
    constexpr const char* CONFLICTING_SYMBOL = "test.bound_conflict.within_pack";
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    GraphMatcherRegistry::registerSymbol(CONFLICTING_SYMBOL, &bindConflictingTokenValue);

    const auto conflictingMatcherId = testId(0xC0);

    const KernelDescriptorPack pack = makePack({GRAPH_MATCHER_ID, conflictingMatcherId});

    const StateManager manager(makeSchema(),
                               {{GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, "test.graph"},
                                {conflictingMatcherId,
                                 "binds a conflicting value",
                                 MatchScope::GRAPH,
                                 CONFLICTING_SYMBOL}},
                               makeTestDispatches(),
                               {pack},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const TestGraph graph(makeGraphId(0x53));
    const auto properties = testDeviceProperties();

    try
    {
        manager.unsortedCatalog(MatchContext{graph, 0, properties});
        ADD_FAILURE() << "expected a within-pack token conflict to be reported";
    }
    catch(const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("binds a conflicting value"), std::string::npos);
        EXPECT_NE(message.find("test.bound_token"), std::string::npos);
    }

    GraphMatcherRegistry::unregisterSymbol(CONFLICTING_SYMBOL);
}

TEST(TestKernelIngestorStateManager, TwoPacksBindingOneTokenToDifferentValuesThrows)
{
    constexpr const char* CONFLICTING_SYMBOL = "test.bound_conflict.conflicting";
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
        const std::string message = error.what();
        EXPECT_NE(message.find("the conflicting pack"), std::string::npos);
        EXPECT_NE(message.find(toString(second.id)), std::string::npos);
        EXPECT_NE(message.find("test.bound_token"), std::string::npos);
    }

    GraphMatcherRegistry::unregisterSymbol(CONFLICTING_SYMBOL);
}

TEST(TestKernelIngestorStateManager, TwoPacksBindingOneTokenToTheSameValueMergeCleanly)
{
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

    EXPECT_EQ(counters().graphCalls, 2);
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
    EXPECT_EQ(counters().graphCalls, 2);
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
    EXPECT_EQ(counters().graphCalls, 1);
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

    EXPECT_EQ(counters().graphCalls, 2);
    EXPECT_EQ(firstDefinitions.size(), 2U);
    EXPECT_EQ(secondDefinitions.size(), 2U);
}

TEST(TestKernelIngestorStateManager, CarriesWhatMatchingBoundThroughToDispatch)
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
    const auto afterMatching = counters().graphCalls;

    const auto secondRead = manager->unsortedCatalog(context).bound;
    const auto thirdRead = manager->sortedCatalog(context).bound;

    EXPECT_EQ(counters().graphCalls, afterMatching);
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

    EXPECT_EQ(counters().graphCalls, 3);
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
                                              + " admitted no kernel of 3 at a kernel-scoped "
                                                "matcher"))
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

    try
    {
        const StateManager manager(
            makeSchema(),
            makeTestMatchers(),
            std::vector<DispatchDescriptor>{
                {DISPATCH_ID, "misspelled dispatch", "test.dispatch.not_registered"}},
            std::vector<KernelDescriptorPack>{makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID})},
            std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
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
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const TestGraph graph(makeGraphId(10));
    const auto properties = testDeviceProperties();
    const auto definitions = manager.unsortedDefinitions(MatchContext{graph, 0, properties});

    ASSERT_EQ(definitions.size(), 1U);
    EXPECT_EQ(definitions.front().getIntMetadata(BLOCK_SIZE), 64);
    EXPECT_EQ(definitions.front().getStringMetadata(DTYPE), "FLOAT");
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
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
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
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
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
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
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
                KernelDescriptorPack pack = makePack({GRAPH_MATCHER_ID});
                pack.kernels.push_back(makeKernel(testId(0x73), "kernel_duplicate", 64, "FLOAT"));
                return std::make_unique<StateManager>(
                    makeSchema(),
                    makeTestMatchers(),
                    makeTestDispatches(),
                    std::vector<KernelDescriptorPack>{pack},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
            }},
        // A feature suffix still names the same base target, so one device satisfies both
        // lists and the tuple really is ambiguous. Plain string equality would let this
        // construct.
        StateManagerConstructionThrowCase{
            "RejectsTwoPacksSharingATupleUnderOverlappingArch",
            "duplicates the metadata tuple",
            [] {
                auto first = makePack({GRAPH_MATCHER_ID}, {"gfx942"});
                first.kernels = {makeKernel(testId(0x93), "kernel_bare", 64, "FLOAT")};
                auto second = makePack({GRAPH_MATCHER_ID}, {"gfx942:sramecc+"});
                second.id = testId(0x94);
                second.kernels = {makeKernel(testId(0x95), "kernel_suffixed", 64, "FLOAT")};
                return std::make_unique<StateManager>(
                    makeSchema(),
                    makeTestMatchers(),
                    makeTestDispatches(),
                    std::vector<KernelDescriptorPack>{first, second},
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
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
                    std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));
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
