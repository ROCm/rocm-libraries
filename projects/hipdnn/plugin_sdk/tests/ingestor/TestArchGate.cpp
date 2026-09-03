// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestArchGate.cpp
 * @brief The KDP's supported-arch list: what it admits, and that it prunes before
 *        matching rather than after (a pack reaching an unsupported device would
 *        otherwise match, rank, and die inside a wrong-target compile at plan build).
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

constexpr const char* DEVICE_ARCH_WITH_SUFFIX = "gfx942:sramecc+:xnack-";
constexpr const char* DEVICE_ARCH_BARE = "gfx942";

DeviceProperties propertiesFor(const std::string& arch)
{
    DeviceProperties properties = testDeviceProperties();
    properties.gcnArchName = arch;
    return properties;
}

TEST(TestIngestorArchGate, AnEmptyListIsArchIndependent)
{
    EXPECT_TRUE(archSupports({}, DEVICE_ARCH_BARE));
    EXPECT_TRUE(archSupports({}, "gfx90a"));
    EXPECT_TRUE(archSupports({}, ""));
}

TEST(TestIngestorArchGate, MatchesADeviceCarryingATargetIdSuffix)
{
    // A pack lists "gfx942"; the device reports "gfx942:sramecc+:xnack-". Raw string
    // comparison would reject every real device.
    EXPECT_TRUE(archSupports({"gfx942"}, DEVICE_ARCH_WITH_SUFFIX));
    EXPECT_TRUE(archSupports({"gfx942"}, DEVICE_ARCH_BARE));
}

TEST(TestIngestorArchGate, RefusesADifferentArchSharingAPrefix)
{
    // Substring matching would let gfx94 admit gfx942, or gfx942 admit gfx9420.
    EXPECT_FALSE(archSupports({"gfx942"}, "gfx950"));
    EXPECT_FALSE(archSupports({"gfx94"}, DEVICE_ARCH_WITH_SUFFIX));
    EXPECT_FALSE(archSupports({"gfx942"}, "gfx9420"));
}

TEST(TestIngestorArchGate, AdmitsWhenAnyListedArchMatches)
{
    const std::vector<std::string> family{"gfx942", "gfx950"};

    EXPECT_TRUE(archSupports(family, DEVICE_ARCH_WITH_SUFFIX));
    EXPECT_TRUE(archSupports(family, "gfx950"));
    EXPECT_FALSE(archSupports(family, "gfx90a"));
}

TEST(TestIngestorArchGate, PrunesAPackWhoseArchExcludesTheDevice)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID}, {"gfx950"})},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               "test.graph");

    const TestGraph graph(makeGraphId(0x71));
    const auto properties = propertiesFor(DEVICE_ARCH_WITH_SUFFIX);

    EXPECT_TRUE(manager.unsortedDefinitions(MatchContext{graph, 0, properties}).empty());
}

TEST(TestIngestorArchGate, PrunesBeforeRunningAnyMatcher)
{
    // Every pack this engine has is excluded, so nothing downstream of the gate runs at
    // all: not the engine's graph match, not a criterion, not a kernel matcher. An engine
    // with even one admissible pack does evaluate its graph match.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID}, {"gfx950"})},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               "test.graph");

    const TestGraph graph(makeGraphId(0x72));
    const auto properties = propertiesFor(DEVICE_ARCH_WITH_SUFFIX);
    static_cast<void>(manager.unsortedDefinitions(MatchContext{graph, 0, properties}));

    EXPECT_EQ(counters().graphMatchCalls, 0);
    EXPECT_EQ(counters().graphCalls, 0);
    EXPECT_EQ(counters().kernelCalls, 0);
}

TEST(TestIngestorArchGate, AdmitsAPackWhoseArchIncludesTheDevice)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    const StateManager manager(
        makeSchema(),
        makeTestMatchers(),
        makeTestDispatches(),
        {makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID}, {"gfx90a", "gfx942"})},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
        "test.graph");

    const TestGraph graph(makeGraphId(0x73));
    const auto properties = propertiesFor(DEVICE_ARCH_WITH_SUFFIX);

    EXPECT_EQ(manager.unsortedDefinitions(MatchContext{graph, 0, properties}).size(), 2U);
    EXPECT_EQ(counters().graphMatchCalls, 1);
    EXPECT_EQ(counters().graphCalls, 1);
}

TEST(TestIngestorArchGate, GatesPerDeviceRatherThanPerGraph)
{
    // On a mixed-architecture box the answer depends on the targeted device, not on
    // what the machine has installed; the catalog cache is keyed on (graph, device).
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto criterion = scopedGraphMatcher("test.graph_criterion", &acceptCriterion);

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID}, {"gfx942"})},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
                               "test.graph");

    const TestGraph graph(makeGraphId(0x74));
    const auto supported = propertiesFor(DEVICE_ARCH_WITH_SUFFIX);
    const auto unsupported = propertiesFor("gfx90a");

    EXPECT_FALSE(manager.unsortedDefinitions(MatchContext{graph, 0, supported}).empty());
    EXPECT_TRUE(manager.unsortedDefinitions(MatchContext{graph, 1, unsupported}).empty());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
