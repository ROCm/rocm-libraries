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
 *        matching rather than after.
 *
 * A pack that reaches a device it cannot run on does not fail cleanly. It matches, wins
 * ranking, and dies inside a wrong-target compile at plan build, past the point
 * RFC 0017 §8.6 turned applicability into a promise. These tests pin the gate that makes
 * that unreachable.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

/// A device reports its target with feature flags attached; a pack lists the base id.
constexpr const char* DEVICE_ARCH_WITH_SUFFIX = "gfx942:sramecc+:xnack-";
constexpr const char* DEVICE_ARCH_BARE = "gfx942";

DeviceProperties propertiesFor(const std::string& arch)
{
    DeviceProperties properties = testDeviceProperties();
    properties.gcnArchName = arch;
    return properties;
}

// ---------------------------------------------------------------------------
// archSupports(): the predicate
// ---------------------------------------------------------------------------

TEST(TestIngestorArchGate, AnEmptyListIsArchIndependent)
{
    // The default every existing pack ships, and what keeps the field additive.
    EXPECT_TRUE(archSupports({}, DEVICE_ARCH_BARE));
    EXPECT_TRUE(archSupports({}, "gfx90a"));
    EXPECT_TRUE(archSupports({}, ""));
}

TEST(TestIngestorArchGate, MatchesADeviceCarryingATargetIdSuffix)
{
    // The detail most likely to be got wrong: a pack lists "gfx942" and the device
    // reports "gfx942:sramecc+:xnack-". Comparing raw strings makes the gate reject
    // every real device, which would look like the pack simply never applying.
    EXPECT_TRUE(archSupports({"gfx942"}, DEVICE_ARCH_WITH_SUFFIX));
    EXPECT_TRUE(archSupports({"gfx942"}, DEVICE_ARCH_BARE));
}

TEST(TestIngestorArchGate, RefusesADifferentArchSharingAPrefix)
{
    // gfx94 must not admit gfx942, and gfx942 must not admit gfx9420. Substring
    // matching would accept both and silently run a pack on hardware it was never
    // built for.
    EXPECT_FALSE(archSupports({"gfx942"}, "gfx950"));
    EXPECT_FALSE(archSupports({"gfx94"}, DEVICE_ARCH_WITH_SUFFIX));
    EXPECT_FALSE(archSupports({"gfx942"}, "gfx9420"));
}

TEST(TestIngestorArchGate, AdmitsWhenAnyListedArchMatches)
{
    // A pack serving a family lists every member; matching is per entry.
    const std::vector<std::string> family{"gfx942", "gfx950"};

    EXPECT_TRUE(archSupports(family, DEVICE_ARCH_WITH_SUFFIX));
    EXPECT_TRUE(archSupports(family, "gfx950"));
    EXPECT_FALSE(archSupports(family, "gfx90a"));
}

// ---------------------------------------------------------------------------
// The gate inside buildCatalog
// ---------------------------------------------------------------------------

TEST(TestIngestorArchGate, PrunesAPackWhoseArchExcludesTheDevice)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID}, {"gfx950"})},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const TestGraph graph(makeGraphId(0x71));
    const auto properties = propertiesFor(DEVICE_ARCH_WITH_SUFFIX);

    EXPECT_TRUE(manager.unsortedDefinitions(MatchContext{graph, 0, properties}).empty());
}

TEST(TestIngestorArchGate, PrunesBeforeRunningAnyMatcher)
{
    // The point of placing the check first. A pack excluded by arch must cost no
    // registry resolve, no matcher body, and no per-kernel metadata completion: all of
    // it is provably wasted, and it scales with kernel count.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID}, {"gfx950"})},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const TestGraph graph(makeGraphId(0x72));
    const auto properties = propertiesFor(DEVICE_ARCH_WITH_SUFFIX);
    static_cast<void>(manager.unsortedDefinitions(MatchContext{graph, 0, properties}));

    EXPECT_EQ(counters().graphCalls, 0);
    EXPECT_EQ(counters().kernelCalls, 0);
}

TEST(TestIngestorArchGate, AdmitsAPackWhoseArchIncludesTheDevice)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    const StateManager manager(
        makeSchema(),
        makeTestMatchers(),
        makeTestDispatches(),
        {makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID}, {"gfx90a", "gfx942"})},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const TestGraph graph(makeGraphId(0x73));
    const auto properties = propertiesFor(DEVICE_ARCH_WITH_SUFFIX);

    // Two FLOAT kernels survive; the HALF one is pruned by the kernel-scoped matcher.
    EXPECT_EQ(manager.unsortedDefinitions(MatchContext{graph, 0, properties}).size(), 2U);
    EXPECT_EQ(counters().graphCalls, 1);
}

TEST(TestIngestorArchGate, GatesPerDeviceRatherThanPerGraph)
{
    // The reason the load-time gate ALMIOPEN-2401 adds does not replace this one: on a
    // mixed-architecture box the answer depends on the device this call targets, not on
    // what the machine has installed. The catalog cache is keyed on (graph, device) for
    // the same reason.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);

    const StateManager manager(makeSchema(),
                               makeTestMatchers(),
                               makeTestDispatches(),
                               {makePack({GRAPH_MATCHER_ID, KERNEL_MATCHER_ID}, {"gfx942"})},
                               std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const TestGraph graph(makeGraphId(0x74));
    const auto supported = propertiesFor(DEVICE_ARCH_WITH_SUFFIX);
    const auto unsupported = propertiesFor("gfx90a");

    EXPECT_FALSE(manager.unsortedDefinitions(MatchContext{graph, 0, supported}).empty());
    EXPECT_TRUE(manager.unsortedDefinitions(MatchContext{graph, 1, unsupported}).empty());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
