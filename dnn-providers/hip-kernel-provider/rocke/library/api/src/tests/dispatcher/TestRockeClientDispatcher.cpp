// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "dispatcher/RockeClientDispatcher.hpp"
#include "tests/dispatcher/DispatcherFixtures.hpp"
#include "tests/dispatcher/SdpaGraphFixture.hpp"

namespace rocke_client::dispatcher
{
namespace
{

using test::buildSdpaGraph;
using test::InstanceParams;
using test::makeInstance;
using test::makeMatchingProblem;
using test::SdpaGraphConfig;

InstanceParams d64Params()
{
    InstanceParams params;
    params.name = "d64";
    params.headSize = 64;
    return params;
}

InstanceParams d128Params()
{
    InstanceParams params;
    params.name = "d128";
    params.headSize = 128;
    return params;
}

RockeClientDispatcher twoInstanceDispatcher()
{
    return RockeClientDispatcher(AotCatalog(
        std::vector<AotInstance>{makeInstance(d64Params()), makeInstance(d128Params())}));
}

TEST(TestRockeClientDispatcher, DisjointProblemsSelectDistinctInstances)
{
    const RockeClientDispatcher dispatcher = twoInstanceDispatcher();

    const std::optional<AotInstance> winnerD64
        = dispatcher.select(makeMatchingProblem(d64Params()));
    const std::optional<AotInstance> winnerD128
        = dispatcher.select(makeMatchingProblem(d128Params()));

    ASSERT_TRUE(winnerD64.has_value());
    ASSERT_TRUE(winnerD128.has_value());
    EXPECT_EQ(winnerD64->name, "d64");
    EXPECT_EQ(winnerD128->name, "d128");
    EXPECT_NE(winnerD64->name, winnerD128->name);
}

TEST(TestRockeClientDispatcher, DeclinesUnavailableProblem)
{
    const RockeClientDispatcher dispatcher = twoInstanceDispatcher();

    InstanceParams d256 = d64Params();
    d256.headSize = 256;
    EXPECT_FALSE(dispatcher.select(makeMatchingProblem(d256)).has_value());
}

TEST(TestRockeClientDispatcher, EmptyCatalogDeclines)
{
    const RockeClientDispatcher dispatcher{AotCatalog{}};
    EXPECT_FALSE(dispatcher.select(makeMatchingProblem(d64Params())).has_value());
}

TEST(TestRockeClientDispatcher, ArchScopesSelection)
{
    const RockeClientDispatcher dispatcher = twoInstanceDispatcher(); // gfx942 instances

    SdpaProblem otherArch = makeMatchingProblem(d64Params());
    otherArch.arch = "gfx950";
    EXPECT_FALSE(dispatcher.select(otherArch).has_value());
}

TEST(TestRockeClientDispatcher, DeterministicFirstMatchOnTie)
{
    InstanceParams first = d64Params();
    first.name = "first_match";
    InstanceParams second = d64Params();
    second.name = "second_match";

    const RockeClientDispatcher dispatcher(
        AotCatalog(std::vector<AotInstance>{makeInstance(first), makeInstance(second)}));

    const std::optional<AotInstance> winner = dispatcher.select(makeMatchingProblem(d64Params()));
    ASSERT_TRUE(winner.has_value());
    EXPECT_EQ(winner->name, "first_match");
}

// The default fixture graph (fp16 / BSHD / mask none / head 64 / batch 2 / heads
// 4) matches the d64 gfx942 instance, so selectForArch drives the full
// graph -> problem -> select path end to end without a device.
TEST(TestRockeClientDispatcher, SelectForArchMatchesValidGraph)
{
    const RockeClientDispatcher dispatcher = twoInstanceDispatcher();
    const auto fixture = buildSdpaGraph(SdpaGraphConfig{});

    const std::optional<AotInstance> winner
        = dispatcher.selectForArch("gfx942", fixture.graphWrapper());

    ASSERT_TRUE(winner.has_value());
    EXPECT_EQ(winner->name, "d64");
}

// Distinguishes the two decline causes: an unresolved arch ("" on host CI) never
// matches even when the catalog is populated and the graph is valid.
TEST(TestRockeClientDispatcher, SelectForArchDeclinesWhenArchUnresolved)
{
    const RockeClientDispatcher dispatcher = twoInstanceDispatcher();
    const auto fixture = buildSdpaGraph(SdpaGraphConfig{});

    EXPECT_FALSE(dispatcher.selectForArch("", fixture.graphWrapper()).has_value());
}

// ...as opposed to an empty catalog, which declines a valid graph for a resolved
// arch. This is the Phase-1 no-op cause.
TEST(TestRockeClientDispatcher, SelectForArchDeclinesWithEmptyCatalog)
{
    const RockeClientDispatcher dispatcher{AotCatalog{}};
    const auto fixture = buildSdpaGraph(SdpaGraphConfig{});

    EXPECT_FALSE(dispatcher.selectForArch("gfx942", fixture.graphWrapper()).has_value());
}

// An unsupported graph is declined by the adapter, so selectForArch yields
// nothing even for a resolved arch with a populated catalog.
TEST(TestRockeClientDispatcher, SelectForArchDeclinesUnsupportedGraph)
{
    const RockeClientDispatcher dispatcher = twoInstanceDispatcher();
    SdpaGraphConfig config;
    config.alibiMask = true; // adapter rejects -> translate returns nullopt
    const auto fixture = buildSdpaGraph(config);

    EXPECT_FALSE(dispatcher.selectForArch("gfx942", fixture.graphWrapper()).has_value());
}

// A valid graph whose shape no instance was built for is declined by selection.
TEST(TestRockeClientDispatcher, SelectForArchDeclinesShapeWithoutInstance)
{
    const RockeClientDispatcher dispatcher = twoInstanceDispatcher();
    SdpaGraphConfig config;
    config.seqlenQ = 999; // no instance has this seqlen
    config.seqlenK = 999;
    const auto fixture = buildSdpaGraph(config);

    EXPECT_FALSE(dispatcher.selectForArch("gfx942", fixture.graphWrapper()).has_value());
}

} // namespace
} // namespace rocke_client::dispatcher
