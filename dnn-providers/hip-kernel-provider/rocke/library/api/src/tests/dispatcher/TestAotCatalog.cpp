// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <vector>

#include "dispatcher/AotCatalog.hpp"
#include "tests/dispatcher/DispatcherFixtures.hpp"

namespace rocke_client::dispatcher
{
namespace
{

using test::InstanceParams;
using test::makeInstance;

TEST(TestAotCatalog, DefaultProductionCatalogIsEmpty)
{
    // Phase 1 invariant: no runtime catalog exists yet (kpack not landed), so the
    // production catalog is empty and nothing can be selected.
    const AotCatalog catalog = AotCatalog::loadDefault();
    EXPECT_TRUE(catalog.empty());
    EXPECT_EQ(catalog.size(), 0u);
    EXPECT_TRUE(catalog.candidatesFor("sdpa_fwd", "gfx942").empty());
}

TEST(TestAotCatalog, DefaultConstructedCatalogIsEmpty)
{
    const AotCatalog catalog;
    EXPECT_TRUE(catalog.empty());
    EXPECT_TRUE(catalog.candidatesFor("sdpa_fwd", "gfx942").empty());
}

TEST(TestAotCatalog, CandidatesForFiltersByOpAndArch)
{
    InstanceParams d64;
    d64.name = "d64_gfx942";
    d64.arch = "gfx942";
    d64.headSize = 64;

    InstanceParams d128;
    d128.name = "d128_gfx942";
    d128.arch = "gfx942";
    d128.headSize = 128;

    InstanceParams other;
    other.name = "d64_gfx950";
    other.arch = "gfx950";
    other.headSize = 64;

    const AotCatalog catalog(
        std::vector<AotInstance>{makeInstance(d64), makeInstance(d128), makeInstance(other)});

    EXPECT_EQ(catalog.size(), 3u);

    const auto gfx942 = catalog.candidatesFor("sdpa_fwd", "gfx942");
    EXPECT_EQ(gfx942.size(), 2u);

    const auto gfx950 = catalog.candidatesFor("sdpa_fwd", "gfx950");
    ASSERT_EQ(gfx950.size(), 1u);
    EXPECT_EQ(gfx950.front().get().name, "d64_gfx950");

    EXPECT_TRUE(catalog.candidatesFor("sdpa_fwd", "gfx90a").empty());
    EXPECT_TRUE(catalog.candidatesFor("conv_fwd", "gfx942").empty());
}

TEST(TestAotCatalog, CandidatesPreserveInsertionOrder)
{
    InstanceParams first;
    first.name = "first";
    InstanceParams second;
    second.name = "second";

    const AotCatalog catalog(std::vector<AotInstance>{makeInstance(first), makeInstance(second)});
    const auto candidates = catalog.candidatesFor("sdpa_fwd", "gfx942");
    ASSERT_EQ(candidates.size(), 2u);
    EXPECT_EQ(candidates[0].get().name, "first");
    EXPECT_EQ(candidates[1].get().name, "second");
}

} // namespace
} // namespace rocke_client::dispatcher
