// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestContext.hpp"

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>

#include <common/CommonGraphs.hpp>

TEST_CASE("AssignIndexExpressions creates Assign nodes", "[kernel-graph]")
{
    using namespace rocRoller;
    using namespace KernelGraph;
    using namespace ControlGraph;

    auto context = TestContext::ForDefaultTarget();

    auto example = rocRollerTest::Graphs::TileDoubleAdd<float>();

    example.setTileSize(16, 8);
    example.setSubTileSize(4, 2);

    auto params = example.getCommandParameters(512, 512);
    auto graph  = example.getKernelGraph();

    std::vector<GraphTransformPtr> transforms{
        std::make_shared<UpdateParameters>(params),
        std::make_shared<AddLDS>(params, context.get()),
        std::make_shared<LowerLinear>(context.get()),
        std::make_shared<LowerTile>(params, context.get()),
        std::make_shared<AssignIndexExpressions>(context.get(), example.getCommand()),
        std::make_shared<UpdateWavefrontParameters>(params),
    };

    for(auto const& xform : transforms)
    {
        graph = graph.transform(xform);
    }

    // Verify that there are Assign nodes that compute offsets and strides
    auto isAssignPredicate = [&graph](int x) { return graph.control.get<Assign>(x).has_value(); };
    auto assignCandidates
        = graph.control.findNodes(*graph.control.roots().begin(), isAssignPredicate)
              .to<std::vector>();
    CHECK(!assignCandidates.empty());

    // Verify that load/store operations have connections to coordinate graph
    auto isLoadStorePredicate = [&graph](int x) {
        return graph.control.get<LoadTiled>(x).has_value()
               || graph.control.get<StoreTiled>(x).has_value()
               || graph.control.get<LoadLDSTile>(x).has_value()
               || graph.control.get<StoreLDSTile>(x).has_value();
    };
    auto loadStoreCandidates
        = graph.control.findNodes(*graph.control.roots().begin(), isLoadStorePredicate)
              .to<std::vector>();

    for(auto const& tag : loadStoreCandidates)
    {
        // Each load/store operation should have coordinate connections
        auto conns = graph.mapper.getConnections(tag);

        // Operations should have connections to coordinate graph nodes
        CHECK(!conns.empty());
    }
}
