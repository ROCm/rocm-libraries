// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelOptions.hpp>

#include "TestContext.hpp"
#include "common/CommonGraphs.hpp"
#include "common/Utilities.hpp"

namespace PrefetchScaleTest
{
    TEST_CASE("PrefetchScale exchange placement", "[kernel-graph][graph-transforms]")
    {
        using namespace rocRoller;
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::ControlGraph;
        using namespace rocRoller::KernelGraph::CoordinateGraph;
        using namespace rocRoller::Operations;

        auto context = TestContext::ForTarget(GPUArchitectureTarget{GPUArchitectureGFX::GFX950});
        auto example = rocRollerTest::Graphs::GEMM(DataType::FP4, DataType::FP4, DataType::Half);

        example.setTileSize(128, 128, 256);
        example.setMFMA(16, 16, 128, 1);
        example.setUseLDS(true, true, false);
        example.setScaling(
            ScaleMode::Separate, ScaleMode::Separate, DataType::E8M0, DataType::E8M0, 32);
        example.setScaleLoadPaths(SolutionParams::LoadPath::BufferToVGPR,
                                  SolutionParams::LoadPath::BufferToVGPR);
        example.setPrefetch(true, 4, 1, true);
        example.setSwizzle(32, 32, 8, 8, true);
        example.setTranspose("T", "N");
        example.setStreamK(StreamKMode::Standard);

        auto command = example.getCommand();
        auto graph   = example.getKernelGraph();
        auto params  = example.getCommandParameters();

        graph = transform<UpdateParameters>(graph, params);
        graph = transform<IdentifyParallelDimensions>(graph);
        graph = transform<OrderMemory>(graph, true);
        graph = transform<AddLDS>(graph, params, context.get());
        graph = transform<LowerLinear>(graph, context.get());
        graph = transform<LowerTile>(graph, params, context.get());
        graph = transform<LowerTensorContraction>(graph, params, context.get());
        graph = transform<Simplify>(graph);
        graph = transform<ConstantPropagation>(graph);
        graph = transform<FuseExpressions>(graph);

        // AddStreamK has pre-loop of K loop inside of
        // another ForLoop which cause the bug.
        {
            auto numWGsArg = findArgumentByName(command, NUMWGS);
            REQUIRE(numWGsArg != nullptr);
            graph = transform<AddStreamK>(graph,
                                          context.get(),
                                          params,
                                          XLOOP,
                                          KLOOP,
                                          std::make_shared<Expression::Expression>(numWGsArg));
        }

        graph = transform<ConnectWorkgroups>(graph, context.get());
        graph = transform<WorkgroupRemapXCC>(graph, context.get(), params->workgroupRemapXCC);
        graph = transform<UnrollLoops>(graph, params, context.get());
        graph = transform<FuseLoops>(graph);
        graph = transform<RemoveDuplicates>(graph);
        graph = transform<OrderEpilogueBlocks>(graph);
        graph = transform<Simplify>(graph);
        graph = transform<CleanLoops>(graph);

        graph = transform<SwizzleScale>(graph, params, context.get());
        graph = transform<AddPrefetch>(graph, params, context.get());
        graph = transform<PrefetchScale>(graph, params, context.get());

        auto findLoop = [&](std::string const& name) -> std::optional<int> {
            auto const rootTag = graph.control.roots().only().value();
            for(auto const loop : filter(graph.control.isElemType<ForLoopOp>(),
                                         graph.control.depthFirstVisit(rootTag)))
            {
                auto forloop = graph.control.get<ForLoopOp>(loop).value();
                if(forloop.loopName == name)
                    return loop;
            }
            FAIL("Loop '" + name + "' not found");
            return std::nullopt;
        };

        auto kLoopTag = findLoop(KLOOP);
        REQUIRE(kLoopTag.has_value());

        SECTION("Exchange nodes are placed inside the K loop body")
        {
            auto isExchange = graph.control.isElemType<Exchange>();

            int exchangesInKLoop  = 0;
            int exchangesOutsideK = 0;
            int totalExchanges    = 0;

            std::optional<int> kLoopTailTag = findLoop(KLOOPTAIL);

            for(auto exchangeTag : graph.control.getNodes().filter(isExchange))
            {
                totalExchanges++;
                auto stack = controlStack(exchangeTag, graph);
                bool insideKLoop
                    = std::find(stack.begin(), stack.end(), kLoopTag.value()) != stack.end();
                bool insideKLoopTail
                    = kLoopTailTag.has_value()
                      && std::find(stack.begin(), stack.end(), kLoopTailTag.value()) != stack.end();
                if(insideKLoop && !insideKLoopTail)
                    exchangesInKLoop++;
                else
                    exchangesOutsideK++;
            }

            INFO("totalExchanges=" << totalExchanges << " inKLoopBody=" << exchangesInKLoop
                                   << " outsideKBody=" << exchangesOutsideK);
            REQUIRE(totalExchanges > 0);

            // With prefetchScale and StreamK, the K loop body must contain
            // Exchange nodes for subiters. A prior bug in
            // DeterminePrefetchPositions caused it to place all
            // exchanges in the pre-loop instead of the K loop body.
            CHECK(exchangesInKLoop > 0);

            // The K loop body should have more exchanges than outside
            // (pre-loop + tail), because it processes all subiters each iteration.
            CHECK(exchangesInKLoop >= exchangesOutsideK);
        }
    }
}
