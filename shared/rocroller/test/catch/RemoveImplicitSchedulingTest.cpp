// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestContext.hpp"
#include <common/CommonGraphs.hpp>
#include <common/Utilities.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/NodeSchedulingUtils.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Transforms/RemoveImplicitScheduling.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

using namespace rocRoller;
using namespace Catch::Matchers;

namespace RemoveImplicitSchedulingTest
{
    TEST_CASE("RemoveImplicitScheduling handles empty graph", "[kernel-graph][graph-transforms]")
    {
        using namespace rocRoller::KernelGraph::ControlGraph;

        rocRoller::KernelGraph::KernelGraph graph;
        auto                                kernel = graph.control.addElement(Kernel());

        // Apply transform on graph with no multiply nodes
        auto transformedGraph = rocRoller::KernelGraph::RemoveImplicitScheduling().apply(graph);

        // Graph structure should be unchanged
        CHECK(transformedGraph.control.getNodes().to<std::vector>().size()
              == graph.control.getNodes().to<std::vector>().size());
    }

    TEST_CASE("RemoveImplicitScheduling works with unrolled loops",
              "[kernel-graph][graph-transforms]")
    {
        using namespace rocRoller::KernelGraph;

        auto dataTypeAB = GENERATE(DataType::Float, DataType::FP4);
        DYNAMIC_SECTION("dataTypeAB=" << dataTypeAB)
        {

            auto dataTypeCD = dataTypeAB == DataType::Float ? DataType::Float : DataType::Half;
            auto example
                = rocRollerTest::Graphs::GEMM(dataTypeAB, dataTypeAB, dataTypeCD, dataTypeCD);

            auto context = TestContext::ForDefaultTarget();

            example.setUseLDS(true, true, false);
            example.setPrefetch(true, 2, 1, true);

            if(dataTypeAB == DataType::Float)
            {
                example.setMFMA(32, 32, 2, 1);
                example.setTileSize(128, 128, 16);
            }
            else
            {
                example.setTileSize(128, 128, 256);
                example.setMFMA(16, 16, 128, 1);
                example.setScaling(Operations::ScaleMode::Separate,
                                   Operations::ScaleMode::Separate,
                                   DataType::E8M0,
                                   DataType::E8M0,
                                   32);
                example.setScaleLoadPaths(SolutionParams::LoadPath::BufferToLDS,
                                          SolutionParams::LoadPath::BufferToLDS);
            }

            auto graph  = example.getKernelGraph();
            auto params = example.getCommandParameters();

            // Apply full transform pipeline up to RemoveImplicitScheduling
            graph = transform<IdentifyParallelDimensions>(graph);
            graph = transform<OrderMemory>(graph, true);
            graph = transform<UpdateParameters>(graph, params);
            graph = transform<AddLDS>(graph, params, context.get());
            graph = transform<LowerLinear>(graph, context.get());
            graph = transform<LowerTile>(graph, params, context.get());
            graph = transform<LowerTensorContraction>(graph, params, context.get());
            graph = transform<Simplify>(graph);
            graph = transform<ConstantPropagation>(graph);
            graph = transform<FuseExpressions>(graph);
            graph = transform<ConnectWorkgroups>(graph, context.get());
            graph = transform<UnrollLoops>(graph, params, context.get());
            graph = transform<FuseLoops>(graph);
            graph = transform<RemoveDuplicates>(graph);
            graph = transform<OrderEpilogueBlocks>(graph);
            graph = transform<Simplify>(graph);
            graph = transform<CleanLoops>(graph);
            graph = transform<AddPrefetch>(graph, params, context.get());

            auto multiplyNodesBefore
                = graph.control.getNodes<ControlGraph::Multiply>().to<std::vector>();
            REQUIRE(!multiplyNodesBefore.empty());

            // Apply RemoveImplicitScheduling
            auto transformedGraph = transform<RemoveImplicitScheduling>(graph);

            REQUIRE(!multiplyNodesBefore.empty());

            auto colouring = rocRoller::KernelGraph::colourByUnrollValue(graph);

            REQUIRE(!multiplyNodesBefore.empty());

            auto multiplyNodesAfter
                = transformedGraph.control.getNodes<ControlGraph::Multiply>().to<std::vector>();

            CHECK(multiplyNodesAfter == multiplyNodesBefore);

            std::set<std::tuple<int, int>> orderedPairsBefore;
            for(auto iter = multiplyNodesBefore.begin(); iter != multiplyNodesBefore.end(); ++iter)
            {
                for(auto iter2 = std::next(iter); iter2 != multiplyNodesBefore.end(); ++iter2)
                {
                    if(graph.control.compareNodes(UpdateCache, *iter, *iter2)
                       != ControlGraph::NodeOrdering::Undefined)
                    {
                        orderedPairsBefore.insert({*iter, *iter2});
                    }
                }
            }

            int newlyUnorderedPairs = 0;
            for(auto const& [node1, node2] : orderedPairsBefore)
            {
                auto order = transformedGraph.control.compareNodes(UpdateCache, node1, node2);
                if(order == ControlGraph::NodeOrdering::Undefined)
                {
                    CHECK(colouring.operationColour.at(node1)
                          != colouring.operationColour.at(node2));
                    newlyUnorderedPairs++;
                }
                else
                {
                    CHECK(order == graph.control.compareNodes(UpdateCache, node1, node2));
                }
            }
            CHECK(newlyUnorderedPairs > 0);
        }
    }

} // namespace RemoveImplicitSchedulingTest
