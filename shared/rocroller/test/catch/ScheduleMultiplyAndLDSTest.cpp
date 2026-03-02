// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestContext.hpp"
#include <common/CommonGraphs.hpp>
#include <common/Utilities.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Transforms/ScheduleMultiplyAndLDS.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

using namespace rocRoller;
using namespace Catch::Matchers;

namespace ScheduleMultiplyAndLDSTest
{
    TEST_CASE("ScheduleMultiplyAndLDS handles empty graph",
              "[kernel-graph][graph-transforms][schedule-multiply-lds]")
    {
        using namespace rocRoller::KernelGraph::ControlGraph;

        rocRoller::KernelGraph::KernelGraph graph;
        auto                                kernel = graph.control.addElement(Kernel());

        // Apply transform on graph with no multiply or LDS nodes
        auto transformedGraph = rocRoller::KernelGraph::ScheduleMultiplyAndLDS().apply(graph);

        // Graph structure should be unchanged
        CHECK(transformedGraph.control.getNodes().to<std::vector>().size()
              == graph.control.getNodes().to<std::vector>().size());
    }

    TEST_CASE("ScheduleMultiplyAndLDS creates sequence edges between Multiply and LoadLDSTile",
              "[kernel-graph][graph-transforms][schedule-multiply-lds]")
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

            // Apply full transform pipeline up to ScheduleMultiplyAndLDS
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
            graph = transform<RemoveImplicitScheduling>(graph);
            graph = transform<OrderMultiplyNodes>(graph);
            graph = transform<Simplify>(graph);

            // Get nodes and edges before applying ScheduleMultiplyAndLDS
            auto multiplyNodesBefore
                = graph.control.getNodes<ControlGraph::Multiply>().to<std::vector>();
            auto ldsNodesBefore
                = graph.control.getNodes<ControlGraph::LoadLDSTile>().to<std::vector>();

            REQUIRE(!multiplyNodesBefore.empty());
            REQUIRE(!ldsNodesBefore.empty());

            // Count Sequence edges before
            auto sequenceEdgesBefore
                = graph.control.getEdges<ControlGraph::Sequence>()
                      .filter([&](auto edge) {
                          return graph.control.get<ControlGraph::Sequence>(edge).has_value();
                      })
                      .to<std::vector>()
                      .size();

            // Apply ScheduleMultiplyAndLDS
            auto transformedGraph = transform<ScheduleMultiplyAndLDS>(graph);

            // Get nodes and edges after transformation
            auto multiplyNodesAfter
                = transformedGraph.control.getNodes<ControlGraph::Multiply>().to<std::vector>();
            auto ldsNodesAfter
                = transformedGraph.control.getNodes<ControlGraph::LoadLDSTile>().to<std::vector>();

            // Verify multiply and LDS nodes are preserved
            CHECK(multiplyNodesAfter == multiplyNodesBefore);
            CHECK(ldsNodesAfter == ldsNodesBefore);

            // Count Sequence edges after
            auto sequenceEdgesAfter
                = transformedGraph.control.getEdges<ControlGraph::Sequence>()
                      .filter([&](auto edge) {
                          return transformedGraph.control.get<ControlGraph::Sequence>(edge)
                              .has_value();
                      })
                      .to<std::vector>()
                      .size();

            // Verify new Sequence edges were created
            CHECK(sequenceEdgesAfter > sequenceEdgesBefore);

            // Verify that some Multiply nodes now have Sequence edges to LoadLDSTile nodes
            bool foundMultiplyToLDSEdge = false;
            for(auto multiplyNode : multiplyNodesAfter)
            {
                auto sequenceTargets
                    = transformedGraph.control
                          .getOutputNodeIndices<ControlGraph::Sequence>(multiplyNode)
                          .to<std::vector>();
                for(auto target : sequenceTargets)
                {
                    if(transformedGraph.control.get<ControlGraph::LoadLDSTile>(target).has_value()
                       || transformedGraph.control.get<ControlGraph::SetCoordinate>(target)
                              .has_value())
                    {
                        foundMultiplyToLDSEdge = true;
                        break;
                    }
                }
                if(foundMultiplyToLDSEdge)
                    break;
            }

            CHECK(foundMultiplyToLDSEdge);
        }
    }

} // namespace ScheduleMultiplyAndLDSTest
