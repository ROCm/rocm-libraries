/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <rocRoller/KernelGraph/Transforms/All.hpp>

#include <rocRoller/KernelGraph/Transforms/SwizzleScale_detail.hpp>

#include "TestContext.hpp"
#include "common/CommonGraphs.hpp"
#include "common/Utilities.hpp"

namespace SwizzleScaleTest
{
    TEST_CASE("SwizzleScale", "[kernel-graph][graph-transforms]")
    {
        using namespace rocRoller;
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::ControlGraph;
        using namespace rocRoller::KernelGraph::CoordinateGraph;
        using namespace rocRoller::KernelGraph::SwizzleScaleDetail;
        using namespace rocRoller::Operations;

        auto context = TestContext::ForDefaultTarget();
        auto example = rocRollerTest::Graphs::GEMM(DataType::FP4, DataType::FP4, DataType::Half);

        example.setTileSize(128, 128, 128);
        example.setMFMA(16, 16, 128, 1);
        example.setUseLDS(true, true, false);
        example.setUnroll(2, 2);
        example.setScaling(
            ScaleMode::Separate, ScaleMode::None, DataType::E8M0, DataType::None, 32);

        auto graph  = example.getKernelGraph();
        auto params = example.getCommandParameters();

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
        graph = transform<CleanLoops>(graph);

        // Find the K-loop in the graph
        int kLoopTag;
        {
            auto const rootTag = graph.control.roots().only().value();

            auto findNamedLoopsBelow = [&](auto startTag, auto name) {
                std::vector<int> loopTags;
                for(auto const loop : filter(graph.control.isElemType<ForLoopOp>(),
                                             graph.control.depthFirstVisit(startTag)))
                {
                    auto forloop = graph.control.get<ForLoopOp>(loop).value();
                    if(forloop.loopName == name)
                    {
                        loopTags.push_back(loop);
                    }
                }
                return loopTags;
            };

            auto kLoopTags = findNamedLoopsBelow(rootTag, KLOOP);
            REQUIRE(kLoopTags.size() == 1);
            kLoopTag = kLoopTags[0];
        }

        SECTION("collectScaleLoadInfo")
        {
            auto scaleLoadsA = collectScaleLoadInfo(graph, NaryArgument::LHS_SCALE, kLoopTag);
            auto scaleLoadsB = collectScaleLoadInfo(graph, NaryArgument::RHS_SCALE, kLoopTag);

            CHECK(not scaleLoadsA.empty());
            CHECK(scaleLoadsB.empty());
        }

        SECTION("getOuterMergeFactors")
        {
            auto scaleLoadsA = collectScaleLoadInfo(graph, NaryArgument::LHS_SCALE, kLoopTag);
            REQUIRE(not scaleLoadsA.empty());

            // Get a MacroTile tag from scaleLoads
            auto macTileTag = scaleLoadsA.begin()->second.second;

            auto [outerFactorMN, outerFactorK] = getOuterMergeFactors(graph, macTileTag);

            CHECK(outerFactorMN > 0);
            CHECK(outerFactorK > 0);
        }

        SECTION("getInnerMergeFactors")
        {
            auto scaleLoadsA = collectScaleLoadInfo(graph, NaryArgument::LHS_SCALE, kLoopTag);
            REQUIRE(not scaleLoadsA.empty());

            // Get a MacroTile tag from scaleLoads
            auto macTileTag = scaleLoadsA.begin()->second.second;

            auto [innerFactorMN, innerFactorK] = getInnerMergeFactors(graph, macTileTag);

            CHECK(innerFactorMN > 0);
            CHECK(innerFactorK > 0);
        }

        SECTION("findMergeableLoads")
        {
            auto scaleLoadsA = collectScaleLoadInfo(graph, NaryArgument::LHS_SCALE, kLoopTag);
            REQUIRE(not scaleLoadsA.empty());

            auto colouring     = colourByUnrollValue(graph);
            auto loadUnrollMap = filterLoadUnrollColouring(colouring, scaleLoadsA);

            REQUIRE(not loadUnrollMap.empty());

            auto mergeables
                = findMergeableLoads(graph, scaleLoadsA, loadUnrollMap, NaryArgument::LHS_SCALE);

            CHECK(not mergeables.empty());
        }
    }
}
