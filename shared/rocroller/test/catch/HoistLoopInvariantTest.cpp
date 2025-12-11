/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
#include <catch2/matchers/catch_matchers_contains.hpp>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Transforms/HoistLoopInvariant_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include <fstream>
#include <iostream>

#include "TestContext.hpp"
#include <common/CommonGraphs.hpp>
#include <common/Utilities.hpp>

void writeDotToFile(const std::string& dotContent, const std::string& filename)
{
    std::ofstream outFile(filename);
    if(outFile.is_open())
    {
        outFile << dotContent;
        outFile.close();
        std::cout << filename << std::endl;
    }
}

TEST_CASE("extractDataFlowTags", "[kernel-graph][hoist-loop-invariant][expression]")
{
    using namespace rocRoller;
    namespace kg = rocRoller::KernelGraph;

    SECTION("Binary operation with DataFlowTags")
    {
        Expression::DataFlowTag tag1{42, Register::Type::Vector, DataType::Float};
        Expression::DataFlowTag tag2{77, Register::Type::Vector, DataType::Float};

        auto tag1Ptr = std::make_shared<Expression::Expression>(tag1);
        auto tag2Ptr = std::make_shared<Expression::Expression>(tag2);

        auto binaryExpr = Expression::Add{{tag1Ptr, tag2Ptr}};

        auto extractedTags = kg::extractDataFlowTags(binaryExpr);

        REQUIRE(extractedTags.size() == 2);
        REQUIRE(extractedTags.count(42) == 1);
        REQUIRE(extractedTags.count(77) == 1);
    }
}

TEST_CASE("hoistNodeBeforeLoop", "[kernel-graph][hoist-loop-invariant]")
{
    using namespace rocRoller;
    namespace kg = rocRoller::KernelGraph;
    using namespace kg::ControlGraph;
    using namespace kg::CoordinateGraph;

    auto ctx = TestContext::ForDefaultTarget();

    kg::KernelGraph graph;

    auto predecessor = graph.control.addElement(NOP{});

    auto loopSize                  = Expression::literal(10);
    auto [loopIndexCoord, forLoop] = kg::rangeFor(graph, loopSize, "TestLoop");

    auto loopBody = graph.control.addElement(NOP{});
    graph.control.addElement(Body{}, {forLoop}, {loopBody});
    auto sequenceEdge = graph.control.addElement(Sequence{}, {predecessor}, {forLoop});

    auto nodeToHoist  = graph.control.addElement(NOP{});
    auto nodeToRemain = graph.control.addElement(NOP{});

    // Connect the nodes: loopBody -> nodeToHoist -> nodeToRemain
    graph.control.addElement(Sequence{}, {loopBody}, {nodeToHoist});
    graph.control.addElement(Sequence{}, {nodeToHoist}, {nodeToRemain});

    std::string dotOutputBefore = graph.toDOT(true);
    writeDotToFile(dotOutputBefore, "hoistNodeBeforeLoop_before.dot");

    // Hoist only the first node
    int result = kg::hoistNodeBeforeLoop(graph, nodeToHoist, forLoop);

    std::string dotOutputAfter = graph.toDOT(true);
    writeDotToFile(dotOutputAfter, "hoistNodeBeforeLoop_after.dot");
}

TEST_CASE("hoist loop invariant helpers", "[kernel-graph][hoist-loop-invariant]")
{
    using namespace rocRoller;
    using namespace rocRoller::KernelGraph;
    namespace kg = rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::CoordinateGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    auto context = TestContext::ForDefaultTarget();

    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

    int macK  = 16;
    int waveK = 2;

    example.setTileSize(256, 64, macK);
    example.setMFMA(32, 32, waveK, 1);
    example.setUseLDS(true, true, false);
    example.setUnroll(2, 2);

    example.setPrefetch(true, 2, 2, false);

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
    graph = transform<WorkgroupRemapXCC>(graph, context.get(), params->workgroupRemapXCC);
    graph = transform<UnrollLoops>(graph, params, context.get());
    graph = transform<FuseLoops>(graph);
    graph = transform<RemoveDuplicates>(graph);
    graph = transform<OrderEpilogueBlocks>(graph);
    graph = transform<CleanLoops>(graph);
    graph = transform<AddPrefetch>(graph, params, context.get());
    graph = transform<AddComputeIndex>(graph);
    graph = transform<AddPRNG>(graph, context.get());
    graph = transform<UpdateWavefrontParameters>(graph, params);
    graph = transform<LoadPacked>(graph, context.get());
    graph = transform<AddConvert>(graph);
    graph = transform<AddDeallocateDataFlow>(graph);
    graph = transform<InlineIncrements>(graph);
    graph = transform<Simplify>(graph);

    ControlFlowRWTracer tracer(graph);
    SECTION("buildCoordinateLoopMapping")
    {
        auto loopMapping = kg::buildCoordinateLoopMapping(graph, tracer);

        for(const auto& [coord, loopSet] : loopMapping)
        {
            Log::info("Coordinate {} is associated with loops:", coord);
            for(const auto& loop : loopSet)
            {
                Log::info("  Loop {}", loop);
            }
        }

        {
            // coord 510 is a MacroTile
            CHECK(loopMapping[510].size() > 0);
            CHECK_NOTHROW(graph.coordinates.getNode<MacroTile>(510));

            // written in k loop and tail loop
            bool foundKLoop    = false;
            bool foundTailLoop = false;
            for(const auto& [loop, writes] : loopMapping[510])
            {
                auto str = Graph::variantToString(graph.control.getElement(loop));
                if(str.find("KLoop") != std::string::npos)
                    foundKLoop = true;
                if(str.find("KLoopTail") != std::string::npos)
                    foundTailLoop = true;
                CHECK(writes.size() >= 8); // written at least 8 times in each loop
                Log::info("Coordinate 510 written in loop {} with {} writes", str, writes);
            }
            CHECK(foundKLoop);
            CHECK(foundTailLoop);
        }
        CHECK(loopMapping.size() == 140);
    }

    SECTION("isCoordinateWrittenInLoop")
    {
        // Written in both
        CHECK(isCoordinateWrittenInLoop(graph, 1122, 510, tracer));
        CHECK(isCoordinateWrittenInLoop(graph, 1127, 510, tracer));

        // For loop variable only written in one loop
        CHECK(isCoordinateWrittenInLoop(graph, 1127, 314, tracer));
        CHECK(!isCoordinateWrittenInLoop(graph, 1122, 314, tracer));
    }

    SECTION("hoistNodeBeforeLoop")
    {
        const int nodeBeforeLoop = 3143;
        const int assignNode     = 3048;
        const int loopNode       = 1127;

        std::vector<int> oldPath
            = {nodeBeforeLoop, 3146, loopNode, 1982, 314, 318, 218, 3650, assignNode};
        // -1 for new nodes/edges
        std::vector<int> newPath
            = {nodeBeforeLoop, 3653, -1, -1, loopNode, 1982, 314, 318, 218, 3650, assignNode};

        const auto compare = [](const std::vector<int>& actual, const std::vector<int>& expected) {
            REQUIRE(actual.size() == expected.size());
            for(size_t i = 0; i < expected.size(); ++i)
            {
                if(expected[i] == -1)
                    continue;
                CHECK(actual[i] == expected[i]);
            }
        };

        const auto oldPathResult = graph.control
                                       .path<Graph::Direction::Downstream>(
                                           std::vector{nodeBeforeLoop}, std::vector{assignNode})
                                       .to<std::vector>();
        compare(oldPathResult, oldPath);

        const auto oldAssignExpression = graph.control.get<Assign>(assignNode)->expression;

        hoistNodeBeforeLoop(graph, assignNode, loopNode);

        const auto newPathResult = graph.control
                                       .path<Graph::Direction::Downstream>(
                                           std::vector{nodeBeforeLoop}, std::vector{assignNode})
                                       .to<std::vector>();
        compare(newPathResult, newPath);

        AssertFatal(newPathResult.size() == 11, newPathResult.size());
        const auto newAssignExpression = graph.control.get<Assign>(newPathResult[2])->expression;

        CHECK(oldAssignExpression == newAssignExpression);
    }
}