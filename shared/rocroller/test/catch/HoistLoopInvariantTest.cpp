// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

TEST_CASE("hoist loop invariant", "[kernel-graph][hoist-loop-invariant]")
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
    graph = transform<Simplify>(graph);
    graph = transform<CleanLoops>(graph);
    graph = transform<SwizzleScale>(graph, params, context.get());
    graph = transform<AddPrefetch>(graph, params, context.get());
    graph = transform<AddPRNG>(graph, context.get());
    graph = transform<UpdateWavefrontParameters>(graph, params);

    ControlFlowRWTracer tracer(graph);

    const auto [a, b, c, d] = example.getOperationTags();

    /** Yields accumulator macro tile tags encountered by following output edges */
    auto accumulatorMacroTiles = [&](auto commandTag) -> Generator<int> {
        for(auto tag : graph.coordinates.getNodes<User>())
        {
            const auto user = graph.coordinates.get<User>(tag).value();
            if(user.commandTag == commandTag)
            {
                auto tags = graph.coordinates.followEdges<DataFlowEdge>({tag});
                for(auto t : tags)
                {
                    const auto node = graph.coordinates.getNode(t);
                    if(std::holds_alternative<MacroTile>(node))
                    {
                        const auto& macroTile = std::get<MacroTile>(node);
                        if(macroTile.layoutType == LayoutType::MATRIX_ACCUMULATOR)
                        {
                            co_yield t;
                        }
                    }
                }
            }
        }
    };

    int kLoop = -1, kLoopTail = -1;
    for(auto tag : graph.control.getNodes<ForLoopOp>())
    {
        const auto loop = graph.control.get<ForLoopOp>(tag).value();
        if(loop.loopName == "KLoop")
            kLoop = tag;
        else if(loop.loopName == "KLoopTail")
            kLoopTail = tag;
    }
    AssertFatal(kLoop != -1 && kLoopTail != -1, ShowValue(kLoop), ShowValue(kLoopTail));

    SECTION("buildCoordinateLoopMapping")
    {
        auto loopMapping = kg::buildCoordinateLoopMapping(graph, tracer);

        for(auto child : accumulatorMacroTiles(a))
        {
            for(const auto tag : graph.coordinates.getInputNodeIndices(child, isEdge<Duplicate>))
            {
                CAPTURE(child);
                CHECK(loopMapping.at(tag).count(kLoop) > 0);
                CHECK(loopMapping.at(tag).at(kLoop).size() == 16);
                CHECK(loopMapping.at(tag).count(kLoopTail) > 0);
                CHECK(loopMapping.at(tag).at(kLoopTail).size() == 8);
            }
            break; // second encountered macro tile and beyond are not used in kloop[tail]
        }
    }

    SECTION("countCoordinateWritesInLoop")
    {
        {
            bool didACheck = false;
            for(auto tag : accumulatorMacroTiles(a))
            {
                CAPTURE(tag);
                CHECK(countCoordinateWritesInLoop(graph, kLoop, tag, tracer) == 16);
                for(const auto upstream :
                    graph.coordinates.getInputNodeIndices(tag, isEdge<Duplicate>))
                {
                    CAPTURE(upstream);
                    CHECK(countCoordinateWritesInLoop(graph, kLoop, upstream, tracer) == 16);
                    CHECK(countCoordinateWritesInLoop(graph, kLoopTail, upstream, tracer) == 8);
                    didACheck = true;
                }
                break; // second encountered macro tile and beyond are not used in kloop[tail]
            }
            CHECK(didACheck);
        }

        {
            bool didACheck = false;
            for(const auto& c : graph.mapper.getConnections(kLoop))
            {
                CAPTURE(c.control);
                if(std::holds_alternative<Connections::JustNaryArgument>(c.connection))
                {
                    CAPTURE(c.coordinate);
                    // KLoop's for loop variable is only written in the KLoop, not in KLoopTail
                    CHECK(countCoordinateWritesInLoop(graph, kLoop, c.coordinate, tracer) == 1);
                    CHECK(countCoordinateWritesInLoop(graph, kLoopTail, c.coordinate, tracer) == 0);
                    didACheck = true;
                }
            }
            CHECK(didACheck);
        }
    }

    SECTION("valid hoist")
    {
        // Add assign node to kloop body that writes a constant to a new coordinate
        // in the body of the kloop
        // Verify that this node is hoisted to above the kloop

        int coord = graph.coordinates.addElement(Linear{});

        auto constantExpr = Expression::literal(42.0f);

        Assign assignOp;
        assignOp.expression = constantExpr;

        int assign = graph.control.addElement(assignOp);

        graph.mapper.connect(assign, coord, NaryArgument::DEST);

        const auto firstDownstreamNode
            = graph.control.getOutputNodeIndices<Body>(kLoop).take(1).only().value();
        AssertFatal(firstDownstreamNode != -1, ShowValue(firstDownstreamNode));
        insertBefore(graph, firstDownstreamNode, assign, assign);

        CHECK(countCoordinateWritesInLoop(graph, kLoop, coord, ControlFlowRWTracer(graph)) == 1);

        graph = transform<HoistLoopInvariant>(graph);

        CHECK(countCoordinateWritesInLoop(graph, kLoop, coord, ControlFlowRWTracer(graph)) == 0);

        int newAssign;
        {
            const auto connections = graph.mapper.getCoordinateConnections(coord);
            REQUIRE(connections.size() == 1);
            newAssign = connections[0].control;
        }

        {
            const auto loopStack   = controlStack(kLoop, graph);
            const auto assignStack = controlStack(newAssign, graph);

            REQUIRE(loopStack.size() == assignStack.size());

            // Expect same control stack excluding itself
            for(size_t i = 0; i < loopStack.size() - 1; ++i)
            {
                CHECK(loopStack[i] == assignStack[i]);
            }
        }
    }

    SECTION("no hoist incremented variable")
    {
        // Add assign node to kloop body that writes to incremented for loop variable
        // Verify that this node is *not* hoisted

        int forLoopCoordinate = -1;
        for(const auto& c : graph.mapper.getConnections(kLoop))
        {
            CAPTURE(c.control);
            if(std::holds_alternative<Connections::JustNaryArgument>(c.connection))
            {
                forLoopCoordinate = c.coordinate;
                break;
            }
        }
        AssertFatal(forLoopCoordinate != -1, "Could not find for loop coordinate");

        Assign assignOp;
        assignOp.expression = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{forLoopCoordinate, Register::Type::Scalar, DataType::UInt32});

        int assign = graph.control.addElement(assignOp);

        graph.mapper.connect(assign, forLoopCoordinate, NaryArgument::DEST);

        const auto firstDownstreamNode
            = graph.control.getOutputNodeIndices<Body>(kLoop).take(1).only().value();

        if(firstDownstreamNode != -1)
        {
            insertBefore(graph, firstDownstreamNode, assign, assign);
        }

        CHECK(
            countCoordinateWritesInLoop(graph, kLoop, forLoopCoordinate, ControlFlowRWTracer(graph))
            >= 1);
        graph = transform<HoistLoopInvariant>(graph);
        CHECK(
            countCoordinateWritesInLoop(graph, kLoop, forLoopCoordinate, ControlFlowRWTracer(graph))
            >= 1);

        {
            const auto loopStack   = controlStack(kLoop, graph);
            const auto assignStack = controlStack(assign, graph);

            REQUIRE(loopStack.size() + 1 == assignStack.size());

            // Assign stack has one more element which is itself
            for(size_t i = 0; i < loopStack.size(); ++i)
            {
                CHECK(loopStack[i] == assignStack[i]);
            }
            CHECK(assignStack.back() == assign);
        }
    }

    SECTION("no hoist when coordinate used after loop")
    {
        // Avoid hoisting when assigned-to coordinate is used after the loop,
        // for the case where loop is zero-iteration

        int coord = graph.coordinates.addElement(Linear{});

        auto   constantExpr = Expression::literal(42.0f);
        Assign assignInLoop;
        assignInLoop.expression = constantExpr;
        int assignInLoopNode    = graph.control.addElement(assignInLoop);
        graph.mapper.connect(assignInLoopNode, coord, NaryArgument::DEST);

        const auto firstDownstreamNode
            = graph.control.getOutputNodeIndices<Body>(kLoop).take(1).only().value();
        if(firstDownstreamNode != -1)
        {
            insertBefore(graph, firstDownstreamNode, assignInLoopNode, assignInLoopNode);
        }

        auto useExpr = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{coord, Register::Type::Scalar, DataType::Float});
        Assign assignAfterLoop;
        assignAfterLoop.expression = useExpr;
        int assignAfterLoopNode    = graph.control.addElement(assignAfterLoop);

        int resultCoord = graph.coordinates.addElement(Linear{});
        graph.mapper.connect(assignAfterLoopNode, resultCoord, NaryArgument::DEST);

        const auto nodeAfterLoop = graph.control.nodesAfter(kLoop).take(1).only().value();
        AssertFatal(nodeAfterLoop != -1, ShowValue(nodeAfterLoop));

        insertBefore(graph, nodeAfterLoop, assignAfterLoopNode, assignAfterLoopNode);
        CHECK(countCoordinateWritesInLoop(graph, kLoop, coord, ControlFlowRWTracer(graph)) == 1);

        graph = transform<HoistLoopInvariant>(graph);

        CHECK(countCoordinateWritesInLoop(graph, kLoop, coord, ControlFlowRWTracer(graph)) == 1);

        {
            const auto loopStack   = controlStack(kLoop, graph);
            const auto assignStack = controlStack(assignInLoopNode, graph);

            REQUIRE(loopStack.size() + 1 == assignStack.size());

            for(size_t i = 0; i < loopStack.size(); ++i)
            {
                CHECK(loopStack[i] == assignStack[i]);
            }
            CHECK(assignStack.back() == assignInLoopNode);
        }
    }

    SECTION("read-before-write prevents hoisting")
    {
        // Verify that when a coordinate is read before being written in a loop,
        // the write cannot be hoisted (as it would change the first iteration's read value)
        int coord     = graph.coordinates.addElement(Linear{});
        int tempCoord = graph.coordinates.addElement(Linear{});

        Assign readNode;
        readNode.expression = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{coord, Register::Type::Scalar, DataType::Float});
        int readNodeIdx = graph.control.addElement(readNode);
        graph.mapper.connect(readNodeIdx, tempCoord, NaryArgument::DEST);

        Assign writeNode;
        writeNode.expression = Expression::literal(5.0f);
        int writeNodeIdx     = graph.control.addElement(writeNode);
        graph.mapper.connect(writeNodeIdx, coord, NaryArgument::DEST);

        const auto firstNode
            = graph.control.getOutputNodeIndices<Body>(kLoop).take(1).only().value();
        insertBefore(graph, firstNode, readNodeIdx, readNodeIdx);
        insertAfter(graph, readNodeIdx, writeNodeIdx, writeNodeIdx);

        CHECK(countCoordinateWritesInLoop(graph, kLoop, coord, ControlFlowRWTracer(graph)) == 1);
        graph = transform<HoistLoopInvariant>(graph);
        CHECK(countCoordinateWritesInLoop(graph, kLoop, coord, ControlFlowRWTracer(graph)) == 1);

        const auto writeStack = controlStack(writeNodeIdx, graph);
        CHECK(std::find(writeStack.begin(), writeStack.end(), kLoop) != writeStack.end());
    }

    SECTION("write-before-read allows hoisting")
    {
        // Verify that when a coordinate is written before being read in a loop,
        // the write can be hoisted (as the read always sees the written value)
        int coord     = graph.coordinates.addElement(Linear{});
        int tempCoord = graph.coordinates.addElement(Linear{});

        Assign writeNode;
        writeNode.expression = Expression::literal(5.0f);
        int writeNodeIdx     = graph.control.addElement(writeNode);
        graph.mapper.connect(writeNodeIdx, coord, NaryArgument::DEST);

        Assign readNode;
        readNode.expression = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{coord, Register::Type::Scalar, DataType::Float});
        int readNodeIdx = graph.control.addElement(readNode);
        graph.mapper.connect(readNodeIdx, tempCoord, NaryArgument::DEST);

        const auto firstNode
            = graph.control.getOutputNodeIndices<Body>(kLoop).take(1).only().value();
        insertBefore(graph, firstNode, writeNodeIdx, writeNodeIdx);
        insertAfter(graph, writeNodeIdx, readNodeIdx, readNodeIdx);

        CHECK(countCoordinateWritesInLoop(graph, kLoop, coord, ControlFlowRWTracer(graph)) == 1);
        graph = transform<HoistLoopInvariant>(graph);
        CHECK(countCoordinateWritesInLoop(graph, kLoop, coord, ControlFlowRWTracer(graph)) == 0);

        int newWriteNode = -1;
        for(const auto& c : graph.mapper.getCoordinateConnections(coord))
        {
            if(graph.control.get<Assign>(c.control).has_value())
            {
                newWriteNode = c.control;
                break;
            }
        }
        REQUIRE(newWriteNode != -1);

        const auto writeStack = controlStack(newWriteNode, graph);
        const auto loopStack  = controlStack(kLoop, graph);
        CHECK(std::find(writeStack.begin(), writeStack.end(), kLoop) == writeStack.end());
        REQUIRE(loopStack.size() == writeStack.size());
        for(size_t i = 0; i < loopStack.size() - 1; ++i)
        {
            CHECK(loopStack[i] == writeStack[i]);
        }
    }
}
