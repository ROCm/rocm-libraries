// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include <common/CommonGraphs.hpp>

#include "TestContext.hpp"

TEST_CASE("Colour by Unroll value", "[kernel-graph]")
{
    using namespace rocRoller::Expression;
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;
    using namespace rocRoller::KernelGraph::CoordinateGraph;

    KernelGraph graph;

    auto unrollX = graph.coordinates.addElement(Unroll());
    auto unrollY = graph.coordinates.addElement(Unroll());
    auto unrollK = graph.coordinates.addElement(Unroll());

    auto userA = graph.coordinates.addElement(User());
    auto userB = graph.coordinates.addElement(User());
    auto tileA = graph.coordinates.addElement(MacroTile());
    auto tileB = graph.coordinates.addElement(MacroTile());
    auto tileD = graph.coordinates.addElement(MacroTile());

    auto DF = [](int tag) {
        return std::make_shared<Expression>(
            DataFlowTag{tag, rocRoller::Register::Type::Vector, {}});
    };

    uint vX = 5u, vY = 7u, vK = 11u;

    auto kernel    = graph.control.addElement(Kernel());
    auto setCoordX = graph.control.addElement(SetCoordinate(literal(vX)));
    auto setCoordY = graph.control.addElement(SetCoordinate(literal(vY)));
    auto setCoordK = graph.control.addElement(SetCoordinate(literal(vK)));

    auto loadA   = graph.control.addElement(LoadTiled());
    auto loadB   = graph.control.addElement(LoadTiled());
    auto assignD = graph.control.addElement(
        Assign(rocRoller::Register::Type::Vector, DF(tileA) + DF(tileB)));

    auto storeD = graph.control.addElement(StoreTiled());

    graph.control.addElement(Body(), {kernel}, {setCoordX});
    graph.control.addElement(Body(), {setCoordX}, {setCoordK});
    graph.control.addElement(Body(), {setCoordK}, {loadA});

    graph.control.addElement(Body(), {kernel}, {setCoordY});
    graph.control.addElement(Body(), {setCoordY}, {loadB});

    auto s1 = graph.control.addElement(Sequence(), {setCoordX}, {assignD});
    auto s2 = graph.control.addElement(Sequence(), {setCoordY}, {assignD});

    graph.control.addElement(Sequence(), {assignD}, {storeD});

    graph.mapper.connect<Unroll>(setCoordX, unrollX);
    graph.mapper.connect<Unroll>(setCoordY, unrollY);
    graph.mapper.connect<Unroll>(setCoordK, unrollK);

    graph.mapper.connect<MacroTile>(loadA, tileA);
    graph.mapper.connect<MacroTile>(loadB, tileB);
    graph.mapper.connect<User>(loadA, userA);
    graph.mapper.connect<User>(loadB, userB);
    graph.mapper.connect<MacroTile>(assignD, tileD);

    graph.mapper.connect<MacroTile>(storeD, tileD);

    /* graph is:
     *
     *                   Kernel
     *                  /     \
     *          SetCoordX     SetCoordY
     *          /       .     .       \
     *  SetCoordK        .   .         LoadTileB
     *      |           AssignD
     *  LoadTileA           .
     *                     StoreTileD
     *
     * where the solid edges are Body, and dotted are Sequence.
     *
     * Note that D is result of A + B.
     */

    auto exclude = GENERATE(false, true);

    auto excludeUnrolls
        = exclude ? std::unordered_set<int>{unrollX, unrollK} : std::unordered_set<int>{};

    auto colouring = colourByUnrollValue(graph, -1, excludeUnrolls);

    // We are relying on the default behaviour of operation[] below:
    // missing entries in a map get default constructed, and therefore
    // missing colours are 0.

    auto cX = exclude ? 0 : vX;
    auto cY = vY;
    auto cK = exclude ? 0 : vK;

    CHECK(colouring.operationColour[setCoordX][unrollX] == cX);
    CHECK(colouring.operationColour[setCoordX][unrollY] == 0);
    CHECK(colouring.operationColour[setCoordX][unrollK] == cK);

    CHECK(colouring.operationColour[setCoordY][unrollX] == 0);
    CHECK(colouring.operationColour[setCoordY][unrollY] == cY);
    CHECK(colouring.operationColour[setCoordY][unrollK] == 0);

    CHECK(colouring.operationColour[setCoordK][unrollX] == cX);
    CHECK(colouring.operationColour[setCoordK][unrollY] == 0);
    CHECK(colouring.operationColour[setCoordK][unrollK] == cK);

    CHECK(colouring.operationColour[loadA][unrollX] == cX);
    CHECK(colouring.operationColour[loadA][unrollY] == 0);
    CHECK(colouring.operationColour[loadA][unrollK] == cK);

    CHECK(colouring.operationColour[loadA][unrollX] == cX);
    CHECK(colouring.operationColour[loadA][unrollY] == 0);
    CHECK(colouring.operationColour[loadA][unrollK] == cK);

    CHECK(colouring.operationColour[loadB][unrollX] == 0);
    CHECK(colouring.operationColour[loadB][unrollY] == cY);
    CHECK(colouring.operationColour[loadB][unrollK] == 0);

    CHECK(colouring.operationColour[assignD][unrollX] == cX);
    CHECK(colouring.operationColour[assignD][unrollY] == cY);
    CHECK(colouring.operationColour[assignD][unrollK] == cK);

    CHECK(colouring.coordinateColour[tileA][unrollX] == cX);
    CHECK(colouring.coordinateColour[tileA][unrollY] == 0);
    CHECK(colouring.coordinateColour[tileA][unrollK] == cK);

    CHECK(colouring.coordinateColour[tileB][unrollX] == 0);
    CHECK(colouring.coordinateColour[tileB][unrollY] == cY);
    CHECK(colouring.coordinateColour[tileB][unrollK] == 0);

    CHECK(colouring.coordinateColour[tileD][unrollX] == cX);
    CHECK(colouring.coordinateColour[tileD][unrollY] == cY);
    CHECK(colouring.coordinateColour[tileD][unrollK] == cK);

    CHECK(colouring.operationColour[storeD][unrollX] == cX);
    CHECK(colouring.operationColour[storeD][unrollY] == cY);
    CHECK(colouring.operationColour[storeD][unrollK] == cK);

    if(not exclude)
    {
        // SetCoordX, SetCoordY, and AssignD all have colours, their
        // colours are different, and they are connected by the s1 and
        // s2 Sequence edges.  To explicitly separate the colours,
        // you'd have to cut s1 and s2.
        CHECK(colouring.separators == std::set<int>{s1, s2});
    }
    else
    {
        // Only Y is coloured, and nothing else has a colour.  There
        // aren't any explicit colour mismatches, and therefore no
        // separators.
        CHECK(colouring.separators == std::set<int>{});
    }
}

TEST_CASE("Colour by NaryArgument - Basic Matrix Multiply", "[kernel-graph]")
{
    using namespace rocRollerTest;
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    using GD = rocRoller::Graph::Direction;

    auto context = TestContext::ForDefaultTarget();

    // Create a simple GEMM without scaling
    auto example = rocRollerTest::Graphs::GEMM(rocRoller::DataType::Float);
    example.setTileSize(64, 64, 64);
    example.setMFMA(32, 32, 2, 1);
    example.setPrefetch(true, 2, 0, false);

    auto graph  = example.getKernelGraph();
    auto params = example.getCommandParameters();

    // Apply necessary transforms to set up the multiply operation
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

    // Call colourByNaryArgument
    auto colouring = colourByNaryArgument(graph);

    CHECK(colouring.coordinateColour.size() > 0);
    CHECK(colouring.operationColour.size() > 0);

    auto forKLoopPredicate = [&](int tag) -> bool {
        auto maybeForLoop = graph.control.get<ForLoopOp>(tag);
        if(!maybeForLoop)
            return false;
        return maybeForLoop->loopName == rocRoller::KLOOP;
    };

    auto kernel  = *only(graph.control.roots());
    auto forLoop = *only(graph.control.findNodes(kernel, forKLoopPredicate, GD::Downstream));
    auto bodies  = graph.control.getOutputNodeIndices<Body>(forLoop).to<std::set>();

    auto shouldHaveColourPredicate = [&](int tag) -> bool {
        return graph.control.get<LoadTiled>(tag).has_value()
               || graph.control.get<LoadLDSTile>(tag).has_value()
               || graph.control.get<StoreLDSTile>(tag).has_value()
               || graph.control.get<SetCoordinate>(tag).has_value();
    };

    auto interestingNodes
        = filter(shouldHaveColourPredicate, graph.control.depthFirstVisit(bodies, GD::Downstream))
              .to<std::vector>();
    CHECK(interestingNodes.size() > 0);
    CHECK(std::all_of(interestingNodes.begin(), interestingNodes.end(), [&](auto tag) {
        return colouring.operationColour.contains(tag);
    }));
}

TEST_CASE("Colour by NaryArgument - Matrix Multiply with Scaling", "[kernel-graph]")
{
    using namespace rocRollerTest;
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    using GD = rocRoller::Graph::Direction;

    auto context = TestContext::ForDefaultTarget();

    // Create a GEMM with scaling enabled
    auto example = rocRollerTest::Graphs::GEMM(rocRoller::DataType::FP4,
                                               rocRoller::DataType::FP4,
                                               rocRoller::DataType::Float,
                                               rocRoller::DataType::Float);
    example.setTileSize(256, 256, 128);
    example.setUseLDS(true, true, false);
    example.setMFMA(32, 32, 64, 1);
    example.setTranspose("T", "N");
    example.setPrefetch(true, 2, 0, false);
    example.setScaling(rocRoller::Operations::ScaleMode::Separate,
                       rocRoller::Operations::ScaleMode::Separate,
                       rocRoller::DataType::E8M0,
                       rocRoller::DataType::E8M0,
                       32);
    example.setSwizzle(64, 64, 4, 1, true);

    auto graph  = example.getKernelGraph();
    auto params = example.getCommandParameters();

    // Apply necessary transforms to set up the multiply operation
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
    graph = transform<SwizzleScale>(graph, params, context.get());
    graph = transform<AddPrefetch>(graph, params, context.get());
    graph = transform<PrefetchScale>(graph, params, context.get());

    auto colouring = colourByNaryArgument(graph);

    CHECK(colouring.coordinateColour.size() > 0);
    CHECK(colouring.operationColour.size() > 0);

    auto forKLoopPredicate = [&](int tag) -> bool {
        auto maybeForLoop = graph.control.get<ForLoopOp>(tag);
        if(!maybeForLoop)
            return false;
        return maybeForLoop->loopName == rocRoller::KLOOP;
    };

    auto kernel  = *only(graph.control.roots());
    auto forLoop = *only(graph.control.findNodes(kernel, forKLoopPredicate, GD::Downstream));
    auto bodies  = graph.control.getOutputNodeIndices<Body>(forLoop).to<std::set>();

    // Make sure there are Exchange operations in the graph
    auto exchangeNodes = filter(graph.control.isElemType<Exchange>(),
                                graph.control.depthFirstVisit(bodies, GD::Downstream))
                             .to<std::vector>();
    CHECK(exchangeNodes.size() > 0);

    // Make sure everything that everything that should have a colour does
    auto shouldHaveColourPredicate = [&](int tag) -> bool {
        return graph.control.get<LoadTiled>(tag).has_value()
               || graph.control.get<LoadLDSTile>(tag).has_value()
               || graph.control.get<StoreLDSTile>(tag).has_value()
               || graph.control.get<SetCoordinate>(tag).has_value()
               || graph.control.get<Exchange>(tag).has_value();
    };

    auto interestingNodes
        = filter(shouldHaveColourPredicate, graph.control.depthFirstVisit(bodies, GD::Downstream))
              .to<std::vector>();
    CHECK(interestingNodes.size() > 0);
    CHECK(std::all_of(interestingNodes.begin(), interestingNodes.end(), [&](auto tag) {
        return colouring.operationColour.contains(tag);
    }));
}
