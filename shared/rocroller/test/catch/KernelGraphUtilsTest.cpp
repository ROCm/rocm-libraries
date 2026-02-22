// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "TestContext.hpp"

TEST_CASE("Replace tile", "[kernel-graph]")
{
    using namespace rocRoller;
    namespace CF = rocRoller::KernelGraph::ControlGraph;
    namespace CT = rocRoller::KernelGraph::CoordinateGraph;

    // Create Dataflow in Coordinate Graph
    auto graph = KernelGraph::KernelGraph();

    auto tileA = graph.coordinates.addElement(CT::MacroTile());
    auto tileB = graph.coordinates.addElement(CT::MacroTile());
    auto tileC = graph.coordinates.addElement(CT::MacroTile());

    auto user = graph.coordinates.addElement(CT::User());

    graph.coordinates.addElement(CT::DataFlow(), {tileA, tileB}, {tileC});
    graph.coordinates.addElement(CT::DataFlow(), {tileC}, {user});

    // Create Control Graph
    auto exprA = std::make_shared<Expression::Expression>(
        Expression::DataFlowTag{tileA, Register::Type::Vector, DataType::UInt32});

    auto exprB = std::make_shared<Expression::Expression>(
        Expression::DataFlowTag{tileB, Register::Type::Vector, DataType::UInt32});

    auto assign = graph.control.addElement(CF::Assign{Register::Type::Vector, exprA + exprB});
    graph.mapper.connect(assign, tileC, NaryArgument::DEST);

    auto store = graph.control.addElement(CF::StoreTiled());
    graph.mapper.connect<CT::User>(store, user);
    graph.mapper.connect<CT::MacroTile>(store, tileC);

    graph.control.addElement(CF::Sequence(), {assign}, {store});

    // Perform replaceMacroTile

    auto newTile1 = graph.coordinates.addElement(CT::MacroTile());

    replaceMacroTile(graph, {assign, store}, tileA, newTile1);

    auto tileCParents
        = graph.coordinates.getInputNodeIndices(tileC, CT::isEdge<CT::DataFlow>).to<std::set>();
    auto tileCChildren
        = graph.coordinates.getOutputNodeIndices(tileC, CT::isEdge<CT::DataFlow>).to<std::set>();
    CHECK(tileCParents.size() == 2);
    CHECK(tileCParents.count(newTile1) == 1);
    CHECK(tileCParents.count(tileA) == 0);
    CHECK(tileCChildren.size() == 1);
    CHECK(tileCChildren.count(user) == 1);
    CHECK(graph.mapper.get<CT::MacroTile>(store) == tileC);
    CHECK(only(graph.mapper.getConnections(assign))->coordinate == tileC);

    auto newTile2 = graph.coordinates.addElement(CT::MacroTile());

    replaceMacroTile(graph, {assign, store}, tileC, newTile2);
    tileCParents
        = graph.coordinates.getInputNodeIndices(tileC, CT::isEdge<CT::DataFlow>).to<std::set>();
    tileCChildren
        = graph.coordinates.getOutputNodeIndices(tileC, CT::isEdge<CT::DataFlow>).to<std::set>();
    CHECK(tileCParents.size() == 0);
    CHECK(tileCChildren.size() == 0);

    auto tile2Parents
        = graph.coordinates.getInputNodeIndices(newTile2, CT::isEdge<CT::DataFlow>).to<std::set>();
    auto tile2Children
        = graph.coordinates.getOutputNodeIndices(newTile2, CT::isEdge<CT::DataFlow>).to<std::set>();
    CHECK(tile2Parents.size() == 2);
    CHECK(tile2Parents.count(newTile1) == 1);
    CHECK(tile2Parents.count(tileB) == 1);
    CHECK(tile2Children.size() == 1);
    CHECK(tile2Children.count(user) == 1);
    CHECK(graph.mapper.get<CT::MacroTile>(store) == newTile2);
    CHECK(only(graph.mapper.getConnections(assign))->coordinate == newTile2);
}

TEST_CASE("ForLoop utils", "[kernel-graph]")
{
    using namespace rocRoller;
    namespace CF = rocRoller::KernelGraph::ControlGraph;
    namespace CT = rocRoller::KernelGraph::CoordinateGraph;

    auto graph = KernelGraph::KernelGraph();

    SECTION("Basic rangeFor")
    {
        auto [forLoopCoord, forLoopOp]
            = KernelGraph::rangeFor(graph, Expression::literal(10), "DummyLoop");

        CHECK(forLoopCoord == graph.mapper.get<CT::ForLoop>(forLoopOp));
    }

    SECTION("Basic purgeFor")
    {
        auto [forLoopCoord, forLoopOp]
            = KernelGraph::rangeFor(graph, Expression::literal(10), "DummyLoop");

        // for loop coord, iterator, and dataflow edge
        CHECK(graph.coordinates.allElements().to<std::vector>().size() == 3);
        // for loop op, init, increment nodes and edges
        CHECK(graph.control.allElements().to<std::vector>().size() == 5);

        purgeFor(graph, forLoopOp);

        // Everything should be gone
        CHECK(graph.coordinates.allElements().to<std::vector>().size() == 0);
        CHECK(graph.control.allElements().to<std::vector>().size() == 0);
    }

    SECTION("Shared purgeFor")
    {
        auto [forLoopCoord0, forLoopOp]
            = KernelGraph::rangeFor(graph, Expression::literal(10), "DummyLoop");

        auto [forLoopCoord, forLoopIterator] = getForLoopCoords(forLoopOp, graph);

        CHECK(forLoopCoord == forLoopCoord0);

        // Add a new ForLoopOp that uses the same iterator manually
        auto newForLoopOp = graph.control.addElement(CF::ForLoopOp());
        graph.mapper.connect(newForLoopOp, forLoopIterator, NaryArgument::DEST);
        graph.mapper.connect<CT::ForLoop>(newForLoopOp, forLoopCoord);

        CHECK(graph.coordinates.allElements().to<std::vector>().size() == 3);
        CHECK(graph.control.allElements().to<std::vector>().size() == 6);

        purgeFor(graph, forLoopOp);

        // The for-loop coord and iterator should still exist
        CHECK(graph.coordinates.allElements().to<std::vector>().size() == 3);

        // The newForLoopOp should still exist
        CHECK(graph.control.allElements().to<std::vector>().size() == 1);

        purgeFor(graph, newForLoopOp);

        // Now everything should be gone
        CHECK(graph.coordinates.allElements().to<std::vector>().size() == 0);
        CHECK(graph.control.allElements().to<std::vector>().size() == 0);
    }

    SECTION("Basic cloneForLoop")
    {
        auto [forLoopCoord, forLoopOp]
            = KernelGraph::rangeFor(graph, Expression::literal(10), "DummyLoop");

        CHECK(graph.control.allElements().to<std::vector>().size() == 5);
        CHECK(graph.coordinates.allElements().to<std::vector>().size() == 3);

        auto clonedForLoopOp = KernelGraph::cloneForLoop(graph, forLoopOp);

        CHECK(graph.control.allElements().to<std::vector>().size() == 10);
        // The new loop re-uses the ForLoop coordinate
        CHECK(graph.coordinates.allElements().to<std::vector>().size() == 5);
    }

    SECTION("Complex cloneForLoop")
    {
        // Manually make a ForLoop, where the condition doesn't
        // exactly match the size of the ForLoop dimension (like will
        // happen in StreamK kernels).

        auto [forLoopCoordTag, forLoopOpTag]
            = KernelGraph::rangeFor(graph, Expression::literal(10u), "DummyLoop");

        auto [_ignore, forLoopIteratorTag] = getForLoopCoords(forLoopOpTag, graph);

        auto forLoopCoord = graph.coordinates.get<CT::ForLoop>(forLoopCoordTag).value();

        auto forLoopOp = graph.control.get<CF::ForLoopOp>(forLoopOpTag).value();
        forLoopOp.condition
            = Expression::dataFlowTag(forLoopIteratorTag, Register::Type::Scalar, DataType::UInt32)
              < Expression::literal(15u);
        graph.control.setElement(forLoopOpTag, forLoopOp);

        auto clonedForLoopOpTag = KernelGraph::cloneForLoop(graph, forLoopOpTag);

        auto clonedForLoopOp = graph.control.get<CF::ForLoopOp>(clonedForLoopOpTag).value();

        // The ForLoop coord size should be unchanged, and is 10.
        CHECK(Expression::identical(forLoopCoord.size, Expression::literal(10u)));

        // The cloned ForLoopOp condition should be unchanged AND IS NOT "i < 10".
        auto [_ignore1, originalUpper]
            = Expression::split<Expression::LessThan>(forLoopOp.condition);
        auto [_ignore2, clonedUpper]
            = Expression::split<Expression::LessThan>(clonedForLoopOp.condition);

        CHECK(not Expression::identical(clonedUpper, Expression::literal(10u)));
        CHECK(Expression::identical(originalUpper, clonedUpper));
    }
}

TEST_CASE("updateThreadTileForLongDwords", "[kernel-graph]")
{
    using namespace rocRoller::KernelGraph;

    SECTION("Factor 4 is optimal")
    {
        int  t_m                      = 8;
        int  t_n                      = 1;
        int  maxWidth                 = 4;
        uint macTileFastMovingDimSize = 64;
        int  numDwordsPerElement      = 1;
        bool avoidDWordX2             = false;

        updateThreadTileForLongDwords(
            t_m, t_n, maxWidth, macTileFastMovingDimSize, numDwordsPerElement, avoidDWordX2);

        // With factor 4: t_m = 8/4 = 2, t_n = 1*4 = 4
        CHECK(t_m == 2);
        CHECK(t_n == 4);
    }

    SECTION("Factor 3 when 4 doesn't divide evenly")
    {
        int  t_m                      = 6;
        int  t_n                      = 1;
        int  maxWidth                 = 4;
        uint macTileFastMovingDimSize = 64;
        int  numDwordsPerElement      = 1;
        bool avoidDWordX2             = false;

        updateThreadTileForLongDwords(
            t_m, t_n, maxWidth, macTileFastMovingDimSize, numDwordsPerElement, avoidDWordX2);

        // With factor 3: t_m = 6/3 = 2, t_n = 1*3 = 3
        CHECK(t_m == 2);
        CHECK(t_n == 3);
    }

    SECTION("Factor 2 when only 2 divides evenly")
    {
        int  t_m                      = 10;
        int  t_n                      = 1;
        int  maxWidth                 = 4;
        uint macTileFastMovingDimSize = 64;
        int  numDwordsPerElement      = 1;
        bool avoidDWordX2             = false;

        updateThreadTileForLongDwords(
            t_m, t_n, maxWidth, macTileFastMovingDimSize, numDwordsPerElement, avoidDWordX2);

        // With factor 2: t_m = 10/2 = 5, t_n = 1*2 = 2
        CHECK(t_m == 5);
        CHECK(t_n == 2);
    }

    SECTION("avoidDWordX2 skips factor 2")
    {
        int  t_m                      = 10;
        int  t_n                      = 1;
        int  maxWidth                 = 4;
        uint macTileFastMovingDimSize = 64;
        int  numDwordsPerElement      = 1;
        bool avoidDWordX2             = true;

        updateThreadTileForLongDwords(
            t_m, t_n, maxWidth, macTileFastMovingDimSize, numDwordsPerElement, avoidDWordX2);

        // Factor 2 is skipped, only factor 1 works: no change
        CHECK(t_m == 10);
        CHECK(t_n == 1);
    }

    SECTION("macTileFastMovingDimSize limits factor")
    {
        int  t_m                      = 8;
        int  t_n                      = 2;
        int  maxWidth                 = 4;
        uint macTileFastMovingDimSize = 4; // t_n * factor must be <= 4
        int  numDwordsPerElement      = 1;
        bool avoidDWordX2             = false;

        updateThreadTileForLongDwords(
            t_m, t_n, maxWidth, macTileFastMovingDimSize, numDwordsPerElement, avoidDWordX2);

        // Factor 4 would make t_n = 8 > 4, factor 3 would make t_n = 6 > 4
        // Factor 2 works: t_n = 2*2 = 4 <= 4
        CHECK(t_m == 4);
        CHECK(t_n == 4);
    }

    SECTION("maxWidth limits factor")
    {
        int  t_m                      = 8;
        int  t_n                      = 1;
        int  maxWidth                 = 2; // Only factors <= 2 can be used
        uint macTileFastMovingDimSize = 64;
        int  numDwordsPerElement      = 1;
        bool avoidDWordX2             = false;

        updateThreadTileForLongDwords(
            t_m, t_n, maxWidth, macTileFastMovingDimSize, numDwordsPerElement, avoidDWordX2);

        // maxWidth = 2, so only factor 2 or 1 possible
        CHECK(t_m == 4);
        CHECK(t_n == 2);
    }

    SECTION("numDwordsPerElement > 1")
    {
        int  t_m                      = 8;
        int  t_n                      = 1;
        int  maxWidth                 = 4;
        uint macTileFastMovingDimSize = 64;
        int  numDwordsPerElement      = 2; // Each element uses 2 dwords
        bool avoidDWordX2             = false;

        updateThreadTileForLongDwords(
            t_m, t_n, maxWidth, macTileFastMovingDimSize, numDwordsPerElement, avoidDWordX2);

        // numDwordsPerWorkitem = 8 * 2 = 16
        // Factor 4: 16 % 4 == 0, dwordFactor = 4/2 = 2
        // t_m = 8/2 = 4, t_n = 1*2 = 2
        CHECK(t_m == 4);
        CHECK(t_n == 2);
    }

    SECTION("No valid factor - no change")
    {
        int  t_m                      = 7; // Prime number
        int  t_n                      = 1;
        int  maxWidth                 = 4;
        uint macTileFastMovingDimSize = 64;
        int  numDwordsPerElement      = 1;
        bool avoidDWordX2             = false;

        updateThreadTileForLongDwords(
            t_m, t_n, maxWidth, macTileFastMovingDimSize, numDwordsPerElement, avoidDWordX2);

        // 7 is not divisible by 4, 3, or 2, so only factor 1 works
        CHECK(t_m == 7);
        CHECK(t_n == 1);
    }
}

TEST_CASE("createInternalTile", "[kernel-graph]")
{
    using namespace rocRoller;
    namespace CT = rocRoller::KernelGraph::CoordinateGraph;

    auto ctx = TestContext::ForDefaultTarget();

    SECTION("Basic VGPR tile creation")
    {
        auto graph  = KernelGraph::KernelGraph();
        auto params = std::make_shared<CommandParameters>();

        // Create a simple MacroTile 64x64 in VGPR
        auto macroTile    = CT::MacroTile({64, 64}, MemoryType::VGPR, {4, 4});
        auto macroTileTag = graph.coordinates.addElement(macroTile);

        // Create internal tile
        auto internalTag = KernelGraph::createInternalTile(
            graph, DataType::Float, macroTileTag, params, ctx.get());

        CHECK(internalTag != -1);
        auto internalTile = graph.coordinates.getNode<CT::MacroTile>(internalTag);
        CHECK(internalTile.sizes[0] == 64);
        CHECK(internalTile.sizes[1] == 64);
        CHECK(internalTile.memoryType == MemoryType::VGPR);
    }

    SECTION("LDS tile creation")
    {
        auto graph  = KernelGraph::KernelGraph();
        auto params = std::make_shared<CommandParameters>();

        // Create a MacroTile in LDS
        auto macroTile    = CT::MacroTile({64, 64}, MemoryType::LDS, {4, 4});
        auto macroTileTag = graph.coordinates.addElement(macroTile);

        // Create internal tile
        auto internalTag = KernelGraph::createInternalTile(
            graph, DataType::Float, macroTileTag, params, ctx.get());

        CHECK(internalTag != -1);
        auto internalTile = graph.coordinates.getNode<CT::MacroTile>(internalTag);
        CHECK(internalTile.sizes[0] == 64);
        CHECK(internalTile.sizes[1] == 64);
        // LDS tiles should be converted to VGPR
        CHECK(internalTile.memoryType == MemoryType::VGPR);
    }

    SECTION("Tile with numWaveTiles")
    {
        auto graph  = KernelGraph::KernelGraph();
        auto params = std::make_shared<CommandParameters>();

        // Create a MacroTile 128x128 in VGPR
        auto macroTile    = CT::MacroTile({128, 128}, MemoryType::VGPR, {4, 4});
        auto macroTileTag = graph.coordinates.addElement(macroTile);

        std::vector<unsigned int> numWaveTiles = {2, 2};

        // Create internal tile with wave tiles
        auto internalTag = KernelGraph::createInternalTile(
            graph, DataType::Float, macroTileTag, numWaveTiles, false, params, ctx.get());

        CHECK(internalTag != -1);
        auto internalTile = graph.coordinates.getNode<CT::MacroTile>(internalTag);
        // Size should be divided by numWaveTiles
        CHECK(internalTile.sizes[0] == 64);
        CHECK(internalTile.sizes[1] == 64);
        CHECK(internalTile.memoryType == MemoryType::VGPR);
    }
}
