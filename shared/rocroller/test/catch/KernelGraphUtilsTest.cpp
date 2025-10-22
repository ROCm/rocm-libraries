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
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Utilities/Generator.hpp>

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
