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

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include "../GenericContextFixture.hpp"

namespace KernelGraphTest
{
    using namespace rocRoller;
    namespace CF = rocRoller::KernelGraph::ControlGraph;
    namespace CT = rocRoller::KernelGraph::CoordinateGraph;

    class KernelGraphUtilsTest : public GenericContextFixture
    {
    };

    TEST_F(KernelGraphUtilsTest, ReplaceMacroTile)
    {
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
        auto tileCChildren = graph.coordinates.getOutputNodeIndices(tileC, CT::isEdge<CT::DataFlow>)
                                 .to<std::set>();
        EXPECT_EQ(tileCParents.size(), 2);
        EXPECT_TRUE(tileCParents.count(newTile1) == 1);
        EXPECT_TRUE(tileCParents.count(tileA) == 0);
        EXPECT_EQ(tileCChildren.size(), 1);
        EXPECT_TRUE(tileCChildren.count(user) == 1);
        EXPECT_EQ(graph.mapper.get<CT::MacroTile>(store), tileC);
        EXPECT_EQ(only(graph.mapper.getConnections(assign))->coordinate, tileC);

        auto newTile2 = graph.coordinates.addElement(CT::MacroTile());

        replaceMacroTile(graph, {assign, store}, tileC, newTile2);
        tileCParents
            = graph.coordinates.getInputNodeIndices(tileC, CT::isEdge<CT::DataFlow>).to<std::set>();
        tileCChildren = graph.coordinates.getOutputNodeIndices(tileC, CT::isEdge<CT::DataFlow>)
                            .to<std::set>();
        EXPECT_EQ(tileCParents.size(), 0);
        EXPECT_EQ(tileCChildren.size(), 0);

        auto tile2Parents
            = graph.coordinates.getInputNodeIndices(newTile2, CT::isEdge<CT::DataFlow>)
                  .to<std::set>();
        auto tile2Children
            = graph.coordinates.getOutputNodeIndices(newTile2, CT::isEdge<CT::DataFlow>)
                  .to<std::set>();
        EXPECT_EQ(tile2Parents.size(), 2);
        EXPECT_TRUE(tile2Parents.count(newTile1) == 1);
        EXPECT_TRUE(tile2Parents.count(tileB) == 1);
        EXPECT_EQ(tile2Children.size(), 1);
        EXPECT_TRUE(tile2Children.count(user) == 1);
        EXPECT_EQ(graph.mapper.get<CT::MacroTile>(store), newTile2);
        EXPECT_EQ(only(graph.mapper.getConnections(assign))->coordinate, newTile2);
    }
}
