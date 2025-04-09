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

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

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
