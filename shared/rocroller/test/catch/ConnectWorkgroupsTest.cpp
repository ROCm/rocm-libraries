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
#include <catch2/generators/catch_generators_range.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/Transforms/ConnectWorkgroups.hpp>
#include <rocRoller/KernelGraph/Transforms/ConnectWorkgroups_detail.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "CustomMatchers.hpp"
#include "ExpressionMatchers.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"
#include "common/Utilities.hpp"

namespace ConnectWorkgroupsTest
{
    TEST_CASE("ConnectWorkgroups", "[kernel-graph]")
    {
        using namespace rocRoller::Expression;
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::CoordinateGraph;
        using namespace rocRoller::KernelGraph::ConnectWorkgroupsDetail;

        using GD = rocRoller::Graph::Direction;

        KernelGraph graph0;

        int vX = 5, vY = 7;

        auto tileNumAD = graph0.coordinates.addElement(MacroTileNumber(0, literal(vX), nullptr));
        auto tileNumBD = graph0.coordinates.addElement(MacroTileNumber(1, literal(vY), nullptr));

        auto middleLinear = graph0.coordinates.addElement(Linear());

        graph0.coordinates.addElement(Flatten(), {tileNumAD, tileNumBD}, {middleLinear});

        auto tileNumAU = graph0.coordinates.addElement(MacroTileNumber(0, literal(vX), nullptr));
        auto tileNumBU = graph0.coordinates.addElement(MacroTileNumber(1, literal(vY), nullptr));

        graph0.coordinates.addElement(Tile(), {middleLinear}, {tileNumAU, tileNumBU});

        /* coordinate graph is:
         *
         *    MacroTileNumber(0, size=vX)        MacroTileNumber(1, size=vY)
         *             \                                  /
         *              ------------- Flatten ------------
         *                               |
         *                             Linear
         *                               |
         *              --------------- Tile -------------
         *             /                                                \
         *    MacroTileNumber(0, size=vX)        MacroTileNumber(1, size=vY)
         *
         */

        auto graph     = graph0;
        auto remapping = connectWorkgroups(graph);

        for(uint wg = 0; wg < vX; ++wg)
        {
            auto exprs = graph.coordinates.forward(
                {literal(wg)}, {remapping[{0, GD::Downstream}]}, {tileNumAD});
            auto tileNumA = getUnsignedInt(evaluate(exprs[0]));

            auto expectedA = wg;
            CHECK(tileNumA == expectedA);
        }

        for(uint wg = 0; wg < vY; ++wg)
        {
            auto exprs = graph.coordinates.forward(
                {literal(wg)}, {remapping[{1, GD::Downstream}]}, {tileNumBD});
            auto tileNumB = getUnsignedInt(evaluate(exprs[0]));

            auto expectedB = wg;
            CHECK(tileNumB == expectedB);
        }
    }
}
