// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
