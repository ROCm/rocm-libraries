// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDSPadding.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include "TestContext.hpp"

namespace rocRoller::KernelGraph
{
    void addLoadWaveTileCT(KernelGraph&                     graph,
                           std::vector<DeferredConnection>& connections,
                           int                              macTileTag,
                           int                              iMacX,
                           int                              iMacY,
                           DataType const&                  dataType,
                           int                              wavefrontSize,
                           bool                             isFromLDS,
                           std::vector<unsigned int> const& jammedTiles,
                           CommandParametersPtr             params,
                           ContextPtr                       context);
};

namespace AddLDSPaddingTest
{
    using namespace rocRoller;
    using namespace rocRoller::KernelGraph::CoordinateGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    TEST_CASE("getNumLDSElements", "[kernel-graph][utils]")
    {
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::Expression;

        SECTION("Simple flatten")
        {
            rocRoller::KernelGraph::KernelGraph graph;

            uint sizeX = 5u;
            uint sizeY = 7u;

            auto indexX = graph.coordinates.addElement(MacroTileIndex(0, literal(sizeX), nullptr));
            auto indexY = graph.coordinates.addElement(MacroTileIndex(1, literal(sizeY), nullptr));

            auto ldsTag = graph.coordinates.addElement(LDS());

            auto flatten = graph.coordinates.addElement(Flatten(), {indexX, indexY}, {ldsTag});

            int ldsElements = GetNumLDSElements(graph, ldsTag);
            CHECK(ldsElements == sizeX * sizeY);
        }

        SECTION("Joined LDS (X)")
        {
            rocRoller::KernelGraph::KernelGraph graph;

            uint sizeX   = 5u;
            uint sizeY   = 7u;
            uint strideX = GENERATE(7u, 10u);
            uint strideY = 1u;

            auto indexX
                = graph.coordinates.addElement(MacroTileIndex(0, literal(sizeX), literal(strideX)));
            auto indexY
                = graph.coordinates.addElement(MacroTileIndex(1, literal(sizeY), literal(strideY)));

            auto ldsTag = graph.coordinates.addElement(LDS());

            auto join = graph.coordinates.addElement(Join(), {indexX, indexY}, {ldsTag});

            int ldsElements = GetNumLDSElements(graph, ldsTag);
            CHECK(ldsElements == strideX * sizeX);
        }

        SECTION("Joined LDS (Y)")
        {
            rocRoller::KernelGraph::KernelGraph graph;

            uint sizeX   = 5u;
            uint sizeY   = 7u;
            uint strideX = 1u;
            uint strideY = GENERATE(5u, 11u);

            auto indexX
                = graph.coordinates.addElement(MacroTileIndex(0, literal(sizeX), literal(strideX)));
            auto indexY
                = graph.coordinates.addElement(MacroTileIndex(1, literal(sizeY), literal(strideY)));

            auto ldsTag = graph.coordinates.addElement(LDS());

            auto join = graph.coordinates.addElement(Join(), {indexX, indexY}, {ldsTag});

            int ldsElements = GetNumLDSElements(graph, ldsTag);
            CHECK(ldsElements == strideY * sizeY);
        }
    }
}
