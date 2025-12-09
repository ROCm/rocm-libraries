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

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDSPadding.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDSPadding_detail.hpp>
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

namespace AddPrefetchTest
{
    using namespace rocRoller;
    using namespace rocRoller::KernelGraph::CoordinateGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;
    using namespace rocRoller::KernelGraph::AddLDSPaddingDetail;

    TEST_CASE("computeDefaultLDSPaddingElements", "[kernel-graph][utils]")
    {
        // Expected padding depends on both data type and fast dimension size
        // Padding is only added when banksTouched % ldsBankCount == 0
        std::map<std::pair<DataType, uint>, uint> expected = {
            // Float (32 bits): needs sizeF*32 bits to be a multiple of 32 banks * 32 bits = 1024 bits
            {{DataType::Float, 11u},
             0u}, // 11*32 = 352 bits -> 11 banks touched (not multiple of 32)
            {{DataType::Float, 16u},
             0u}, // 16*32 = 512 bits -> 16 banks touched (not multiple of 32)
            {{DataType::Float, 17u},
             0u}, // 17*32 = 544 bits -> 17 banks touched (not multiple of 32)
            {{DataType::Float, 32u},
             1u}, // 32*32 = 1024 bits -> 32 banks touched -> padding = 32/32 = 1
            {{DataType::Float, 64u},
             1u}, // 64*32 = 2048 bits -> 64 banks touched -> padding = 32/32 = 1
            {{DataType::Float, 128u}, 1u}, // 128*32 = 4096 bits -> 128 banks touched -> padding = 1
            {{DataType::Float, 256u}, 1u}, // 256*32 = 8192 bits -> 256 banks touched -> padding = 1
            {{DataType::Float, 264u},
             0u}, // 264*32 = 8448 bits -> 264 banks touched (264%32=8, not multiple)

            // FP8 (8 bits): needs sizeF*8 bits to be a multiple of 1024 bits
            {{DataType::FP8, 11u}, 0u}, // 11*8 = 88 bits -> 3 banks touched (not multiple of 32)
            {{DataType::FP8, 16u}, 0u}, // 16*8 = 128 bits -> 4 banks touched (not multiple of 32)
            {{DataType::FP8, 17u}, 0u}, // 17*8 = 136 bits -> 5 banks touched (not multiple of 32)
            {{DataType::FP8, 32u}, 0u}, // 32*8 = 256 bits -> 8 banks touched (not multiple of 32)
            {{DataType::FP8, 64u}, 0u}, // 64*8 = 512 bits -> 16 banks touched (not multiple of 32)
            {{DataType::FP8, 128u},
             4u}, // 128*8 = 1024 bits -> 32 banks touched -> padding = 32/8 = 4
            {{DataType::FP8, 256u}, 4u}, // 256*8 = 2048 bits -> 64 banks touched -> padding = 4
            {{DataType::FP8, 264u},
             0u}, // 264*8 = 2112 bits -> 66 banks touched (66%32=2, not multiple)

            // FP6 (6 bits): handled as special case, elementBits % 4 != 0 -> always 0
            {{DataType::FP6, 11u}, 0u},
            {{DataType::FP6, 16u}, 0u},
            {{DataType::FP6, 17u}, 0u},
            {{DataType::FP6, 32u}, 0u},
            {{DataType::FP6, 64u}, 0u},
            {{DataType::FP6, 128u}, 0u},
            {{DataType::FP6, 256u}, 0u},
            {{DataType::FP6, 264u}, 0u},

            // FP4 (4 bits): needs sizeF*4 bits to be a multiple of 1024 bits
            {{DataType::FP4, 11u}, 0u}, // 11*4 = 44 bits -> 2 banks touched (not multiple of 32)
            {{DataType::FP4, 16u}, 0u}, // 16*4 = 64 bits -> 2 banks touched (not multiple of 32)
            {{DataType::FP4, 17u}, 0u}, // 17*4 = 68 bits -> 3 banks touched (not multiple of 32)
            {{DataType::FP4, 32u}, 0u}, // 32*4 = 128 bits -> 4 banks touched (not multiple of 32)
            {{DataType::FP4, 64u}, 0u}, // 64*4 = 256 bits -> 8 banks touched (not multiple of 32)
            {{DataType::FP4, 128u},
             0u}, // 128*4 = 512 bits -> 16 banks touched (not multiple of 32)
            {{DataType::FP4, 256u},
             8u}, // 256*4 = 1024 bits -> 32 banks touched -> padding = 32/4 = 8
            {{DataType::FP4, 264u},
             0u}, // 264*4 = 1056 bits -> 33 banks touched (33%32=1, not multiple)
        };

        uint sizeS = 33u;
        uint sizeF = GENERATE(11u, 16u, 17u, 32u, 64u, 128u, 256u, 264u);

        KernelGraph::KernelGraph graph;

        auto ldsTag    = graph.coordinates.addElement(LDS());
        auto upstreamS = graph.coordinates.addElement(
            MacroTileIndex(0, rocRoller::Expression::literal(sizeS), nullptr));
        auto upstreamF = graph.coordinates.addElement(
            MacroTileIndex(1, rocRoller::Expression::literal(sizeF), nullptr));

        auto flatten = graph.coordinates.addElement(Flatten(), {upstreamS, upstreamF}, {ldsTag});

        auto dataType = GENERATE(DataType::Float, DataType::FP8, DataType::FP6, DataType::FP4);

        LDSPaddingInfo info{ldsTag, // ldsTag
                            flatten, // upstreamEdge
                            0, // downstreamEdge
                            {upstreamS, upstreamF}, // upstreamTags
                            {0, 0}, // downstreamTags
                            dataType, // DataType
                            LayoutType::MATRIX_A}; // LayoutType

        auto padding = computeDefaultLDSPaddingElements(graph, info, nullptr);

        CHECK(padding == expected[std::make_pair(dataType, sizeF)]);
    }

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

            int ldsElements = getNumLDSElements(graph, -1, ldsTag);
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

            int ldsElements = getNumLDSElements(graph, -1, ldsTag);
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

            int ldsElements = getNumLDSElements(graph, -1, ldsTag);
            CHECK(ldsElements == strideY * sizeY);
        }
    }
}
