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

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDSPadding.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDSPadding_detail.hpp>

namespace AddPrefetchTest
{
    using namespace rocRoller;
    using namespace rocRoller::KernelGraph::CoordinateGraph;
    using namespace rocRoller::KernelGraph::AddLDSPaddingDetail;

    TEST_CASE("AddLDSPadding", "[kernel-graph][graph-transforms]")
    {
        // FP6 is handled as a special case
        std::map<std::pair<DataType, LayoutType>, uint> expected
            = {{{DataType::Float, LayoutType::MATRIX_A}, 1u},
               {{DataType::Float, LayoutType::MATRIX_B}, 1u},
               {{DataType::FP8, LayoutType::MATRIX_A}, 4u},
               {{DataType::FP8, LayoutType::MATRIX_B}, 4u},
               {{DataType::FP6, LayoutType::MATRIX_A}, 0u},
               {{DataType::FP6, LayoutType::MATRIX_B}, 0u},
               {{DataType::FP4, LayoutType::MATRIX_A}, 8u},
               {{DataType::FP4, LayoutType::MATRIX_B}, 8u}};

        auto dataType   = GENERATE(DataType::Float, DataType::FP8, DataType::FP6, DataType::FP4);
        auto layoutType = GENERATE(LayoutType::MATRIX_A, LayoutType::MATRIX_B);

        KernelGraph::KernelGraph graph;

        LDSPaddingInfo info{0, // ldsTag
                            0, // upstreamEdge
                            0, // downstreamEdge
                            {0, 0}, // upstreamTags
                            {0, 0}, // downstreamTags
                            dataType,
                            layoutType};

        auto padding = computeDefaultLDSPaddingElements(graph, info);
        auto key     = std::make_pair(dataType, layoutType);

        // Only CHECK when have an expected value
        if(expected.contains(key))
        {
            CHECK(padding == expected[key]);
        }
    }
}
