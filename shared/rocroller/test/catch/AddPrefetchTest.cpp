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

#include <rocRoller/KernelGraph/Transforms/All.hpp>

#include <rocRoller/KernelGraph/Transforms/AddPrefetch_detail.hpp>

#include "TestContext.hpp"
#include "common/CommonGraphs.hpp"
#include "common/Utilities.hpp"

namespace AddPrefetchTest
{
    TEST_CASE("AddPrefetch", "[kernel-graph][graph-transforms]")
    {
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::AddPrefetchDetail;

        auto context = TestContext::ForDefaultTarget();

        auto typeA   = rocRoller::DataType::Float;
        auto example = rocRollerTest::Graphs::GEMM(typeA);

        int  prefetchInFlight  = GENERATE(1, 2, 4);
        int  prefetchLDSFactor = GENERATE(0, 1, 2);
        bool prefetchMixMemOps = GENERATE(false, true);

        example.setTileSize(256, 64, 16);
        example.setMFMA(32, 32, 2, 1);
        example.setUseLDS(true, true, false);
        example.setUnroll(2, 2);
        example.setPrefetch(true, prefetchInFlight, prefetchLDSFactor, prefetchMixMemOps);

        auto graph  = example.getKernelGraph();
        auto params = example.getCommandParameters();

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
        graph = transform<ConnectWorkgroups>(
            graph, context.get(), params->workgroupMappingDim, params->workgroupRemapXCC);
        graph = transform<UnrollLoops>(graph, params, context.get());
        graph = transform<FuseLoops>(graph);
        graph = transform<RemoveDuplicates>(graph);
        graph = transform<OrderEpilogueBlocks>(graph);
        graph = transform<CleanLoops>(graph);
        graph = transform<AddPrefetch>(graph, params, context.get());

        SECTION("findPrefetch")
        {
            auto prefetchLoops = findPrefetch(graph);
            CHECK(prefetchLoops.size() == 1);

            auto numUnroll = (*prefetchLoops.cbegin()).second;
            CHECK(numUnroll == std::max(2, prefetchInFlight));
        }
    }
}
