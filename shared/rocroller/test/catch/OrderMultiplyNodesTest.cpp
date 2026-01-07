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
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>

#include "SimpleTest.hpp"
#include "TestContext.hpp"
#include "common/SourceMatcher.hpp"
#include <common/CommonGraphs.hpp>
#include <common/Utilities.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Transforms/OrderMultiplyNodes.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>

using namespace rocRoller;
using namespace Catch::Matchers;

namespace OrderMultiplyNodesTest
{
    // Helper function to get all multiply nodes from the graph
    std::vector<int> getMultiplyNodes(KernelGraph::KernelGraph const& graph)
    {
        using namespace rocRoller::KernelGraph;
        return graph.control.getNodes()
            .filter([&graph](int idx) {
                return graph.control.get<ControlGraph::Multiply>(idx).has_value();
            })
            .to<std::vector>();
    }

    TEST_CASE("OrderMultiplyNodes transformation works.", "[kernel-graph][graph-transforms]")
    {
        using namespace rocRoller::KernelGraph;

        auto context = TestContext::ForDefaultTarget();

        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        example.setTileSize(256, 64, 16);
        example.setMFMA(32, 32, 2, 1);

        auto lds               = GENERATE(true, false);
        auto unrollX           = GENERATE(0, 2);
        auto unrollY           = GENERATE(0, 2);
        auto prefetch          = false;
        auto prefetchInFlight  = 0;
        auto prefetchLDSFactor = 0;
        bool prefetchMixMemOps = false;
        if(lds)
        {
            prefetch = GENERATE(true, false);
            if(prefetch)
            {
                prefetchInFlight  = 4;
                prefetchLDSFactor = 1;
                prefetchMixMemOps = GENERATE(true, false);
            }
        }

        DYNAMIC_SECTION("lds=" << lds << ", unrollX=" << unrollX << ", unrollY=" << unrollY
                               << ", prefetch=" << prefetch << ", prefetchInFlight="
                               << prefetchInFlight << ", prefetchLDSFactor=" << prefetchLDSFactor
                               << ", prefetchMixMemOps=" << prefetchMixMemOps)
        {

            example.setUseLDS(lds, lds, true);
            example.setUnroll(unrollX, unrollY);

            example.setPrefetch(prefetch, prefetchInFlight, prefetchLDSFactor, prefetchMixMemOps);

            auto graph  = example.getKernelGraph();
            auto params = example.getCommandParameters();

            // Apply transformations up to OrderMultiplyNodes
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
            graph = transform<ConnectWorkgroups>(graph, context.get());
            graph = transform<WorkgroupRemapXCC>(graph, context.get(), params->workgroupRemapXCC);
            graph = transform<UnrollLoops>(graph, params, context.get());
            graph = transform<FuseLoops>(graph);
            graph = transform<RemoveDuplicates>(graph);
            graph = transform<OrderEpilogueBlocks>(graph);
            graph = transform<CleanLoops>(graph);
            graph = transform<AddPrefetch>(graph, params, context.get());
            graph = transform<AddPRNG>(graph, context.get());
            graph = transform<UpdateWavefrontParameters>(graph, params);
            graph = transform<AssignIndexExpressions>(graph, context.get(), example.getCommand());
            graph = transform<LoadPacked>(graph, context.get());
            graph = transform<AddConvert>(graph);
            graph = transform<AddDeallocateDataFlow>(graph);
            graph = transform<InlineIncrements>(graph);
            graph = transform<Simplify>(graph);

            auto multiplyNodesBefore = getMultiplyNodes(graph);
            REQUIRE(multiplyNodesBefore.size() > 0);

            int unorderedPairsBefore = 0;
            for(size_t i = 0; i < multiplyNodesBefore.size(); i++)
            {
                for(size_t j = i + 1; j < multiplyNodesBefore.size(); j++)
                {
                    auto order = graph.control.compareNodes(
                        UpdateCache, multiplyNodesBefore[i], multiplyNodesBefore[j]);
                    if(order == ControlGraph::NodeOrdering::Undefined)
                    {
                        unorderedPairsBefore++;
                    }
                }
            }

            REQUIRE(unorderedPairsBefore > 0);

            {
                auto check = NoUnorderedMultiplyNodes(graph);
                CHECK_FALSE(check.satisfied);
                CHECK_THAT(check.explanation, ContainsSubstring("Unordered multiply nodes found"));
            }

            INFO("Unordered pairs before transformation: " << unorderedPairsBefore);

            graph = transform<OrderMultiplyNodes>(graph);

            auto multiplyNodesAfter = getMultiplyNodes(graph);
            CHECK(multiplyNodesAfter == multiplyNodesBefore);

            int unorderedPairsAfter = 0;
            for(size_t i = 0; i < multiplyNodesAfter.size(); i++)
            {
                for(size_t j = i + 1; j < multiplyNodesAfter.size(); j++)
                {
                    auto order = graph.control.compareNodes(
                        UpdateCache, multiplyNodesAfter[i], multiplyNodesAfter[j]);

                    if(order == ControlGraph::NodeOrdering::Undefined)
                    {
                        unorderedPairsAfter++;
                        INFO("Nodes " << multiplyNodesAfter[i] << " and " << multiplyNodesAfter[j]
                                      << " are still unordered");
                    }
                }
            }

            {
                auto check = NoUnorderedMultiplyNodes(graph);
                CHECK(check.satisfied);
                CHECK(check.explanation == "");
            }

            INFO("Unordered pairs after transformation: " << unorderedPairsAfter);

            CHECK(unorderedPairsAfter == 0);
        }
    }
}
