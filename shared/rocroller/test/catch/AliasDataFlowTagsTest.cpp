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

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <omp.h>

#include <algorithm>
#include <random>

#include "SimpleTest.hpp"
#include "TestContext.hpp"
#include "common/SourceMatcher.hpp"
#include <common/CommonGraphs.hpp>
#include <common/Utilities.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/Transforms/AliasDataFlowTags_detail.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>

using namespace rocRoller;
using namespace Catch::Matchers;

namespace std
{
    std::ostream& operator<<(std::ostream& stream, std::pair<int, int> const& pair)
    {
        auto const& [a, b] = pair;
        return stream << "{" << a << ", " << b << "}";
    }
}
namespace AliasDataFlowTagsTest
{
    TEST_CASE("AliasDataFlowTags transformation works.", "[kernel-graph][graph-transforms]")
    {
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::CoordinateGraph;
        using namespace rocRoller::KernelGraph::ControlGraph;

        auto context = TestContext::ForDefaultTarget();

        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        int macK  = 16;
        int waveK = 2;

        example.setTileSize(256, 64, macK);
        example.setMFMA(32, 32, waveK, 1);
        example.setUseLDS(true, true, false);
        example.setUnroll(2, 2);

        example.setPrefetch(true, 2, 2, false);

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
        graph = transform<ConnectWorkgroups>(graph, params, context.get());
        graph = transform<UnrollLoops>(graph, params, context.get());
        graph = transform<FuseLoops>(graph);
        graph = transform<RemoveDuplicates>(graph);
        graph = transform<OrderEpilogueBlocks>(graph);
        graph = transform<CleanLoops>(graph);
        graph = transform<AddPrefetch>(graph, params, context.get());
        graph = transform<AddComputeIndex>(graph);
        graph = transform<AddPRNG>(graph, context.get());
        graph = transform<UpdateWavefrontParameters>(graph, params);
        graph = transform<LoadPacked>(graph, context.get());
        graph = transform<AddConvert>(graph);
        graph = transform<AddDeallocateDataFlow>(graph);
        graph = transform<InlineIncrements>(graph);
        graph = transform<Simplify>(graph);

        SECTION("Call findAliasCandidates directly.")
        {
            auto transformAliases = AliasDataFlowTagsDetail::findAliasCandidates(graph);
            CAPTURE(transformAliases);
            CHECK(transformAliases.size() == 59);
        }

        SECTION("Try different orders of inputs")
        {
            using namespace AliasDataFlowTagsDetail;
            auto groupedExtents = getGroupedTagExtents(graph);

            auto seed = Catch::getSeed();

            for(int j = 0; j < 10; j++)
            {
                std::map<int, int> aliases;

                for(auto& [typeKey, extents] : groupedExtents)
                {
                    std::mt19937 gen(seed);
                    seed++;

                    std::vector<TagExtent> reorder{extents.begin(), extents.end()};
                    std::ranges::shuffle(reorder, gen);

                    std::list<TagExtent> e2{reorder.begin(), reorder.end()};

                    auto theseAliases = findAliasCandidatesForExtents(graph, e2);
                    aliases.insert(theseAliases.begin(), theseAliases.end());
                }

                CHECK_FALSE(aliases.empty());
            }
        }

        SECTION("Apply graph transform.")
        {
            graph = transform<AliasDataFlowTags>(graph);

            namespace CT = rocRoller::KernelGraph::CoordinateGraph;

            auto isAliasEdge = [&](int x) {
                auto el = graph.coordinates.getEdge<CT::DataFlowEdge>(x);
                return std::holds_alternative<CT::Alias>(el);
            };

            {
                auto aliasEdges = graph.coordinates.getEdges<CT::DataFlowEdge>()
                                      .filter(isAliasEdge)
                                      .to<std::vector>();
                CHECK(aliasEdges.size() == 59);
            }

            {
                auto isATile = [&](int x) {
                    auto mt = graph.coordinates.getNode<CT::MacroTile>(x);
                    return mt.layoutType == LayoutType::MATRIX_A;
                };

                auto isAliasForATiles = [&](int x) {
                    auto loc = graph.coordinates.getLocation(x);
                    REQUIRE(isATile(loc.incoming.at(0)) == isATile(loc.outgoing.at(0)));

                    return isATile(loc.incoming.at(0));
                };

                auto aliasEdgesForA = graph.coordinates.getEdges<CT::DataFlowEdge>()
                                          .filter(isAliasEdge)
                                          .filter(isAliasForATiles)
                                          .to<std::vector>();
                CHECK(aliasEdgesForA.size() == 48);
            }

            {
                auto isBTile = [&](int x) {
                    auto mt = graph.coordinates.getNode<CT::MacroTile>(x);
                    return mt.layoutType == LayoutType::MATRIX_B;
                };

                auto isAliasForBTiles = [&](int x) {
                    auto loc = graph.coordinates.getLocation(x);
                    REQUIRE(isBTile(loc.incoming.at(0)) == isBTile(loc.outgoing.at(0)));

                    return isBTile(loc.incoming.at(0));
                };

                auto aliasEdgesForB = graph.coordinates.getEdges<CT::DataFlowEdge>()
                                          .filter(isAliasEdge)
                                          .filter(isAliasForBTiles)
                                          .to<std::vector>();
                CHECK(aliasEdgesForB.size() == 11);
            }
        }
    }
}
