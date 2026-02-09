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
#include <rocRoller/KernelGraph/Utils.hpp>

#include <common/CommonGraphs.hpp>

#include "TestContext.hpp"

#include <algorithm>
#include <ranges>

namespace TopologicalCompareTest
{
    using namespace rocRoller;

    TEST_CASE("TopologicalCompare works.", "[kernel-graph][utils]")
    {
        auto context = TestContext::ForTestDevice();
        auto gemm    = rocRollerTest::Graphs::GEMM(DataType::Float);

        gemm.setTileSize(64, 64, 8);
        gemm.setMFMA(16, 16, 1, 1);
        gemm.setUseLDS(false, false, false);

        auto graph = gemm.getKernelGraph();

        std::string constructionMethod = GENERATE("reference", "pointer");
        DYNAMIC_SECTION("Construction from " << constructionMethod)
        {
            auto testComparator = [&](KernelGraph::TopologicalCompare&& comp) {
                // Get some nodes that have a defined, non-body order with each other.

                auto notContainerNode = [&](int nodeIdx) {
                    using namespace KernelGraph::ControlGraph;
                    auto node = graph.control.getNode(nodeIdx);

                    auto visitor = [&](auto const& node) {
                        using T = std::decay_t<decltype(node)>;
                        return !COperationWithBody<T>;
                    };

                    return std::visit(visitor, node);
                };

                std::vector<int> nodes;

                auto definedOrderWithOtherNodes = [&](int nodeIdx) -> bool {
                    using namespace KernelGraph::ControlGraph;
                    for(auto otherIdx : nodes)
                    {
                        auto order
                            = graph.control.compareNodes(rocRoller::UpdateCache, otherIdx, nodeIdx);
                        if(order == NodeOrdering::Undefined
                           || order == NodeOrdering::LeftInBodyOfRight
                           || order == NodeOrdering::RightInBodyOfLeft)
                            return false;
                    }

                    return true;
                };

                auto nonContainerNodes = graph.control.getNodes().filter(notContainerNode);
                for(auto node : nonContainerNodes)
                {
                    if(definedOrderWithOtherNodes(node))
                        nodes.push_back(node);
                }

                std::ranges::sort(nodes, comp);

                for(int idx = 0; idx + 1 < nodes.size(); ++idx)
                {
                    CHECK(graph.control.compareNodes(
                              rocRoller::UpdateCache, nodes[idx], nodes[idx + 1])
                          == KernelGraph::ControlGraph::NodeOrdering::LeftFirst);
                }
            };

            if(constructionMethod == "reference")
            {
                testComparator(KernelGraph::TopologicalCompare(graph));
            }
            else if(constructionMethod == "pointer")
            {
                testComparator(KernelGraph::TopologicalCompare(&graph));
            }
            else
            {
                FAIL();
            }
        }
    }
}
