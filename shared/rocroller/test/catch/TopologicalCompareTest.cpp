// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
