// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/NodeSchedulingUtils.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

using namespace rocRoller;
using namespace rocRoller::KernelGraph::ControlGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;

namespace NodeSchedulingUtilsTest
{

    std::vector<int> addMultiplyNodes(KernelGraph::KernelGraph& graph, int parent, int count)
    {
        std::vector<int> rv;
        for(int i = 0; i < count; ++i)
        {
            auto multiply = graph.control.addElement(Multiply{});
            graph.control.addElement(Body(), {parent}, {multiply});
            rv.push_back(multiply);
        }
        return rv;
    }

    TEST_CASE("getGroupedNodes", "[kernel-graph][utils][node-scheduling]")
    {
        KernelGraph::KernelGraph graph;

        auto kernel = graph.control.addElement(Kernel());

        auto forLoop = graph.control.addElement(ForLoopOp{Expression::literal(10), "Loop"});

        auto multipliesInLoop = addMultiplyNodes(graph, forLoop, 10);

        auto multipliesOutsideLoop = addMultiplyNodes(graph, kernel, 12);

        SECTION("Groups nodes correctly")
        {

            auto expected = std::unordered_map<int, std::vector<int>>{
                {forLoop, multipliesInLoop}, {kernel, multipliesOutsideLoop}};
            CHECK(rocRoller::KernelGraph::NodeScheduling::getGroupedNodes<Multiply>(graph)
                  == expected);

            SECTION("Sequence edges don't affect grouping")
            {
                for(int i = 0; i + 1 < multipliesInLoop.size(); ++i)
                {
                    graph.control.chain<Sequence>(multipliesInLoop[i], multipliesInLoop[i + 1]);
                }

                for(int i = 0; i + 1 < multipliesOutsideLoop.size(); ++i)
                {
                    graph.control.chain<Sequence>(multipliesOutsideLoop[i],
                                                  multipliesOutsideLoop[i + 1]);
                }

                CHECK(rocRoller::KernelGraph::NodeScheduling::getGroupedNodes<Multiply>(graph)
                      == expected);

                SECTION("Simplify doesn't affect grouping")
                {
                    auto simplifiedGraph = rocRoller::KernelGraph::Simplify().apply(graph);
                    CHECK(rocRoller::KernelGraph::NodeScheduling::getGroupedNodes<Multiply>(
                              simplifiedGraph)
                          == expected);
                }
            }
        }
    }

    TEST_CASE("createSubGraph", "[kernel-graph][node-scheduling]")
    {
        KernelGraph::KernelGraph graph;

        auto kernel  = graph.control.addElement(Kernel());
        auto forLoop = graph.control.addElement(ForLoopOp{Expression::literal(10), "Loop"});
        graph.control.addElement(Body(), {kernel}, {forLoop});
        auto multiplies = addMultiplyNodes(graph, forLoop, 10);

        // 2 chains of 5 multiply nodes each
        for(int i = 0; i < 4; i++)
        {
            graph.control.chain<Sequence>(multiplies.at(i), multiplies.at(i + 1));
            graph.control.chain<Sequence>(multiplies.at(i + 5), multiplies.at(i + 6));
        }

        auto multipliesOutsideLoop = addMultiplyNodes(graph, kernel, 12);
        for(int i = 0; i < 3; i++)
        {
            graph.control.chain<Sequence>(multipliesOutsideLoop.at(i),
                                          multipliesOutsideLoop.at(i + 1));
            graph.control.chain<Sequence>(multipliesOutsideLoop.at(i + 4),
                                          multipliesOutsideLoop.at(i + 5));
            graph.control.chain<Sequence>(multipliesOutsideLoop.at(i + 8),
                                          multipliesOutsideLoop.at(i + 9));
        }

        bool simplify = GENERATE(true, false);

        DYNAMIC_SECTION("simplify: " << simplify)
        {
            auto inputGraph = simplify ? rocRoller::KernelGraph::Simplify().apply(graph) : graph;

            std::vector nodesToKeep = {forLoop,
                                       multiplies[0],
                                       multiplies[4],
                                       multiplies[5],
                                       multiplies[7],
                                       multiplies[9],
                                       multipliesOutsideLoop[0],
                                       multipliesOutsideLoop[1],
                                       multipliesOutsideLoop[4],
                                       multipliesOutsideLoop[5],
                                       multipliesOutsideLoop[8],
                                       multipliesOutsideLoop[9]};

            auto subGraph
                = rocRoller::KernelGraph::NodeScheduling::createSubGraph(inputGraph, nodesToKeep);

            CHECK(subGraph.getNodes().to<std::vector>() == nodesToKeep);

            SECTION("Preserves ordering")
            {
                for(auto iterA = nodesToKeep.begin(); iterA != nodesToKeep.end(); ++iterA)
                {
                    INFO(*iterA);
                    for(auto iterB = iterA + 1; iterB != nodesToKeep.end(); ++iterB)
                    {
                        INFO(*iterB);
                        auto origOrder = graph.control.compareNodes(UpdateCache, *iterA, *iterB);
                        auto subOrder  = subGraph.compareNodes(UpdateCache, *iterA, *iterB);
                        CHECK(subOrder == origOrder);
                    }
                }
            }
        }
    }

    TEST_CASE("orderNodes", "[kernel-graph][node-scheduling]")
    {
        KernelGraph::KernelGraph graph;

        auto kernel  = graph.control.addElement(Kernel());
        auto forLoop = graph.control.addElement(ForLoopOp{Expression::literal(10), "Loop"});
        graph.control.addElement(Body(), {kernel}, {forLoop});
        auto multiplies = addMultiplyNodes(graph, forLoop, 10);

        // 2 chains of 5 multiply nodes each
        for(int i = 0; i < 4; i++)
        {
            graph.control.chain<Sequence>(multiplies.at(i), multiplies.at(i + 1));
            graph.control.chain<Sequence>(multiplies.at(i + 5), multiplies.at(i + 6));
        }

        auto multipliesOutsideLoop = addMultiplyNodes(graph, kernel, 12);
        for(int i = 0; i < 3; i++)
        {
            graph.control.chain<Sequence>(multipliesOutsideLoop.at(i),
                                          multipliesOutsideLoop.at(i + 1));
            graph.control.chain<Sequence>(multipliesOutsideLoop.at(i + 4),
                                          multipliesOutsideLoop.at(i + 5));
            graph.control.chain<Sequence>(multipliesOutsideLoop.at(i + 8),
                                          multipliesOutsideLoop.at(i + 9));
        }

        bool simplify = GENERATE(true, false);

        DYNAMIC_SECTION("simplify: " << simplify)
        {
            auto inputGraph = simplify ? rocRoller::KernelGraph::Simplify().apply(graph) : graph;

            std::vector<int> nodesToOrder = multiplies;
            nodesToOrder.insert(
                nodesToOrder.end(), multipliesOutsideLoop.begin(), multipliesOutsideLoop.end());

            auto comp = [](int a, int b) { return a > b; };

            rocRoller::KernelGraph::NodeScheduling::orderNodes(inputGraph, nodesToOrder, comp);

            for(auto iterA = nodesToOrder.begin(); iterA != nodesToOrder.end(); ++iterA)
            {
                for(auto iterB = iterA + 1; iterB != nodesToOrder.end(); ++iterB)
                {
                    auto order = graph.control.compareNodes(UpdateCache, *iterA, *iterB);
                    CAPTURE(order);
                    CHECK((order == NodeOrdering::LeftFirst
                           || order == NodeOrdering::RightInBodyOfLeft
                           || order == NodeOrdering::Undefined));

                    if(order == NodeOrdering::Undefined)
                    {
                        CHECK(*iterA > *iterB);
                    }
                }
            }
        }
    }
}
