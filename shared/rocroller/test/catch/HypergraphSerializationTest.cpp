// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <compare>
#include <fstream>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "ExpressionMatchers.hpp"
#include "TestContext.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <common/SourceMatcher.hpp>

#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/Serialization/ControlGraph.hpp>
#include <rocRoller/Serialization/Hypergraph.hpp>
#include <rocRoller/Serialization/YAML.hpp>

namespace HypergraphSerializationTest
{
    using namespace rocRoller;

    TEST_CASE("Hypergraph is ordered when deserialized",
              "[hypergraph][kernel-graph][serialization]")
    {
        // NOTE: ControlGraph Operations and Edges are used here for their serialization properties
        using pseudoControlGraph = Graph::Hypergraph<KernelGraph::ControlGraph::Operation,
                                                     KernelGraph::ControlGraph::ControlEdge,
                                                     true>;
        pseudoControlGraph g;

        auto u0  = g.addElement(KernelGraph::ControlGraph::SetCoordinate{});
        auto op0 = g.addElement(KernelGraph::ControlGraph::DoWhileOp{});
        auto op1 = g.addElement(KernelGraph::ControlGraph::ConditionalOp{});
        auto op2 = g.addElement(KernelGraph::ControlGraph::AssertOp{});
        auto op3 = g.addElement(KernelGraph::ControlGraph::Barrier{});

        auto body0 = g.addElement(KernelGraph::ControlGraph::Body{}, {u0}, {op0, op2, op1});
        auto body1 = g.addElement(KernelGraph::ControlGraph::Body{}, {op0, op2, op1}, {op3});

        SECTION("Elements are sorted in the order they were added")
        {
            CHECK(g.getNeighbours<Graph::Direction::Downstream>(body0)
                  == std::vector<int>{op0, op2, op1});
            CHECK(g.getNeighbours<Graph::Direction::Upstream>(body1)
                  == std::vector<int>{op0, op2, op1});
        }

        SECTION("Serialization preserves ordering")
        {
            auto yaml          = Serialization::toYAML(g);
            auto gDeserialized = Serialization::fromYAML<pseudoControlGraph>(yaml);
            CHECK(gDeserialized.getNeighbours<Graph::Direction::Downstream>(body0)
                  == std::vector<int>{op0, op2, op1});
            CHECK(gDeserialized.getNeighbours<Graph::Direction::Upstream>(body1)
                  == std::vector<int>{op0, op2, op1});
        }
    }
}
