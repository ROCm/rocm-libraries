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
