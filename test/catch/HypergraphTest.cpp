/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

namespace HypergraphTest
{
    using namespace rocRoller;

    struct TestForLoop
    {
        std::string toString() const
        {
            return "TestForLoop";
        }
        friend std::ostream& operator<<(std::ostream& os, const TestForLoop& tfl)
        {
            return os << tfl.toString();
        }
    };

    struct TestSubDimension
    {
        std::string toString() const
        {
            return "TestSubDimension";
        }
        friend std::ostream& operator<<(std::ostream& os, const TestSubDimension& tsd)
        {
            return os << tsd.toString();
        }
    };
    struct TestUser
    {
        std::string toString() const
        {
            return "TestUser";
        }
        friend std::ostream& operator<<(std::ostream& os, const TestUser& tu)
        {
            return os << tu.toString();
        }
    };
    struct TestVGPR
    {
        std::string toString() const
        {
            return "TestVGPR";
        }
        friend std::ostream& operator<<(std::ostream& os, const TestVGPR& tv)
        {
            return os << tv.toString();
        }
    };

    using TestDimension = std::variant<TestForLoop, TestSubDimension, TestUser, TestVGPR>;

    std::string toString(TestDimension const& t)
    {
        return std::visit([](const auto& a) { return a.toString(); }, t);
    }

    template <typename T>
    requires(!std::same_as<TestDimension,
                           std::decay_t<T>> && std::constructible_from<TestDimension, T>) bool
        operator==(T const& lhs, T const& rhs)
    {
        // Since none of these have any members, if the types are the same, they are equal.
        return true;
    }

    struct TestForget
    {
        std::string toString() const
        {
            return "TestForget";
        }
        friend std::ostream& operator<<(std::ostream& os, const TestForget& tf)
        {
            return os << tf.toString();
        }
    };
    struct TestSplit
    {
        std::string toString() const
        {
            return "TestSplit";
        }
        friend std::ostream& operator<<(std::ostream& os, const TestSplit& ts)
        {
            return os << ts.toString();
        }
    };

    using TestTransform = std::variant<TestForget, TestSplit>;

    std::string toString(TestTransform const& t)
    {
        return std::visit([](const auto& a) { return a.toString(); }, t);
    }

    template <typename T>
    requires(!std::same_as<TestTransform,
                           std::decay_t<T>> && std::constructible_from<TestTransform, T>) bool
        operator==(T const& lhs, T const& rhs)
    {
        // Since none of these have any members, if the types are the same, they are equal.
        return true;
    }

    using myHypergraph = Graph::Hypergraph<TestDimension, TestTransform>;

    TEST_CASE("Basic Hypergraph", "[kernel-graph]")
    {

        myHypergraph g;

        auto u0  = g.addElement(TestUser{});
        auto sd0 = g.addElement(TestSubDimension{});
        auto sd1 = g.addElement(TestSubDimension{});

        auto TestSplit0 = g.addElement(TestSplit{}, {u0}, {sd0, sd1});

        auto TestVGPR0   = g.addElement(TestVGPR{});
        auto TestForget0 = g.addElement(TestForget{}, {sd0, sd1}, {TestVGPR0});

        auto TestVGPR1   = g.addElement(TestVGPR{});
        auto TestForget1 = g.addElement(TestForget{}, {sd0, sd1}, {TestVGPR1});

        SECTION("toDOT")
        {
            std::string expected = R".(
            digraph {
                    "1"[label="TestUser(1)"];
                    "2"[label="TestSubDimension(2)"];
                    "3"[label="TestSubDimension(3)"];
                    "4"[label="TestSplit(4)",shape=box];
                    "5"[label="TestVGPR(5)"];
                    "6"[label="TestForget(6)",shape=box];
                    "7"[label="TestVGPR(7)"];
                    "8"[label="TestForget(8)",shape=box];
                    "1" -> "4"
                    "2" -> "6"
                    "2" -> "8"
                    "3" -> "6"
                    "3" -> "8"
                    "4" -> "2"
                    "4" -> "3"
                    "6" -> "5"
                    "8" -> "7"
                    {
                        rank=same
                        "2"->"3"[style=invis]
                        rankdir=LR
                    }
                    {
                        rank=same
                        "2"->"3"[style=invis]
                        rankdir=LR
                    }
                    {
                        rank=same
                        "2"->"3"[style=invis]
                        rankdir=LR
                    }
                }
            ).";

            CHECK(NormalizedSource(expected) == NormalizedSource(g.toDOT()));
        }

        SECTION("Get should not throw")
        {
            CHECK_NOTHROW(std::get<TestUser>(std::get<TestDimension>(g.getElement(u0))));
            CHECK_NOTHROW(std::get<TestForget>(std::get<TestTransform>(g.getElement(TestForget1))));
        }

        SECTION("Get TestSubDimension")
        {
            auto subDimensions = g.getNodes<TestSubDimension>().to<std::vector>();
            CHECK(subDimensions.size() == 2);
            CHECK(std::count(subDimensions.begin(), subDimensions.end(), sd0) == 1);
            CHECK(std::count(subDimensions.begin(), subDimensions.end(), sd1) == 1);
        }

        SECTION("Get TestSplit")
        {
            auto inputNodes = g.getInputNodeIndices<TestSplit>(sd0).to<std::vector>();
            CHECK(inputNodes.size() == 1);
            CHECK(inputNodes.at(0) == u0);
        }

        SECTION("Get TestForget")
        {
            auto outputNodes = g.getOutputNodeIndices<TestForget>(sd0).to<std::vector>();
            CHECK(outputNodes.size() == 2);
            CHECK(std::count(outputNodes.begin(), outputNodes.end(), TestVGPR0) == 1);
            CHECK(std::count(outputNodes.begin(), outputNodes.end(), TestVGPR1) == 1);
        }

        SECTION("depthFirstVisit")
        {
            auto             nodes = g.depthFirstVisit(u0).to<std::vector>();
            std::vector<int> expectedNodes{
                u0, TestSplit0, sd0, TestForget0, TestVGPR0, TestForget1, TestVGPR1, sd1};
            CHECK(expectedNodes == nodes);

            auto loc = g.getLocation(nodes[0]);
            CHECK(u0 == loc.index);
            CHECK(std::holds_alternative<TestDimension>(loc.element));
            CHECK(std::holds_alternative<TestUser>(std::get<TestDimension>(loc.element)));
            CHECK(0 == loc.incoming.size());
            CHECK(std::vector<int>{TestSplit0} == loc.outgoing);

            auto loc2 = g.getLocation(u0);
            CHECK(loc == loc2);

            loc = g.getLocation(nodes[1]);
            myHypergraph::Location expected{
                TestSplit0, {u0}, {sd0, sd1}, TestTransform{TestSplit{}}};
            CHECK(expected == loc);

            CHECK(TestSplit0 == loc.index);

            CHECK(myHypergraph::Element{TestTransform{TestSplit{}}} == loc.element);
            CHECK(std::vector<int>{u0} == loc.incoming);
            CHECK((std::vector<int>{sd0, sd1}) == loc.outgoing);

            CHECK(std::vector<int>{u0} == g.parentNodes(sd0).to<std::vector>());
            CHECK((std::vector<int>{sd0, sd1}) == g.childNodes(u0).to<std::vector>());

            CHECK(std::vector<int>{u0} == g.parentNodes(TestSplit0).to<std::vector>());
            CHECK((std::vector<int>{sd0, sd1}) == g.childNodes(TestSplit0).to<std::vector>());

            CHECK((std::vector<int>{sd0, sd1}) == g.parentNodes(TestVGPR0).to<std::vector>());
            CHECK((std::vector<int>{TestVGPR0, TestVGPR1}) == g.childNodes(sd1).to<std::vector>());
        }

        SECTION("depthFirstVisit with multiple leaf nodes doesn't give us whole graph")
        {
            auto nodes = g.depthFirstVisit(TestVGPR0, Graph::Direction::Upstream).to<std::vector>();
            std::vector<int> expectedNodes{TestVGPR0, TestForget0, sd0, TestSplit0, u0, sd1};
            CHECK(expectedNodes == nodes);
        }

        SECTION("Visiting from all the leaf nodes gives us the whole graph")
        {
            // TODO: "Make generators less lazy" once the generator semantics have been made less lazy, this can be collapsed into the next line and we can avoid converting the 'leaves' generator into a vector.
            auto leaves = g.leaves().to<std::vector>();
            auto nodes  = g.depthFirstVisit(leaves, Graph::Direction::Upstream).to<std::vector>();
            std::vector<int> expectedNodes{
                TestVGPR0, TestForget0, sd0, TestSplit0, u0, sd1, TestVGPR1, TestForget1};
            CHECK(expectedNodes == nodes);
        }

        SECTION("breadthFirstVisit")
        {
            auto             nodes = g.breadthFirstVisit(u0).to<std::vector>();
            std::vector<int> expectedNodes{
                u0, TestSplit0, sd0, sd1, TestForget0, TestForget1, TestVGPR0, TestVGPR1};
            CHECK(expectedNodes == nodes);
        }

        SECTION("roots")
        {
            auto             nodes         = g.roots().to<std::vector>();
            std::vector<int> expectedNodes = {u0};
            CHECK(expectedNodes == nodes);
        }

        SECTION("leaves")
        {
            auto             nodes         = g.leaves().to<std::vector>();
            std::vector<int> expectedNodes = {TestVGPR0, TestVGPR1};
            CHECK(expectedNodes == nodes);
        }

        SECTION("getNeighbours")
        {
            CHECK((std::vector<int>{u0})
                  == g.getNeighbours<Graph::Direction::Upstream>(TestSplit0).to<std::vector>());
            CHECK((std::vector<int>{TestSplit0})
                  == g.getNeighbours<Graph::Direction::Upstream>(sd0).to<std::vector>());
            CHECK((std::vector<int>{TestSplit0})
                  == g.getNeighbours<Graph::Direction::Upstream>(sd1).to<std::vector>());
            CHECK((std::vector<int>{sd0, sd1})
                  == g.getNeighbours<Graph::Direction::Downstream>(TestSplit0).to<std::vector>());
        }

        SECTION("Adding a node to the graph")
        {
            // Add a for loop.
            auto loop = g.addElement(TestForLoop{}, {TestSplit0}, {TestForget0});

            {
                auto             loc           = g.getLocation(TestSplit0);
                std::vector<int> expectedNodes = {sd0, sd1, loop};
                CHECK(expectedNodes == loc.outgoing);
            }

            {
                auto             loc           = g.getLocation(TestForget0);
                std::vector<int> expectedNodes = {sd0, sd1, loop};
                CHECK(expectedNodes == loc.incoming);
            }

            {
                auto             nodes = g.depthFirstVisit(u0).to<std::vector>();
                std::vector<int> expectedNodes{
                    u0, TestSplit0, sd0, TestForget0, TestVGPR0, TestForget1, TestVGPR1, sd1, loop};
                CHECK(expectedNodes == nodes);
            }

            {
                std::string expected = R".(
                digraph {
                    "1"[label="TestUser(1)"];
                    "2"[label="TestSubDimension(2)"];
                    "3"[label="TestSubDimension(3)"];
                    "4"[label="TestSplit(4)",shape=box];
                    "5"[label="TestVGPR(5)"];
                    "6"[label="TestForget(6)",shape=box];
                    "7"[label="TestVGPR(7)"];
                    "8"[label="TestForget(8)",shape=box];
                    "9"[label="TestForLoop(9)"];
                    "1" -> "4"
                    "2" -> "6"
                    "2" -> "8"
                    "3" -> "6"
                    "3" -> "8"
                    "4" -> "2"
                    "4" -> "3"
                    "4" -> "9"
                    "6" -> "5"
                    "8" -> "7"
                    "9" -> "6"
                    {
                        rank=same
                        "2"->"3"->"9"[style=invis]
                        rankdir=LR
                    }
                    {
                        rank=same
                        "2"->"3"->"9"[style=invis]
                        rankdir=LR
                    }
                    {
                        rank=same
                        "2"->"3"[style=invis]
                        rankdir=LR
                    }
                }
            ).";

                CHECK(NormalizedSource(expected) == NormalizedSource(g.toDOT()));
            }

            CHECK(std::set({u0, sd0, sd1, TestVGPR0, TestVGPR1, loop})
                  == g.getNodes().to<std::set>());
            CHECK(std::set({TestSplit0, TestForget0, TestForget1}) == g.getEdges().to<std::set>());

            CHECK(std::set({TestVGPR0, TestVGPR1}) == g.getNodes<TestVGPR>().to<std::set>());
            CHECK(std::set({TestForget0, TestForget1})
                  == g.getElements<TestForget>().to<std::set>());

            {
                CHECK(std::get<TestUser>(std::get<TestDimension>(g.getElement(u0))) == TestUser{});

                CHECK_THROWS_AS(g.getElement(-1), FatalError);
            }
        }
    }

    TEST_CASE("Hypergraph pathing", "[kernel-graph]")
    {

        myHypergraph g;

        auto u1  = g.addElement(TestUser{});
        auto sd2 = g.addElement(TestSubDimension{});
        auto sd3 = g.addElement(TestSubDimension{});

        auto TestSplit4 = g.addElement(TestSplit{}, {u1}, {sd2, sd3});

        auto sd5 = g.addElement(TestSubDimension{});
        auto sd6 = g.addElement(TestSubDimension{});
        auto sd7 = g.addElement(TestSubDimension{});
        auto sd8 = g.addElement(TestSubDimension{});

        auto TestSplit9  = g.addElement(TestSplit{}, {sd2}, {sd8});
        auto TestSplit10 = g.addElement(TestSplit{}, {sd3, sd8}, {sd6});
        auto TestSplit11 = g.addElement(TestSplit{}, {sd3}, {sd7});

        CHECK((std::vector<int>{u1, TestSplit4, sd2})
              == g.path<Graph::Direction::Downstream>(std::vector<int>{u1}, std::vector<int>{sd2})
                     .to<std::vector>());

        CHECK(
            (std::vector<int>{sd2, sd3, TestSplit4, u1})
            == g.path<Graph::Direction::Upstream>(std::vector<int>{sd2, sd3}, std::vector<int>{u1})
                   .to<std::vector>());

        CHECK((std::vector<int>{})
              == g.path<Graph::Direction::Upstream>(std::vector<int>{sd2}, std::vector<int>{u1})
                     .to<std::vector>());

        CHECK((std::vector<int>{sd6, TestSplit10, sd8, TestSplit9, sd2, sd3, TestSplit4, u1})
              == g.path<Graph::Direction::Upstream>(std::vector<int>{sd6}, std::vector<int>{u1})
                     .to<std::vector>());

        CHECK((std::vector<int>{u1, TestSplit4, sd3, TestSplit11, sd7})
              == g.path<Graph::Direction::Downstream>(std::vector<int>{u1}, std::vector<int>{sd7})
                     .to<std::vector>());

        CHECK((std::vector<int>{u1, TestSplit4, sd3, sd2, TestSplit9, sd8, TestSplit10, sd6})
              == g.path<Graph::Direction::Downstream>(std::vector<int>{u1}, std::vector<int>{sd6})
                     .to<std::vector>());
    }

    TEST_CASE("Bad Hypergraph setup", "[kernel-graph]")
    {
        myHypergraph g;

        auto u0  = g.addElement(TestUser{});
        auto sd0 = g.addElement(TestSubDimension{});
        auto sd1 = g.addElement(TestSubDimension{});

        auto TestSplit0 = g.addElement(TestSplit{}, {u0}, {sd0, sd1});

        auto TestVGPR0   = g.addElement(TestVGPR{});
        auto TestForget0 = g.addElement(TestForget{}, {sd0, sd1}, {TestVGPR0});

        auto TestVGPR1   = g.addElement(TestVGPR{});
        auto TestForget1 = g.addElement(TestForget{}, {sd0, sd1}, {TestVGPR1});

        // Edges to Edges
        CHECK_THROWS_AS(g.addElement(TestForget{}, {u0}, {TestSplit0}), FatalError);
        CHECK_THROWS_AS(g.addElement(TestForget{}, {}, {TestSplit0}), FatalError);

        // Nodes to nodes
        CHECK_THROWS_AS(g.addElement(TestSubDimension{}, {u0}, {}), FatalError);
    }

    TEST_CASE("Hypergraph sorting and visiting", "[kernel-graph]")
    {
        myHypergraph g;

        auto u1 = g.addElement(TestUser{});

        auto sd2 = g.addElement(TestSubDimension{});
        auto sp3 = g.addElement(TestSplit{}, {u1}, {sd2});

        auto sd4 = g.addElement(TestSubDimension{});
        auto sp5 = g.addElement(TestSplit{}, {u1}, {sd4});

        auto sd6 = g.addElement(TestSubDimension{});
        auto sp7 = g.addElement(TestSplit{}, {sd4}, {sd6});

        auto sd8 = g.addElement(TestSubDimension{});
        auto sp9 = g.addElement(TestSplit{}, {sd2, sd6}, {sd8});

        // 1
        // | \
        // 3  5
        // |  |
        // 2  4
        // |  |
        // |  7
        // |  |
        // |  6
        // | /
        // 9
        // |
        // 8

        SECTION("Topological sort")
        {
            auto topo = g.topologicalSort().to<std::vector>();
            CHECK(topo == std::vector<int>({1, 3, 2, 5, 4, 7, 6, 9, 8}));
        }

        SECTION("Breadth first visit")
        {
            auto bfs = g.breadthFirstVisit(*g.roots().begin(), Graph::Direction::Downstream)
                           .to<std::vector>();
            CHECK(bfs == std::vector<int>({1, 3, 5, 2, 4, 9, 7, 8, 6}));

            bfs = g.breadthFirstVisit(sp3, Graph::Direction::Downstream).to<std::vector>();
            CHECK(bfs == std::vector<int>({3, 2, 9, 8}));

            bfs = g.breadthFirstVisit(sp9, Graph::Direction::Downstream).to<std::vector>();
            CHECK(bfs == std::vector<int>({9, 8}));

            bfs = g.breadthFirstVisit(sp9, Graph::Direction::Upstream).to<std::vector>();
            CHECK(bfs == std::vector<int>({9, 2, 6, 3, 7, 1, 4, 5}));

            bfs = g.breadthFirstVisit(sp7, Graph::Direction::Upstream).to<std::vector>();
            CHECK(bfs == std::vector<int>({7, 4, 5, 1}));
        }
    }

    TEST_CASE("Hypergraph delete element", "[kernel-graph]")
    {
        myHypergraph g;

        auto u0  = g.addElement(TestUser{});
        auto sd0 = g.addElement(TestSubDimension{});
        auto sd1 = g.addElement(TestSubDimension{});

        auto TestSplit0 = g.addElement(TestSplit{}, {u0}, {sd0, sd1});

        auto TestVGPR0   = g.addElement(TestVGPR{});
        auto TestForget0 = g.addElement(TestForget{}, {sd0, sd1}, {TestVGPR0});

        auto TestVGPR1   = g.addElement(TestVGPR{});
        auto TestForget1 = g.addElement(TestForget{}, {sd0, sd1}, {TestVGPR1});

        CHECK_THROWS_AS(
            g.deleteElement<TestForget>(std::vector<int>{sd0}, std::vector<int>{TestVGPR0}),
            FatalError);

        g.deleteElement<TestForget>(std::vector<int>{sd0, sd1}, std::vector<int>{TestVGPR0});
    }

    TEST_CASE("Hypergraph with parallel edges", "[kernel-graph]")
    {
        myHypergraph g;

        auto u0  = g.addElement(TestUser{});
        auto sd0 = g.addElement(TestSubDimension{});
        auto sd1 = g.addElement(TestSubDimension{});

        auto TestSplit0 = g.addElement(TestSplit{}, {u0}, {sd0, sd1});
        auto TestSplit1 = g.addElement(TestSplit{}, {u0}, {sd0, sd1});

        auto childVec  = g.childNodes(1).to<std::vector>();
        auto parentVec = g.parentNodes(2).to<std::vector>();

        CHECK(childVec == std::vector<int>({2, 3}));
        CHECK(parentVec == std::vector<int>({1}));
    }

    TEST_CASE("Follow Hypergraph edges", "[kernel-graph]")
    {
        myHypergraph g;

        auto u0  = g.addElement(TestUser{});
        auto sd0 = g.addElement(TestSubDimension{});
        auto sd1 = g.addElement(TestSubDimension{});
        auto sd2 = g.addElement(TestSubDimension{});

        auto TestSplit0  = g.addElement(TestSplit{}, {u0}, {sd0});
        auto TestSplit1  = g.addElement(TestSplit{}, {u0}, {sd1});
        auto TestSplit2  = g.addElement(TestSplit{}, {sd1}, {sd2});
        auto TestForget0 = g.addElement(TestForget{}, {sd0}, {sd2});

        CHECK(g.followEdges<TestSplit>({}) == std::set<int>());
        CHECK(g.followEdges<TestSplit>({u0}) == std::set<int>({u0, sd0, sd1, sd2}));
        CHECK(g.followEdges<TestSplit>({sd0}) == std::set<int>({sd0}));
        CHECK(g.followEdges<TestSplit>({sd1}) == std::set<int>({sd1, sd2}));
        CHECK(g.followEdges<TestSplit>({sd2}) == std::set<int>({sd2}));
        CHECK(g.followEdges<TestForget>({sd0}) == std::set<int>({sd0, sd2}));
        CHECK(g.followEdges<TestForget>({sd2}) == std::set<int>({sd2}));
    }

    TEST_CASE("Hypergraph with reachable nodes", "[kernel-graph]")
    {
        myHypergraph g;

        auto u0  = g.addElement(TestUser{});
        auto sd0 = g.addElement(TestSubDimension{});
        auto sd1 = g.addElement(TestSubDimension{});
        auto sd2 = g.addElement(TestSubDimension{});
        auto sd3 = g.addElement(TestSubDimension{});
        auto u1  = g.addElement(TestUser{});
        auto u2  = g.addElement(TestUser{});
        auto v0  = g.addElement(TestVGPR{});

        g.addElement(TestSplit{}, {u0}, {sd0});
        g.addElement(TestSplit{}, {u0}, {sd1});
        g.addElement(TestSplit{}, {sd1}, {sd2, u1});
        g.addElement(TestForget{}, {sd0}, {sd2});
        g.addElement(TestSplit{}, {u1}, {sd3});
        g.addElement(TestSplit{}, {u2}, {v0});
        g.addElement(TestSplit{}, {v0}, {sd1});

        auto isSubDimension
            = [](auto const& node) { return std::holds_alternative<TestSubDimension>(node); };

        auto isSplit = [](auto const& edge) { return std::holds_alternative<TestSplit>(edge); };

        auto isUser = [](auto const& node) { return std::holds_alternative<TestUser>(node); };

        auto truePred = [](auto const& node) { return true; };

        CHECK(reachableNodes<Graph::Direction::Downstream>(
                  g, u0, isSubDimension, isSplit, isSubDimension)
                  .to<std::set>()
              == std::set<int>({sd0, sd1, sd2}));

        CHECK(reachableNodes<Graph::Direction::Downstream>(g, u0, isSubDimension, isSplit, truePred)
                  .to<std::set>()
              == std::set<int>({sd0, sd1, sd2, u1}));

        CHECK(reachableNodes<Graph::Direction::Upstream>(g, sd3, isSubDimension, truePred, isUser)
                  .to<std::set>()
              == std::set<int>({u1}));

        CHECK(reachableNodes<Graph::Direction::Upstream>(g, u1, isSubDimension, truePred, isUser)
                  .to<std::set>()
              == std::set<int>({u0}));

        CHECK(reachableNodes<Graph::Direction::Upstream>(g, u1, truePred, truePred, isUser)
                  .to<std::set>()
              == std::set<int>({u0, u2}));
    }

}
