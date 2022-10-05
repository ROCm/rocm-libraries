
#include <compare>
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "SourceMatcher.hpp"

#include <rocRoller/Graph/Hypergraph.hpp>

namespace rocRollerTest
{
    using namespace rocRoller;

    struct TestForLoop
    {
        std::string toString() const
        {
            return "TestForLoop";
        }
    };

    struct TestSubDimension
    {
        std::string toString() const
        {
            return "TestSubDimension";
        }
    };
    struct TestUser
    {
        std::string toString() const
        {
            return "TestUser";
        }
    };
    struct TestVGPR
    {
        std::string toString() const
        {
            return "TestVGPR";
        }
    };

    using TestDimension = std::variant<TestForLoop, TestSubDimension, TestUser, TestVGPR>;

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
    };
    struct TestSplit
    {
        std::string toString() const
        {
            return "TestSplit";
        }
    };

    using TestTransform = std::variant<TestForget, TestSplit>;

    template <typename T>
    requires(!std::same_as<TestTransform,
                           std::decay_t<T>> && std::constructible_from<TestTransform, T>) bool
        operator==(T const& lhs, T const& rhs)
    {
        // Since none of these have any members, if the types are the same, they are equal.
        return true;
    }

    using myHypergraph = Graph::Hypergraph<TestDimension, TestTransform>;

    TEST(HypergraphTest, Basic)
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

            EXPECT_EQ(NormalizedSource(expected), NormalizedSource(g.toDOT()));
        }

        {
            EXPECT_NO_THROW(std::get<TestUser>(std::get<TestDimension>(g.getElement(u0))));
            EXPECT_NO_THROW(
                std::get<TestForget>(std::get<TestTransform>(g.getElement(TestForget1))));
        }

        {
            auto             nodes = g.depthFirstVisit(u0).to<std::vector>();
            std::vector<int> expectedNodes{
                u0, TestSplit0, sd0, TestForget0, TestVGPR0, TestForget1, TestVGPR1, sd1};
            EXPECT_EQ(expectedNodes, nodes);

            auto loc = g.getLocation(nodes[0]);
            EXPECT_EQ(u0, loc.index);
            EXPECT_TRUE(std::holds_alternative<TestDimension>(loc.element));
            EXPECT_TRUE(std::holds_alternative<TestUser>(std::get<TestDimension>(loc.element)));
            EXPECT_EQ(0, loc.incoming.size());
            EXPECT_EQ(std::vector<int>{TestSplit0}, loc.outgoing);

            auto loc2 = g.getLocation(u0);
            EXPECT_EQ(loc, loc2);

            loc = g.getLocation(nodes[1]);
            myHypergraph::Location expected{
                TestSplit0, {u0}, {sd0, sd1}, TestTransform{TestSplit{}}};
            EXPECT_TRUE(expected == loc);

            EXPECT_EQ(TestSplit0, loc.index);

            EXPECT_EQ(myHypergraph::Element{TestTransform{TestSplit{}}}, loc.element);
            EXPECT_EQ(std::vector<int>{u0}, loc.incoming);
            EXPECT_EQ((std::vector<int>{sd0, sd1}), loc.outgoing);

            EXPECT_EQ(std::vector<int>{u0}, g.parentNodes(sd0).to<std::vector>());
            EXPECT_EQ((std::vector<int>{sd0, sd1}), g.childNodes(u0).to<std::vector>());

            EXPECT_EQ(std::vector<int>{u0}, g.parentNodes(TestSplit0).to<std::vector>());
            EXPECT_EQ((std::vector<int>{sd0, sd1}), g.childNodes(TestSplit0).to<std::vector>());

            EXPECT_EQ((std::vector<int>{sd0, sd1}), g.parentNodes(TestVGPR0).to<std::vector>());
            EXPECT_EQ((std::vector<int>{TestVGPR0, TestVGPR1}),
                      g.childNodes(sd1).to<std::vector>());
        }

        {
            // Since there are multiple leaf nodes, we don't expect this to visit the entire graph.
            auto nodes = g.depthFirstVisit(TestVGPR0, Graph::Direction::Upstream).to<std::vector>();
            std::vector<int> expectedNodes{TestVGPR0, TestForget0, sd0, TestSplit0, u0, sd1};
            EXPECT_EQ(expectedNodes, nodes);
        }

        {
            // Visiting from all the leaf nodes gives us the whole graph.
            // TODO: "Make generators less lazy" once the generator semantics have been made less lazy, this can be collapsed into the next line and we can avoid converting the 'leaves' generator into a vector.
            auto leaves = g.leaves().to<std::vector>();
            auto nodes  = g.depthFirstVisit(leaves, Graph::Direction::Upstream).to<std::vector>();
            std::vector<int> expectedNodes{
                TestVGPR0, TestForget0, sd0, TestSplit0, u0, sd1, TestVGPR1, TestForget1};
            EXPECT_EQ(expectedNodes, nodes);
        }

        {
            auto             nodes = g.breadthFirstVisit(u0).to<std::vector>();
            std::vector<int> expectedNodes{
                u0, TestSplit0, sd0, sd1, TestForget0, TestForget1, TestVGPR0, TestVGPR1};
            EXPECT_EQ(expectedNodes, nodes);
        }

        {
            auto             nodes         = g.roots().to<std::vector>();
            std::vector<int> expectedNodes = {u0};
            EXPECT_EQ(expectedNodes, nodes);
        }

        {
            auto             nodes         = g.leaves().to<std::vector>();
            std::vector<int> expectedNodes = {TestVGPR0, TestVGPR1};
            EXPECT_EQ(expectedNodes, nodes);
        }

        {
            EXPECT_EQ((std::vector<int>{u0}),
                      g.getNeighbours<Graph::Direction::Upstream>(TestSplit0).to<std::vector>());
            EXPECT_EQ((std::vector<int>{TestSplit0}),
                      g.getNeighbours<Graph::Direction::Upstream>(sd0).to<std::vector>());
            EXPECT_EQ((std::vector<int>{TestSplit0}),
                      g.getNeighbours<Graph::Direction::Upstream>(sd1).to<std::vector>());
            EXPECT_EQ((std::vector<int>{sd0, sd1}),
                      g.getNeighbours<Graph::Direction::Downstream>(TestSplit0).to<std::vector>());
        }

        // Add a for loop.
        auto loop = g.addElement(TestForLoop{}, {TestSplit0}, {TestForget0});

        {
            auto             loc           = g.getLocation(TestSplit0);
            std::vector<int> expectedNodes = {sd0, sd1, loop};
            EXPECT_EQ(expectedNodes, loc.outgoing);
        }

        {
            auto             loc           = g.getLocation(TestForget0);
            std::vector<int> expectedNodes = {sd0, sd1, loop};
            EXPECT_EQ(expectedNodes, loc.incoming);
        }

        {
            auto             nodes = g.depthFirstVisit(u0).to<std::vector>();
            std::vector<int> expectedNodes{
                u0, TestSplit0, sd0, TestForget0, TestVGPR0, TestForget1, TestVGPR1, sd1, loop};
            EXPECT_EQ(expectedNodes, nodes);
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

            EXPECT_EQ(NormalizedSource(expected), NormalizedSource(g.toDOT()));
        }

        {
            EXPECT_EQ(std::get<TestUser>(std::get<TestDimension>(g.getElement(u0))), TestUser{});

            EXPECT_THROW(g.getElement(-1), FatalError);
        }
    }

    TEST(HypergraphTest, Path)
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

        std::map<int, bool> visited;
        EXPECT_EQ((std::vector<int>{u1, TestSplit4, sd2}),
                  g.path<Graph::Direction::Downstream>(
                       std::vector<int>{u1}, std::vector<int>{sd2}, visited)
                      .to<std::vector>());

        visited.clear();
        EXPECT_EQ((std::vector<int>{sd2, sd3, TestSplit4, u1}),
                  g.path<Graph::Direction::Upstream>(
                       std::vector<int>{sd2, sd3}, std::vector<int>{u1}, visited)
                      .to<std::vector>());

        visited.clear();
        EXPECT_EQ(
            (std::vector<int>{}),
            g.path<Graph::Direction::Upstream>(std::vector<int>{sd2}, std::vector<int>{u1}, visited)
                .to<std::vector>());

        visited.clear();
        EXPECT_EQ(
            (std::vector<int>{sd6, TestSplit10, sd8, TestSplit9, sd2, sd3, TestSplit4, u1}),
            g.path<Graph::Direction::Upstream>(std::vector<int>{sd6}, std::vector<int>{u1}, visited)
                .to<std::vector>());

        visited.clear();
        EXPECT_EQ((std::vector<int>{u1, TestSplit4, sd3, TestSplit11, sd7}),
                  g.path<Graph::Direction::Downstream>(
                       std::vector<int>{u1}, std::vector<int>{sd7}, visited)
                      .to<std::vector>());

        visited.clear();
        EXPECT_EQ((std::vector<int>{u1, TestSplit4, sd3, sd2, TestSplit9, sd8, TestSplit10, sd6}),
                  g.path<Graph::Direction::Downstream>(
                       std::vector<int>{u1}, std::vector<int>{sd6}, visited)
                      .to<std::vector>());
    }
}
