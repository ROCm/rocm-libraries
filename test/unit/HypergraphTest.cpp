
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
    };

    struct TestSubDimension
    {
    };
    struct TestUser
    {
    };
    struct TestVGPR
    {
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
    };
    struct TestSplit
    {
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
            std::string expected = R"(
                digraph {
                    "1"
                    "2"
                    "3"
                    "4"[shape=box]
                    "5"
                    "6"[shape=box]
                    "7"
                    "8"[shape=box]
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
            )";

            EXPECT_EQ(NormalizedSource(expected), NormalizedSource(g.toDOT()));
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
            std::string expected = R"(
                digraph {
                    "1"
                    "2"
                    "3"
                    "4"[shape=box]
                    "5"
                    "6"[shape=box]
                    "7"
                    "8"[shape=box]
                    "9"
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
            )";

            EXPECT_EQ(NormalizedSource(expected), NormalizedSource(g.toDOT()));
        }

        {
            EXPECT_EQ(std::get<TestUser>(std::get<TestDimension>(g.getElement(u0))), TestUser{});

            EXPECT_THROW(g.getElement(-1), FatalError);
        }
    }

}
