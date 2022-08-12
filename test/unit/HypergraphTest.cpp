
#include <compare>
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "SourceMatcher.hpp"

#include <rocRoller/Graph/Hypergraph.hpp>

namespace rocRollerTest
{
    using namespace rocRoller;

    struct ForLoop
    {
    };

    struct SubDimension
    {
    };
    struct User
    {
    };
    struct VGPR
    {
    };

    using Dimension = std::variant<ForLoop, SubDimension, User, VGPR>;

    template <typename T>
    requires(
        !std::same_as<Dimension, std::decay_t<T>> && std::constructible_from<Dimension, T>) bool
        operator==(T const& lhs, T const& rhs)
    {
        // Since none of these have any members, if the types are the same, they are equal.
        return true;
    }

    struct Forget
    {
    };
    struct Split
    {
    };

    using Transform = std::variant<Forget, Split>;

    template <typename T>
    requires(
        !std::same_as<Transform, std::decay_t<T>> && std::constructible_from<Transform, T>) bool
        operator==(T const& lhs, T const& rhs)
    {
        // Since none of these have any members, if the types are the same, they are equal.
        return true;
    }

    using myHypergraph = Graph::Hypergraph<Dimension, Transform>;

    TEST(HypergraphTest, Basic)
    {

        myHypergraph g;

        auto u0  = g.addElement(User{});
        auto sd0 = g.addElement(SubDimension{});
        auto sd1 = g.addElement(SubDimension{});

        auto split0 = g.addElement(Split{}, {u0}, {sd0, sd1});

        auto vgpr0   = g.addElement(VGPR{});
        auto forget0 = g.addElement(Forget{}, {sd0, sd1}, {vgpr0});

        auto vgpr1   = g.addElement(VGPR{});
        auto forget1 = g.addElement(Forget{}, {sd0, sd1}, {vgpr1});

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
            std::vector<int> expectedNodes{u0, split0, sd0, forget0, vgpr0, forget1, vgpr1, sd1};
            EXPECT_EQ(expectedNodes, nodes);

            auto loc = g.getLocation(nodes[0]);
            EXPECT_EQ(u0, loc.index);
            EXPECT_TRUE(std::holds_alternative<Dimension>(loc.element));
            EXPECT_TRUE(std::holds_alternative<User>(std::get<Dimension>(loc.element)));
            EXPECT_EQ(0, loc.incoming.size());
            EXPECT_EQ(std::vector<int>{split0}, loc.outgoing);

            auto loc2 = g.getLocation(u0);
            EXPECT_EQ(loc, loc2);

            loc = g.getLocation(nodes[1]);
            myHypergraph::Location expected{split0, {u0}, {sd0, sd1}, Transform{Split{}}};
            EXPECT_TRUE(expected == loc);

            EXPECT_EQ(split0, loc.index);

            EXPECT_EQ(myHypergraph::Element{Transform{Split{}}}, loc.element);
            EXPECT_EQ(std::vector<int>{u0}, loc.incoming);
            EXPECT_EQ((std::vector<int>{sd0, sd1}), loc.outgoing);
        }

        {
            // Since there are multiple leaf nodes, we don't expect this to visit the entire graph.
            auto nodes = g.depthFirstVisit(vgpr0, Graph::Direction::Upstream).to<std::vector>();
            std::vector<int> expectedNodes{vgpr0, forget0, sd0, split0, u0, sd1};
            EXPECT_EQ(expectedNodes, nodes);
        }

        {
            // Visiting from all the leaf nodes gives us the whole graph.
            // TODO: "Make generators less lazy" once the generator semantics have been made less lazy, this can be collapsed into the next line and we can avoid converting the 'leaves' generator into a vector.
            auto leaves = g.leaves().to<std::vector>();
            auto nodes  = g.depthFirstVisit(leaves, Graph::Direction::Upstream).to<std::vector>();
            std::vector<int> expectedNodes{vgpr0, forget0, sd0, split0, u0, sd1, vgpr1, forget1};
            EXPECT_EQ(expectedNodes, nodes);
        }

        {
            auto             nodes = g.breadthFirstVisit(u0).to<std::vector>();
            std::vector<int> expectedNodes{u0, split0, sd0, sd1, forget0, forget1, vgpr0, vgpr1};
            EXPECT_EQ(expectedNodes, nodes);
        }

        {
            auto             nodes         = g.roots().to<std::vector>();
            std::vector<int> expectedNodes = {u0};
            EXPECT_EQ(expectedNodes, nodes);
        }

        {
            auto             nodes         = g.leaves().to<std::vector>();
            std::vector<int> expectedNodes = {vgpr0, vgpr1};
            EXPECT_EQ(expectedNodes, nodes);
        }

        // Add a for loop.
        auto loop = g.addElement(ForLoop{}, {split0}, {forget0});

        {
            auto             loc           = g.getLocation(split0);
            std::vector<int> expectedNodes = {sd0, sd1, loop};
            EXPECT_EQ(expectedNodes, loc.outgoing);
        }

        {
            auto             loc           = g.getLocation(forget0);
            std::vector<int> expectedNodes = {sd0, sd1, loop};
            EXPECT_EQ(expectedNodes, loc.incoming);
        }

        {
            auto             nodes = g.depthFirstVisit(u0).to<std::vector>();
            std::vector<int> expectedNodes{
                u0, split0, sd0, forget0, vgpr0, forget1, vgpr1, sd1, loop};
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
    }

}
