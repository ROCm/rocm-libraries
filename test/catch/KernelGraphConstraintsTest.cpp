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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelGraph/Constraints.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

TEST_CASE("Constraint status can be combined", "[kernel-graph]")
{
    rocRoller::KernelGraph::ConstraintStatus c1;
    CHECK(c1.satisfied);
    CHECK(c1.explanation == "");

    c1.combine(true, "TEST1");

    CHECK(c1.satisfied);
    CHECK(c1.explanation == "TEST1");

    c1.combine(false, "TEST2");

    CHECK_FALSE(c1.satisfied);
    CHECK(c1.explanation == "TEST1\nTEST2");

    c1.combine(true, "");

    CHECK_FALSE(c1.satisfied);
    CHECK(c1.explanation == "TEST1\nTEST2");

    rocRoller::KernelGraph::ConstraintStatus c2;
    c2.combine(c1);

    CHECK_FALSE(c2.satisfied);
    CHECK(c2.explanation == "TEST1\nTEST2");
}

TEST_CASE("Empty constraints pass and have an empty explanation", "[kernel-graph]")
{
    rocRoller::KernelGraph::KernelGraph                  kgraph;
    std::vector<rocRoller::KernelGraph::GraphConstraint> emptyConstraints;

    auto check = kgraph.checkConstraints(emptyConstraints);
    CHECK(check.satisfied);
    CHECK(check.explanation == "");
}

SCENARIO("SingleControlRoot constraint works", "[kernel-graph]")
{
    using namespace Catch::Matchers;
    namespace kg = rocRoller::KernelGraph;
    GIVEN("An empty control graph")
    {
        kg::KernelGraph kgraph;

        std::vector<rocRoller::KernelGraph::GraphConstraint> constraints{kg::SingleControlRoot};

        THEN("The constraint fails.")
        {
            auto check = kgraph.checkConstraints(constraints);
            CHECK_FALSE(check.satisfied);
            CHECK_THAT(check.explanation,
                       ContainsSubstring("Single Control Root")
                           && ContainsSubstring("one root node"));
        }

        WHEN("A first root node is added")
        {
            auto kernel = kgraph.control.addElement(kg::ControlGraph::Kernel{});

            THEN("The constraint passes.")
            {
                auto check = kgraph.checkConstraints(constraints);
                CHECK(check.satisfied);
                CHECK(check.explanation == "");
            }

            WHEN("A second root node is added")
            {
                auto assign = kgraph.control.addElement(kg::ControlGraph::Assign{});

                THEN("The constraint fails.")
                {
                    auto check2 = kgraph.checkConstraints(constraints);
                    CHECK_FALSE(check2.satisfied);
                    CHECK_THAT(check2.explanation,
                               ContainsSubstring("Single Control Root")
                                   && ContainsSubstring("one root node"));
                }

                WHEN("The second node is no longer a root node")
                {
                    kgraph.control.addElement(kg::ControlGraph::Body{}, {kernel}, {assign});

                    THEN("The constraint passes.")
                    {
                        auto check = kgraph.checkConstraints(constraints);
                        CHECK(check.satisfied);
                        CHECK(check.explanation == "");
                    }
                }
            }
        }
    }
}

SCENARIO("NoDanglingMappings constraint works", "[kernel-graph]")
{
    using namespace rocRoller;
    namespace kg = rocRoller::KernelGraph;
    using namespace Catch::Matchers;

    GIVEN("An empty graph")
    {
        kg::KernelGraph                  kgraph;
        std::vector<kg::GraphConstraint> danglingMapping{&kg::NoDanglingMappings};

        THEN("The constraint passes.")
        {
            auto check = kgraph.checkConstraints(danglingMapping);
            CHECK(check.satisfied);
            CHECK(check.explanation == "");
        }

        WHEN("An unconnected mapping is added")
        {

            kgraph.mapper.connect(1, 1, NaryArgument::DEST);

            THEN("The constraint fails.")
            {
                auto check = kgraph.checkConstraints(danglingMapping);
                CHECK_FALSE(check.satisfied);
                CHECK_THAT(check.explanation,
                           ContainsSubstring("Dangling Mapping")
                               && ContainsSubstring("Control node 1 does not exist")
                               && ContainsSubstring("coordinate node 1"));
            }

            WHEN("The matching control node is added")
            {
                auto kernel = kgraph.control.addElement(kg::ControlGraph::Kernel{});
                REQUIRE(kernel == 1);

                THEN("The constraint still fails because the coordinate side is "
                     "still not connected.")
                {
                    auto check = kgraph.checkConstraints(danglingMapping);
                    CHECK_FALSE(check.satisfied);
                    CHECK_THAT(check.explanation,
                               ContainsSubstring("Dangling Mapping")
                                   && !ContainsSubstring("Control node 1 does not exist")
                                   && ContainsSubstring("coordinate node 1"));
                }

                WHEN("The matching coordinate node is added")
                {
                    auto user = kgraph.coordinates.addElement(kg::CoordinateGraph::User{});
                    REQUIRE(user == 1);

                    THEN("The constraint passes.")
                    {
                        auto check = kgraph.checkConstraints(danglingMapping);
                        CHECK(check.satisfied);
                        CHECK(check.explanation == "");
                    }
                }
            }

            WHEN("The matching coordinate node is added")
            {
                auto user = kgraph.coordinates.addElement(kg::CoordinateGraph::User{});
                REQUIRE(user == 1);

                THEN("The constraint still fails because the control side is "
                     "still not connected.")
                {
                    auto check = kgraph.checkConstraints(danglingMapping);
                    CHECK_FALSE(check.satisfied);
                    CHECK_THAT(check.explanation,
                               ContainsSubstring("Dangling Mapping")
                                   && ContainsSubstring("Control node 1 does not exist.")
                                   && !ContainsSubstring("coordinate node 1"));
                }
            }
        }
    }
}
