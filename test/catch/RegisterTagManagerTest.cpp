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

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>

#include <common/CommonGraphs.hpp>
#include <common/SourceMatcher.hpp>

#include "TestContext.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace rocRoller;

TEST_CASE("RegisterTagManager RegisterExpressionAttributes toString works",
          "[codegen][kernel-graph]")
{
    // clang-format off
    auto expected = R"({
    t.dataType = Int32
    t.unitStride = 0
    t.elementBlockSize = 0
    t.elementBlockStride = nullptr
    t.trLoadPairStride = nullptr
})";
    // clang-format on
    CHECK(NormalizedSource(toString(RegisterExpressionAttributes{DataType::Int32, false}))
          == NormalizedSource(expected));
}

SCENARIO("RegisterTagManager works", "[codegen][kernel-graph]")
{
    auto context = TestContext::ForDefaultTarget();

    RegisterTagManager manager(context.get());

    GIVEN("An empty manager")
    {
        THEN("Attempting to access any values will throw.")
        {
            CHECK_THROWS(manager.getRegister(5));
            CHECK_THROWS(manager.getExpression(3));
        }

        WHEN("A register is added")
        {
            auto reg = Register::Value::Placeholder(
                context.get(), Register::Type::Vector, DataType::Int32, 1);
            manager.addRegister(8, reg);

            auto reg2 = Register::Value::Placeholder(
                context.get(), Register::Type::Vector, DataType::Int32, 2);
            manager.addRegister(9, reg2);

            THEN("We can access that value via getRegister.")
            {
                CHECK(manager.getRegister(8) == reg);

                CHECK(manager.getRegister(8, reg->placeholder()) == reg);

                CHECK(manager.getRegister(8, Register::Type::Vector, DataType::Int32, 1, {})
                      == reg);
            }

            AND_THEN("Bad accesses still fail.")
            {
                CHECK_THROWS(manager.getRegister(5));
                CHECK_THROWS(manager.getExpression(3));
                CHECK_THROWS(manager.getExpression(8));

                {
                    auto reg3 = Register::Value::Placeholder(
                        context.get(), Register::Type::Scalar, DataType::Int32, 1);
                    CHECK_THROWS(manager.getRegister(8, reg3));
                }
                {
                    auto reg3 = Register::Value::Placeholder(
                        context.get(), Register::Type::Vector, DataType::Float, 1);
                    CHECK_THROWS(manager.getRegister(8, reg3));
                }
                {
                    auto reg3 = Register::Value::Placeholder(
                        context.get(), Register::Type::Vector, DataType::Int32, 2);
                    CHECK_THROWS(manager.getRegister(8, reg3));
                }

                CHECK_THROWS(
                    manager.getRegister(8, Register::Type::Scalar, DataType::Int32, 1, {}));
                CHECK_THROWS(
                    manager.getRegister(8, Register::Type::Vector, DataType::Int64, 1, {}));
                CHECK_THROWS(
                    manager.getRegister(8, Register::Type::Vector, DataType::Int32, 2, {}));
            }

            AND_WHEN("A register is allocated")
            {
                reg->allocateNow();
                int regIdx = reg->allocation()->registerIndices().at(0);

                THEN("The manager shows the allocation")
                {
                    CHECK(manager.getRegister(8)->allocationState()
                          == Register::AllocationState::Allocated);

                    CHECK(manager.getRegister(8)->allocation()->registerIndices().at(0) == regIdx);

                    CHECK_FALSE(context->allocator(Register::Type::Vector)->isFree(regIdx));
                }

                AND_WHEN("A register is deleted")
                {
                    reg.reset();
                    manager.deleteTag(8);

                    THEN("The register is freed and can no longer be accessed.")
                    {
                        CHECK(context->allocator(Register::Type::Vector)->isFree(regIdx));
                        CHECK_THROWS(manager.getRegister(8));
                        CHECK_THROWS(manager.getExpression(8));
                    }
                }
            }
        }

        WHEN("A register is added by template")
        {
            auto reg = Register::Value::Placeholder(
                context.get(), Register::Type::Scalar, DataType::BFloat16, 4);

            auto reg2 = manager.getRegister(8, reg);

            THEN("The accessed register has the same properties but is different.")
            {
                CHECK(reg2 != reg);
                CHECK(reg2->regType() == Register::Type::Scalar);
                CHECK(reg2->variableType() == DataType::BFloat16);
                CHECK(reg2->valueCount() == 4);

                CHECK(manager.getRegister(8) == reg2);
                CHECK(manager.getRegister(8, reg) == reg2);
                CHECK(manager.getRegister(8, reg2) == reg2);
            }
        }

        WHEN("Expressions are added")
        {
            auto exp1 = Expression::literal(5u) + Expression::literal(2u);
            auto exp2 = Expression::literal(1) + Expression::literal(9);

            manager.addExpression(8, exp1, {DataType::UInt32, false});
            manager.addExpression(3, exp2, {DataType::Int32, true});

            THEN("We can access those expressions.")
            {
                {
                    auto const& [exp, info] = manager.getExpression(8);

                    CHECK(exp1.get() == exp.get());
                    CHECK(exp2.get() != exp.get());
                    CHECK(info == RegisterExpressionAttributes{DataType::UInt32, false});
                }
                {
                    auto const& [exp, info] = manager.getExpression(3);

                    CHECK(exp1.get() != exp.get());
                    CHECK(exp2.get() == exp.get());
                    CHECK(info == RegisterExpressionAttributes{DataType::Int32, true});
                }
            }

            AND_THEN("Bad accesses still fail.")
            {
                CHECK_THROWS(manager.getExpression(7));
                CHECK_THROWS(manager.getExpression(1));
            }

            AND_WHEN("An expression is deleted")
            {
                manager.deleteTag(3);

                THEN("That expression can no longer be accessed.")
                {
                    CHECK_THROWS(manager.getExpression(3));
                    CHECK_THROWS(manager.getRegister(3));
                }
            }
        }
    }
}

SCENARIO("RegisterTagManager aliasing works", "[codegen][kernel-graph]")
{
    auto context = TestContext::ForDefaultTarget();

    RegisterTagManager manager(context.get());

    manager.addAlias(3, 4);
    manager.addAlias(9, 1);
    manager.addAlias(5, 1);

    GIVEN("An empty manager with aliases")
    {
        THEN("Getters for alias-related properties return the correct answer")
        {
            CHECK(manager.isAliased(4));
            CHECK_FALSE(manager.hasAlias(4));
            CHECK_FALSE(manager.isBorrowed(4));

            CHECK_FALSE(manager.isAliased(3));
            CHECK(manager.hasAlias(3));
            CHECK_FALSE(manager.isBorrowed(3));

            CHECK_FALSE(manager.isAliased(5));
            CHECK(manager.hasAlias(5));
            CHECK_FALSE(manager.isBorrowed(5));

            CHECK(manager.isAliased(1));
            CHECK_FALSE(manager.hasAlias(1));
            CHECK_FALSE(manager.isBorrowed(1));

            CHECK_FALSE(manager.isAliased(9));
            CHECK(manager.hasAlias(9));
            CHECK_FALSE(manager.isBorrowed(9));
        }

        THEN("Duplicate aliases can't be added.")
        {
            // A -> C and B -> C can coexist, but A->B and B->C can't.

            CHECK_THROWS(manager.addAlias(3, 4));
            CHECK_THROWS(manager.addAlias(3, 5));
            CHECK_THROWS(manager.addAlias(4, 5));
            CHECK_THROWS(manager.addAlias(9, 2));
            CHECK_THROWS(manager.addAlias(1, 2));
        }

        AND_THEN("Aliasing registers fail.")
        {
            CHECK_THROWS(manager.getRegister(3));
            CHECK_THROWS(manager.getRegister(5));
            CHECK_THROWS(manager.getRegister(9));
        }

        WHEN("An aliased register is added")
        {
            auto reg = Register::Value::Placeholder(
                context.get(), Register::Type::Vector, DataType::Int32, 1);
            reg = manager.getRegister(1, reg);
            reg->allocateNow();
            auto regIdx = reg->registerIndices().only().value();

            THEN("Aliased and non-aliased registers can be used normally.")
            {
                auto reg2 = Register::Value::Placeholder(
                    context.get(), Register::Type::Vector, DataType::Int32, 2);
                reg2 = manager.getRegister(8, reg2);

                CHECK(manager.getRegister(1) == reg);
                CHECK(manager.getRegister(8) == reg2);
            }

            AND_THEN("The aliasing register can't be added with the wrong type.")
            {
                auto tmpl = Register::Value::Placeholder(
                    context.get(), Register::Type::Scalar, DataType::Int32, 1);

                CHECK_THROWS(manager.getRegister(9, tmpl));

                tmpl = Register::Value::Placeholder(
                    context.get(), Register::Type::Vector, DataType::Int32, 2);

                CHECK_THROWS(manager.getRegister(9, tmpl));
            }

            AND_WHEN("The aliasing register is added")
            {
                auto tmpl = Register::Value::Placeholder(
                    context.get(), Register::Type::Vector, DataType::Int32, 1);

                auto regOut = manager.getRegister(9, tmpl);

                CHECK(manager.isBorrowed(1));

                THEN("The aliasing register shares the allocation with the borrowed register.")
                {
                    CHECK(regOut != tmpl);
                    CHECK(regOut->registerIndices().only().value() == regIdx);

                    auto regOut2 = manager.getRegister(9);
                    CHECK(regOut == regOut2);

                    AND_THEN("Other aliasing registers still can't be accessed.")
                    {
                        CHECK_THROWS(manager.getRegister(3));
                        CHECK_THROWS(manager.getRegister(5));
                    }

                    AND_THEN("We can't set a conflicting aliasing register.")
                    {
                        CHECK_THROWS(manager.getRegister(5, tmpl));
                    }
                }

                AND_THEN("The borrowed register can't be accessed.")
                {
                    CHECK_THROWS(manager.getRegister(1));
                }

                WHEN("The aliasing register is deleted")
                {
                    manager.deleteTag(9);

                    THEN("The borrowed register can be accessed again.")
                    {
                        CHECK_FALSE(manager.isBorrowed(1));
                        CHECK(manager.getRegister(1) == reg);
                    }

                    AND_THEN("The conflicting register can be set.")
                    {
                        auto tmpl = Register::Value::Placeholder(
                            context.get(), Register::Type::Vector, DataType::UInt32, 1);

                        auto regOut = manager.getRegister(5, tmpl);

                        CHECK(manager.isBorrowed(1));

                        CHECK(regOut != tmpl);
                        CHECK(regOut->registerIndices().only().value() == regIdx);

                        CHECK_THROWS(manager.getRegister(1));

                        AND_WHEN("The conflicting register is deleted")
                        {
                            manager.deleteTag(5);

                            THEN("The borrowed register can be accessed again.")
                            {
                                CHECK_FALSE(manager.isBorrowed(1));
                                CHECK(manager.getRegister(1) == reg);
                            }
                        }
                    }
                }
            }
        }
    }
}

SCENARIO("RegisterTagManager initialize adds alias edges.", "[codegen][kernel-graph]")
{
    auto context = TestContext::ForDefaultTarget();

    using namespace rocRollerTest;
    namespace CT = rocRoller::KernelGraph::CoordinateGraph;

    auto vec = Graphs::VectorAdd<int>(true);

    auto kgraph = vec.getKernelGraph();

    RegisterTagManager manager(context.get());

    GIVEN("A kernel graph with no alias edges")
    {
        {
            std::ofstream file("vecadd.dot");
            file << kgraph.toDOT() << std::endl;
        }

        THEN("initialize() will not register any aliases.")
        {
            manager.initialize(kgraph);

            for(auto node : kgraph.coordinates.getNodes())
            {
                CHECK_FALSE(manager.isAliased(node));
                CHECK_FALSE(manager.hasAlias(node));
            }
        }
    }

    AND_GIVEN("A kernel graph with some alias edges")
    {
        auto vgprs = kgraph.coordinates.getNodes<CT::VGPR>().to<std::vector>();
        REQUIRE(vgprs.size() == 2);
        auto v0 = vgprs[0];
        auto v1 = vgprs[1];

        auto linears = kgraph.coordinates.getNodes<CT::Linear>().to<std::vector>();
        REQUIRE(linears.size() == 5);
        // CHECK(linears == std::vector<int>{});

        auto l2 = linears[2];
        auto l3 = linears[3];
        auto l4 = linears[4];

        // CHECK()
        kgraph.coordinates.addElement(CT::Alias{}, {v0}, {v1});

        kgraph.coordinates.addElement(CT::Alias{}, {l2}, {l4});
        kgraph.coordinates.addElement(CT::Alias{}, {l3}, {l4});

        THEN("initialize() will register the aliases")
        {
            manager.initialize(kgraph);

            CHECK_FALSE(manager.isAliased(v0));
            CHECK(manager.isAliased(v1));

            CHECK(manager.hasAlias(v0));
            CHECK_FALSE(manager.hasAlias(v1));

            CHECK(manager.getAlias(v0) == v1);

            CHECK_FALSE(manager.isAliased(l2));
            CHECK_FALSE(manager.isAliased(l3));
            CHECK(manager.isAliased(l4));

            CHECK(manager.hasAlias(l2));
            CHECK(manager.hasAlias(l3));
            CHECK_FALSE(manager.hasAlias(l4));

            CHECK(manager.getAlias(l2) == l4);
            CHECK(manager.getAlias(l3) == l4);
        }
    }

    AND_GIVEN("Kernel graphs with some bad Alias edges")
    {
        auto vgprs = kgraph.coordinates.getNodes<CT::VGPR>().to<std::vector>();
        REQUIRE(vgprs.size() == 2);
        auto v0 = vgprs[0];
        auto v1 = vgprs[1];

        auto linears = kgraph.coordinates.getNodes<CT::Linear>().to<std::vector>();
        REQUIRE(linears.size() == 5);
        // CHECK(linears == std::vector<int>{});

        auto l2 = linears[2];
        auto l3 = linears[3];
        auto l4 = linears[4];

        THEN("initialize() will throw with bad dest.")
        {
            kgraph.coordinates.addElement(CT::Alias{}, {v0}, {v1, l2});
            CHECK_THROWS(manager.initialize(kgraph));
        }

        THEN("initialize() will throw with bad src.")
        {
            kgraph.coordinates.addElement(CT::Alias{}, {v0, v1}, {l2});
            CHECK_THROWS(manager.initialize(kgraph));
        }
    }
}
