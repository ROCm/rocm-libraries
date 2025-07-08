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

#include <cmath>
#include <memory>

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "ExpressionMatchers.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <catch2/catch_test_macros.hpp>

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <common/SourceMatcher.hpp>
#include <common/TestValues.hpp>

using namespace rocRoller;

namespace ExpressionTest
{
    TEST_CASE("Expression identical and equivalent", "[expression]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto a    = Expression::literal(1u);
        auto ap   = Expression::literal(1);
        auto b    = Expression::literal(2u);
        auto zero = Expression::literal(0u);

        auto rc = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        rc->allocateNow();

        auto rd = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 1);
        rd->allocateNow();

        auto c = rc->expression();
        auto d = rd->expression();

        auto cve = std::make_shared<CommandArgument>(nullptr, DataType::Float, 0);
        auto cvf = std::make_shared<CommandArgument>(nullptr, DataType::Float, 8);

        auto e = std::make_shared<Expression::Expression>(cve);
        auto f = std::make_shared<Expression::Expression>(cvf);

        auto expr1 = a + b;
        auto expr2 = a + b;

        auto expr3 = a - b;

        CHECK(identical(expr1, expr2));
        CHECK_FALSE(identical(expr1, expr3));
        CHECK_FALSE(identical(ap + b, expr3));

        CHECK_THAT(expr1, IdenticalTo(expr2));
        CHECK_THAT(expr1, !IdenticalTo(expr3));
        CHECK_THAT(ap + b, !IdenticalTo(expr3));

        CHECK(equivalent(expr1, expr2));
        CHECK_FALSE(equivalent(expr1, expr3));
        CHECK_FALSE(equivalent(ap + b, expr3));

        CHECK_THAT(expr1, EquivalentTo(expr2));
        CHECK_THAT(expr1, !EquivalentTo(expr3));
        CHECK_THAT(ap + b, !EquivalentTo(expr3));
        CHECK_THAT(b + ap, !EquivalentTo(expr3));

        auto expr4 = c + convert(DataType::UInt32, d);
        auto expr5 = c + convert(DataType::UInt32, d) + zero;

        CHECK_FALSE(identical(expr1, expr4));
        CHECK_FALSE(identical(expr4, expr5));
        CHECK(identical(expr4, simplify(expr5)));

        CHECK_THAT(expr1, !IdenticalTo(expr4));
        CHECK_THAT(expr4, !IdenticalTo(expr5));
        CHECK_THAT(simplify(expr5), IdenticalTo(expr4));

        CHECK_FALSE(equivalent(expr1, expr4));
        CHECK_FALSE(equivalent(expr4, expr5));
        CHECK(equivalent(expr4, simplify(expr5)));

        auto expr6 = e / f % d;
        auto expr7 = convert(DataType::Float, a) + f;

        CHECK_FALSE(identical(expr6, expr7));
        CHECK_FALSE(identical(e, f));

        CHECK_THAT(expr6, !IdenticalTo(expr7));
        CHECK_THAT(e, !IdenticalTo(f));

        CHECK(Expression::identical(nullptr, nullptr));
        CHECK_FALSE(identical(nullptr, a));
        CHECK_FALSE(identical(a, nullptr));

        CHECK_THAT(nullptr, IdenticalTo(nullptr));
        CHECK_THAT(nullptr, !IdenticalTo(a));
        CHECK_THAT(a, !IdenticalTo(nullptr));

        CHECK_FALSE(equivalent(expr6, expr7));
        CHECK_FALSE(equivalent(e, f));

        CHECK_THAT(expr6, !EquivalentTo(expr7));
        CHECK_THAT(e, !EquivalentTo(f));

        CHECK(Expression::equivalent(nullptr, nullptr));
        CHECK_FALSE(equivalent(nullptr, a));
        CHECK_FALSE(equivalent(a, nullptr));

        // Commutative tests
        CHECK_FALSE(identical(a + b, b + a));
        CHECK_FALSE(identical(a - b, b - a));

        CHECK(equivalent(a + b, b + a));
        CHECK_FALSE(equivalent(a - b, b - a));
        CHECK(equivalent(a * b, b * a));
        CHECK_FALSE(equivalent(a / b, b / a));
        CHECK_FALSE(equivalent(a % b, b % a));
        CHECK_FALSE(equivalent(a << b, b << a));
        CHECK_FALSE(equivalent(a >> b, b >> a));
        CHECK(equivalent(a & b, b & a));
        CHECK(equivalent(a | b, b | a));
        CHECK(equivalent(a ^ b, b ^ a));

        // Unallocated
        auto rg = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);

        // Unallocated
        auto rh = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Int32, 1);

        CHECK(Expression::identical(rg->expression(), rg->expression()));
        CHECK_FALSE(Expression::identical(rg->expression(), rh->expression()));

        CHECK(Expression::equivalent(rg->expression(), rg->expression()));
        CHECK_FALSE(Expression::equivalent(rg->expression(), rh->expression()));

        // Null
        Expression::ExpressionPtr n = nullptr;
        CHECK_FALSE(Expression::equivalent(n + n, a + n));
        CHECK_FALSE(Expression::equivalent(n + n, n + a));
    }

    TEST_CASE("Expression contains and split", "[expression]")
    {
        using Expression::literal;

        auto context = TestContext::ForDefaultTarget();

        auto lit = literal(5);

        auto reg = [&]() {
            auto rc = std::make_shared<Register::Value>(
                context.get(), Register::Type::Vector, DataType::Int32, 1);
            rc->allocateNow();

            return rc->expression();
        }();

        auto commandArg = []() {
            auto command = std::make_shared<Command>();
            auto scalar  = command->addOperation(Operations::Scalar(DataType::Int64));
            auto commandArgPtr
                = command->allocateArgument(DataType::Int64, scalar, ArgumentType::Value);

            return commandArgPtr->expression();
        }();

        auto kernelArg = []() {
            auto kernelArgPtr = std::make_shared<AssemblyKernelArgument>(
                AssemblyKernelArgument{"arg", DataType::Int32});
            return std::make_shared<Expression::Expression>(kernelArgPtr);
        }();

        auto dataFlowTag = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{4, Register::Type::Vector, DataType::Int16});

        auto waveTile = []() {
            auto waveTilePtr = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
            return std::make_shared<Expression::Expression>(waveTilePtr);
        }();

        SECTION("Simple")
        {
            auto expr = reg + commandArg;

            CHECK(contains<Expression::Add>(expr));
            CHECK_THAT(expr, Contains<Expression::Add>());

            CHECK_FALSE(contains<Expression::Subtract>(expr));
            CHECK_THAT(expr, !Contains<Expression::Subtract>());

            CHECK_FALSE(contains<Expression::Negate>(expr));
            CHECK_THAT(expr, !Contains<Expression::Negate>());

            CHECK(contains<Register::ValuePtr>(expr));
            CHECK_THAT(expr, Contains<Register::ValuePtr>());

            CHECK(contains<CommandArgumentPtr>(expr));
            CHECK_THAT(expr, Contains<CommandArgumentPtr>());

            CHECK_FALSE(contains<CommandArgumentValue>(expr));
            CHECK_THAT(expr, !Contains<CommandArgumentValue>());

            CHECK_FALSE(contains<Expression::WaveTilePtr>(expr));
            CHECK_THAT(expr, !Contains<Expression::WaveTilePtr>());

            SECTION("Split")
            {
                auto [lhs, rhs] = split<Expression::Add>(expr);

                CHECK_THAT(lhs, IdenticalTo(reg));
                CHECK_THAT(rhs, IdenticalTo(commandArg));

                CHECK_THROWS_AS(split<Expression::Subtract>(expr), FatalError);
            }
        }

        SECTION("More complex")
        {
            auto expr = (reg + commandArg) * -lit;

            CHECK_THAT(expr, Contains<Expression::Add>());
            CHECK_THAT(expr, !Contains<Expression::Subtract>());
            CHECK_THAT(expr, Contains<Expression::Multiply>());
            CHECK_THAT(expr, Contains<Expression::Negate>());
            CHECK_THAT(expr, !Contains<Expression::MatrixMultiply>());
            CHECK_THAT(expr, Contains<CommandArgumentPtr>());
            CHECK_THAT(expr, Contains<CommandArgumentValue>());

            SECTION("Split")
            {
                auto [lhs, rhs] = split<Expression::Multiply>(expr);

                CHECK_THAT(lhs, IdenticalTo(reg + commandArg));
                CHECK_THAT(rhs, IdenticalTo(-lit));

                CHECK_THROWS_AS(split<Expression::Subtract>(expr), FatalError);

                auto [llhs, lrhs] = split<Expression::Add>(lhs);
                CHECK_THAT(llhs, IdenticalTo(reg));
                CHECK_THAT(lrhs, IdenticalTo(commandArg));

                auto [irhs] = split<Expression::Negate>(rhs);
                CHECK_THAT(irhs, IdenticalTo(lit));
            }
        }

        SECTION("Complex")
        {
            auto expr = (dataFlowTag & lit << ~kernelArg) < (commandArg ^ lit);

            CHECK_THAT(expr, !Contains<Expression::Add>());

            CHECK_THAT(expr, Contains<Expression::BitwiseAnd>());

            CHECK_THAT(expr, Contains<Expression::BitwiseXor>());

            CHECK_THAT(expr, Contains<Expression::LessThan>());

            CHECK_THAT(expr, Contains<Expression::ShiftL>());

            CHECK_THAT(expr, Contains<CommandArgumentPtr>());

            CHECK_THAT(expr, Contains<CommandArgumentValue>());

            CHECK_THAT(expr, Contains<Expression::DataFlowTag>());

            CHECK_THAT(expr, Contains<AssemblyKernelArgumentPtr>());
        }

        SECTION("Matrix Multiply")
        {
            auto expr = std::make_shared<Expression::Expression>(
                Expression::MatrixMultiply(waveTile, waveTile, waveTile));

            CHECK_THAT(expr, !Contains<Expression::Add>());

            CHECK_THAT(expr, Contains<Expression::MatrixMultiply>());

            CHECK_THAT(expr, Contains<Expression::WaveTilePtr>());
        }

        SECTION("Ternary")
        {
            auto expr = conditional(reg < lit,
                                    addShiftL(reg, dataFlowTag + reg, lit),
                                    multiplyAdd(reg, lit, dataFlowTag));

            CHECK_THAT(expr, !Contains<Expression::BitwiseAnd>());
            CHECK_THAT(expr, !Contains<Expression::BitwiseXor>());

            CHECK_THAT(expr, Contains<Expression::Conditional>());
            CHECK_THAT(expr, Contains<Expression::LessThan>());
            CHECK_THAT(expr, Contains<Expression::AddShiftL>());
            CHECK_THAT(expr, Contains<Expression::Add>());
            CHECK_THAT(expr, Contains<Expression::MultiplyAdd>());

            CHECK_THAT(expr, !Contains<Expression::ShiftLAdd>());

            CHECK_THAT(expr, Contains<CommandArgumentValue>());
            CHECK_THAT(expr, Contains<Expression::DataFlowTag>());

            SECTION("Split")
            {
                auto [lhs, r1hs, r2hs] = split<Expression::Conditional>(expr);

                CHECK_THAT(lhs, IdenticalTo(reg < lit));
                CHECK_THAT(r1hs, IdenticalTo(addShiftL(reg, dataFlowTag + reg, lit)));
                CHECK_THAT(r2hs, IdenticalTo(multiplyAdd(reg, lit, dataFlowTag)));

                CHECK_THROWS_AS(split<Expression::AddShiftL>(expr), FatalError);

                SECTION("Split <")
                {
                    auto [compLhs, compRhs] = split<Expression::LessThan>(lhs);
                    CHECK_THAT(compLhs, IdenticalTo(reg));
                    CHECK_THAT(compRhs, IdenticalTo(lit));
                }

                SECTION("Split AddShiftL")
                {
                    auto [aslLhs, aslR1hs, aslR2hs] = split<Expression::AddShiftL>(r1hs);
                    CHECK_THAT(aslLhs, IdenticalTo(reg));
                    CHECK_THAT(aslR1hs, IdenticalTo(dataFlowTag + reg));
                    CHECK_THAT(aslR2hs, IdenticalTo(lit));
                }

                SECTION("Split MultiplyAdd")
                {
                    auto [fmaLhs, fmaR1hs, fmaR2hs] = split<Expression::MultiplyAdd>(r2hs);
                    CHECK_THAT(fmaLhs, IdenticalTo(reg));
                    CHECK_THAT(fmaR1hs, IdenticalTo(lit));
                    CHECK_THAT(fmaR2hs, IdenticalTo(dataFlowTag));
                }
            }
        }
    }
}
