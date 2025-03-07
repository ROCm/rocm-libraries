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

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "ExpressionMatchers.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace rocRoller;

namespace ExpressionTest
{
    TEST_CASE("Expression contains sub-expressions", "[expression][sub-expression]")
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
        {
            CHECK(containsSubExpression(expr1, a));
            CHECK(containsSubExpression(expr1, b));
            CHECK(containsSubExpression(expr1 + expr3, expr1));
            CHECK(containsSubExpression(expr1 + expr3, expr2)); // expr2 is equivalent to expr1
            CHECK(containsSubExpression(expr1, expr1));

            CHECK_FALSE(containsSubExpression(expr1, ap));
            CHECK_FALSE(containsSubExpression(expr1, e));
            CHECK_FALSE(containsSubExpression(expr1, f));
            CHECK_FALSE(containsSubExpression(expr1 + expr2, expr3));
        }

        auto expr4 = c + convert(DataType::UInt32, d);
        auto expr5 = c + convert(DataType::UInt32, d) + zero;
        {
            CHECK(containsSubExpression(expr4, c));
            CHECK(containsSubExpression(expr4, d));
            CHECK(containsSubExpression(expr5, zero));
            CHECK(containsSubExpression(expr5, expr4));
            CHECK(containsSubExpression(simplify(expr5), expr4));

            CHECK_FALSE(containsSubExpression(expr4, expr5));
        }

        auto expr6 = e / f % d;
        auto expr7 = convert(DataType::Float, a) + f;
        {
            CHECK(containsSubExpression(expr6, e));
            CHECK(containsSubExpression(expr6, f));
            CHECK(containsSubExpression(expr6, d));
            CHECK(containsSubExpression(expr7, a));
            CHECK(containsSubExpression(expr7, f));
        }

        auto dft_int64 = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{-1, Register::Type::Vector, DataType::Int64});
        auto dft_int64_2 = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{2, Register::Type::Vector, DataType::Int64});
        auto dft_float = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{-1, Register::Type::Vector, DataType::Float});
        auto expr8 = expr7 + dft_int64;
        {
            CHECK(containsSubExpression(expr8, expr7));
            CHECK(containsSubExpression(expr8, dft_int64));

            CHECK_FALSE(containsSubExpression(expr8, dft_float));
            CHECK_FALSE(containsSubExpression(expr8, dft_int64_2));
        }

        auto waveTile = [](int rank) {
            auto waveTilePtr  = std::make_shared<KernelGraph::CoordinateGraph::WaveTile>();
            waveTilePtr->rank = rank;
            return std::make_shared<Expression::Expression>(waveTilePtr);
        };
        auto wt1    = waveTile(1);
        auto wt2    = waveTile(2);
        auto expr9  = expr8 - wt1;
        auto expr10 = expr8 / wt2;
        {
            CHECK(containsSubExpression(expr9, wt1));
            CHECK(containsSubExpression(expr10, wt2));

            CHECK_FALSE(containsSubExpression(expr9, wt2));
            CHECK_FALSE(containsSubExpression(expr10, wt1));
        }
    }

}
