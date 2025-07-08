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

#include "GenericContextFixture.hpp"

#include <rocRoller/CommonSubexpressionElim.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/Operations/Command.hpp>

using namespace rocRoller;

namespace ExpressionTest
{
    struct CommonSubexpressionElimTest : public GenericContextFixture
    {
        std::string toString(Expression::ExpressionTree tree)
        {
            std::stringstream ss;
            int               i = 0;
            for(auto result : tree)
            {
                if(result.reg)
                    ss << i++ << ShowValue(result.reg->toString()) << ShowValue(result.expr);
                else
                    ss << ShowValue(result.expr);
                for(auto dep : result.deps)
                {
                    ss << dep << ",";
                }
                ss << std::endl;
                ss << "Consolidated " << result.consolidationCount << std::endl << std::endl;
            }
            ss << "********************" << std::endl;

            return ss.str();
        }
    };

    TEST_F(CommonSubexpressionElimTest, NoEffect)
    {
        auto one = Expression::literal(1);

        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        {
            auto results = consolidateSubExpressions(a, m_context);
            EXPECT_EQ(results.size(), 1);
            EXPECT_TRUE(identical(results.at(0).expr, a));
            EXPECT_EQ(getConsolidationCount(results), 0);

            EXPECT_TRUE(identical(a, rebuildExpression(results)));
        }

        {
            auto results = consolidateSubExpressions(one, m_context);
            EXPECT_EQ(results.size(), 1);
            EXPECT_TRUE(identical(results.at(0).expr, one));
            EXPECT_EQ(getConsolidationCount(results), 0);

            EXPECT_TRUE(identical(one, rebuildExpression(results)));
        }

        {
            auto results = consolidateSubExpressions(a + b, m_context);
            EXPECT_EQ(results.size(), 3);
            EXPECT_TRUE(identical(results.back().expr, a + b));
            EXPECT_EQ(getConsolidationCount(results), 0);

            EXPECT_TRUE(identical(a + b, rebuildExpression(results)));
        }

        {
            auto results = consolidateSubExpressions(a + one, m_context);
            EXPECT_EQ(results.size(), 3);
            EXPECT_TRUE(identical(results.back().expr, a + one));
            EXPECT_EQ(getConsolidationCount(results), 0);

            EXPECT_TRUE(identical(a + one, rebuildExpression(results)));
        }

        {
            auto results = consolidateSubExpressions(a + a, m_context);
            EXPECT_EQ(results.size(), 1 + 1);
            EXPECT_TRUE(identical(results.back().expr, a + a));
            EXPECT_EQ(getConsolidationCount(results), 0);

            EXPECT_TRUE(identical(a + a, rebuildExpression(results)));
        }

        auto rc = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rc->allocateNow();

        auto rd = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rd->allocateNow();

        auto c = rc->expression();
        auto d = rd->expression();

        {
            auto results = consolidateSubExpressions(a + (b * c), m_context);

            EXPECT_EQ(results.size(), 3 + 2);
            EXPECT_EQ(getConsolidationCount(results), 0);

            EXPECT_TRUE(identical(a + (b * c), rebuildExpression(results)))
                << ShowValue(rebuildExpression(results));
        }
    }

    TEST_F(CommonSubexpressionElimTest, Precedence01)
    {

        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Half, 1);
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Half, 1);
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        {
            auto expr = convert<DataType::Half>(
                convert<DataType::Float>(a)
                + (convert<DataType::Float>(a) * convert<DataType::Float>(b)));

            auto results = consolidateSubExpressions(expr, m_context);

            EXPECT_EQ(results.size(), 2 + 5);
            EXPECT_TRUE(
                identical(results.at(4).expr,
                          results.at(1).reg->expression() * results.at(3).reg->expression()));
            EXPECT_FALSE(
                identical(results.at(4).expr,
                          results.at(1).reg->expression() * results.at(1).reg->expression()));
            EXPECT_EQ(getConsolidationCount(results), 1);

            EXPECT_TRUE(identical(expr, rebuildExpression(results)));
        }
    }

    TEST_F(CommonSubexpressionElimTest, Precedence02)
    {
        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::UInt32, 1);
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();
        {
            auto expr = (a + b) << b;

            auto results = consolidateSubExpressions(expr, m_context);
            EXPECT_EQ(results.size(), 2 + 2);

            EXPECT_TRUE(
                identical(results.at(3).expr,
                          results.at(2).reg->expression() << results.at(1).reg->expression()));
            EXPECT_EQ(getConsolidationCount(results), 0);

            EXPECT_TRUE(identical(expr, rebuildExpression(results)));
        }
    }

    TEST_F(CommonSubexpressionElimTest, Simple)
    {
        auto one = Expression::literal(1);

        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        {
            auto results = consolidateSubExpressions((a + b) * (a + b), m_context);

            EXPECT_EQ(results.size(), 2 + 2);
            EXPECT_TRUE(identical(results.at(2).expr, a + b));
            EXPECT_TRUE(
                identical(results.at(3).expr,
                          results.at(2).reg->expression() * results.at(2).reg->expression()));
            EXPECT_EQ(getConsolidationCount(results), 1) << toString(results);

            EXPECT_TRUE(identical((a + b) * (a + b), rebuildExpression(results)));
        }

        {
            auto expr1 = a + b;
            auto expr2 = b * expr1;

            auto results = consolidateSubExpressions(expr2, m_context);

            EXPECT_EQ(results.size(), 2 + 2);
            EXPECT_EQ(getConsolidationCount(results), 0) << toString(results);

            EXPECT_TRUE(identical(expr2, rebuildExpression(results)));
        }
    }

    TEST_F(CommonSubexpressionElimTest, XL)
    {
        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        {
            auto expr    = ((a + b) * (a + b)) - (Expression::literal(5) / ((a + b) * (a + b)));
            auto results = consolidateSubExpressions(expr, m_context);
            EXPECT_EQ(results.size(), 3 + 4);
            EXPECT_EQ(results.at(2).deps, (std::set<int>{0, 1}));
            EXPECT_EQ(results.at(3).deps, (std::set<int>{2}));
            EXPECT_EQ(results.at(5).deps, (std::set<int>{3, 4})) << toString(results);
            EXPECT_EQ(getConsolidationCount(results), 4) << toString(results);

            EXPECT_TRUE(identical(expr, rebuildExpression(results)));
        }

        {
            auto expr    = a - ((a + b) * b + (a + b));
            auto results = consolidateSubExpressions(expr, m_context);
            EXPECT_EQ(results.size(), 2 + 4);
            EXPECT_EQ(results.at(2).deps, (std::set<int>{0, 1}));
            EXPECT_EQ(results.at(3).deps, (std::set<int>{1, 2}));
            EXPECT_EQ(results.at(4).deps, (std::set<int>{2, 3}));
            EXPECT_EQ(results.at(5).deps, (std::set<int>{0, 4}));
            EXPECT_EQ(getConsolidationCount(results), 1) << toString(results);

            EXPECT_TRUE(identical(expr, rebuildExpression(results)));
        }
    }

    TEST_F(CommonSubexpressionElimTest, FastDiv)
    {
        auto command = std::make_shared<Command>();

        auto reg
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);
        auto a = reg->expression();

        auto bTag = command->allocateTag();
        auto b    = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bTag, ArgumentType::Value));

        {
            auto expr    = Expression::fastDivision(a / b, m_context);
            auto results = consolidateSubExpressions(expr, m_context);
            EXPECT_GT(results.size(), 3) << rocRoller::Expression::toString(a / b) << "\n"
                                         << rocRoller::Expression::toString(expr);
            EXPECT_GE(getConsolidationCount(results), 1) << toString(results);

            EXPECT_TRUE(identical(expr, rebuildExpression(results)));
        }
    }

    TEST_F(CommonSubexpressionElimTest, LiterallyDifferent)
    {
        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->allocateNow();

        auto a = ra->expression();

        auto expr = ((a + Expression::literal(1l)) * (a + Expression::literal(2ul)));

        {
            auto results = consolidateSubExpressions(expr, m_context);
            EXPECT_EQ(results.size(), 3 + 3);
            EXPECT_EQ(getConsolidationCount(results), 0);

            EXPECT_TRUE(identical(expr, rebuildExpression(results)));
        }
    }

    TEST_F(CommonSubexpressionElimTest, MultipleSCC)
    {
        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        {
            auto expr    = (a <= b) && (b <= a);
            auto results = consolidateSubExpressions(expr, m_context);
            EXPECT_EQ(results.size(), 2 + 3);
            EXPECT_TRUE(
                !identical(results[2].reg->expression(), m_context->getSCC()->expression())
                || !identical(results[3].reg->expression(), m_context->getSCC()->expression()));
            EXPECT_EQ(getConsolidationCount(results), 0);

            EXPECT_TRUE(identical(expr, rebuildExpression(results)));
        }
    }
}
