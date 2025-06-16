
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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "ExpressionMatchers.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <common/TestValues.hpp>

#include <catch2/catch_test_macros.hpp>

namespace FastDivisionTest
{

    template <typename T>
    auto getMagicMultiple(T x, rocRoller::ContextPtr ctx = nullptr)
    {
        using namespace rocRoller;
        namespace Ex = Expression;

        Ex::ExpressionPtr expr;

        if constexpr(std::same_as<T, Ex::ExpressionPtr>)
        {
            expr = x;
        }
        else
        {
            expr = Ex::literal(x);
        }

        expr = magicMultiple(expr);

        if(ctx)
        {
            return ctx->kernel()->findArgumentForExpression(expr);
        }
        else
        {
            return std::make_shared<Ex::Expression>(evaluate(expr));
        }
    }

    template <typename T>
    auto getMagicShifts(T x, rocRoller::ContextPtr ctx = nullptr)
    {
        using namespace rocRoller;
        namespace Ex = Expression;

        Ex::ExpressionPtr expr;

        if constexpr(std::same_as<T, Ex::ExpressionPtr>)
        {
            expr = x;
        }
        else
        {
            expr = Ex::literal(x);
        }

        expr = magicShifts(expr);

        if(ctx)
        {
            return ctx->kernel()->findArgumentForExpression(expr);
        }
        else
        {
            return std::make_shared<Ex::Expression>(evaluate(expr));
        }
    }

    template <typename T>
    auto getMagicSign(T x, rocRoller::ContextPtr ctx = nullptr)
    {
        using namespace rocRoller;
        namespace Ex = Expression;

        Ex::ExpressionPtr expr;

        if constexpr(std::same_as<T, Ex::ExpressionPtr>)
        {
            expr = x;
        }
        else
        {
            expr = Ex::literal(x);
        }

        expr = magicSign(expr);

        if(ctx)
        {
            return ctx->kernel()->findArgumentForExpression(expr);
        }
        else
        {
            return std::make_shared<Ex::Expression>(evaluate(expr));
        }
    }

    TEST_CASE("FastDivision ExpressionTransformation works for constant expressions.",
              "[division][expression][expression-transformation]")
    {
        using namespace rocRoller;
        namespace Ex = Expression;

        auto context = TestContext::ForDefaultTarget({.minLaunchTimeExpressionComplexity = 49});

        auto command = std::make_shared<rocRoller::Command>();

        auto aTag = command->allocateTag();
        auto a    = command
                     ->allocateArgument(
                         {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value)
                     ->expression();

        auto b = Register::Value::Placeholder(
                     context.get(), Register::Type::Vector, DataType::Int32, 1)
                     ->expression();

        auto argsBefore = context->kernel()->arguments().size();

        auto expr      = a / Ex::literal(8u);
        auto expr_fast = fastDivision(expr, context.get());
        CHECK_THAT(expr_fast, IdenticalTo(a >> Ex::literal(3u)));

        expr      = a / Ex::literal(8);
        expr_fast = fastDivision(expr, context.get());
        CHECK_THAT(expr_fast,
                   IdenticalTo((a + Ex::logicalShiftR((a >> Ex::literal(31u)), Ex::literal(29u))
                                >> Ex::literal(3))));

        expr      = b / Ex::literal(7u);
        expr_fast = fastDivision(expr, context.get());

        {
            auto mulHigh = multiplyHigh(b, getMagicMultiple(7u));
            CHECK_THAT(
                expr_fast,
                EquivalentTo((((b - mulHigh) >> Ex::literal(1u)) + mulHigh) >> Ex::literal(2)));
        }

        expr      = b / Ex::literal(1);
        expr_fast = fastDivision(expr, context.get());
        CHECK_THAT(expr_fast, EquivalentTo(b));

        expr      = b / Ex::literal(-5);
        expr_fast = fastDivision(expr, context.get());

        {
            auto mulPlusB = multiplyHigh(b, getMagicMultiple(-5)) + b;
            CHECK_THAT(expr_fast,
                       EquivalentTo((((mulPlusB + ((mulPlusB >> Ex::literal(31)) & Ex::literal(4)))
                                      >> Ex::literal(2))
                                     ^ Ex::literal(-1))
                                    + Ex::literal(1)));
        }

        expr      = a / Ex::literal(8u);
        expr_fast = fastDivision(expr, context.get());
        CHECK_THAT(expr_fast, EquivalentTo(a >> Ex::literal(3u)));

        expr      = a / Ex::literal(128u);
        expr_fast = fastDivision(expr, context.get());
        CHECK_THAT(expr_fast, EquivalentTo(a >> Ex::literal(7u)));

        expr      = a / (Ex::literal(43u) + Ex::literal(85u));
        expr_fast = fastDivision(expr, context.get());
        CHECK_THAT(expr_fast, EquivalentTo(a >> Ex::literal(7u)));

        auto argsAfter = context->kernel()->arguments().size();
        CHECK(argsBefore == argsAfter);
    }

    TEST_CASE("FastDivision ExpressionTransformation works for argument expressions.",
              "[division][expression][expression-transformation]")
    {
        using namespace rocRoller;
        namespace Ex = Expression;

        auto context = TestContext::ForDefaultTarget({.minLaunchTimeExpressionComplexity = 49});

        auto command = std::make_shared<rocRoller::Command>();

        auto reg = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        auto a = reg->expression();

        auto reg2 = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::UInt32, 1);
        auto a_unsigned = reg2->expression();

        auto bSignedTag = command->allocateTag();
        auto b_signed   = command
                            ->allocateArgument({DataType::Int32, PointerType::Value},
                                               bSignedTag,
                                               ArgumentType::Value)
                            ->expression();

        auto bUnsignedTag = command->allocateTag();
        auto b_unsigned   = command
                              ->allocateArgument({DataType::UInt32, PointerType::Value},
                                                 bUnsignedTag,
                                                 ArgumentType::Value)
                              ->expression();

        auto expr      = a / b_signed;
        auto expr_fast = fastDivision(expr, context.get());

        {
            auto mul   = getMagicMultiple(b_signed, context.get());
            auto shift = getMagicShifts(b_signed, context.get());
            auto sign  = getMagicSign(b_signed, context.get());

            auto mulPlusA = a + multiplyHigh(a, mul);

            CHECK_THAT(expr_fast,
                       EquivalentTo(
                           ((((mulPlusA
                               + ((mulPlusA >> Ex::literal(31))
                                  & ((Ex::literal(1) << shift)
                                     + Ex::conditional(
                                         mul == Ex::literal(0), Ex::literal(-1), Ex::literal(0)))))
                              >> shift)
                             ^ sign)
                            - sign)));
        }

        expr      = a_unsigned / b_unsigned;
        expr_fast = fastDivision(expr, context.get());

        {
            auto mul   = getMagicMultiple(b_unsigned, context.get());
            auto shift = getMagicShifts(b_unsigned, context.get());

            auto mulHigh = multiplyHigh(a_unsigned, mul);

            CHECK_THAT(
                expr_fast,
                EquivalentTo((((a_unsigned - mulHigh) >> Ex::literal(1u)) + mulHigh) >> shift));
        }
    }

    TEST_CASE("FastDivision ExpressionTransformation works for constant modulo expressions.",
              "[division][expression][expression-transformation]")
    {
        using namespace rocRoller;
        namespace Ex = Expression;

        auto context = TestContext::ForDefaultTarget({.minLaunchTimeExpressionComplexity = 49});

        auto command = std::make_shared<Command>();

        auto regPtr = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::UInt32, 1);
        auto reg = regPtr->expression();

        auto aTag = command->allocateTag();
        auto a    = command
                     ->allocateArgument(
                         {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value)
                     ->expression();

        a = context->kernel()->addArgument(
            {.name = "arg", .variableType = DataType::Int32, .expression = a});

        auto argsBefore = context->kernel()->arguments();

        auto expr      = a % Ex::literal(8u);
        auto expr_fast = fastDivision(expr, context.get());
        CHECK_THAT(expr_fast, EquivalentTo(a & Ex::literal(7u)));

        expr      = a % Ex::literal(8);
        expr_fast = fastDivision(expr, context.get());
        CHECK_THAT(expr_fast,
                   EquivalentTo(a
                                - ((a + logicalShiftR((a >> Ex::literal(31u)), Ex::literal(29u)))
                                   & Ex::literal(-8))));

        expr      = a % Ex::literal(7u);
        expr_fast = fastDivision(expr, context.get());
        CHECK_THAT(
            expr_fast,
            EquivalentTo(a - (fastDivision(a / Ex::literal(7u), context.get()) * Ex::literal(7u))));

        expr      = a % Ex::literal(1);
        expr_fast = fastDivision(expr, context.get());
        CHECK_THAT(expr_fast, EquivalentTo(Ex::literal(0)));

        expr      = a % Ex::literal(1u);
        expr_fast = fastDivision(expr, context.get());
        CHECK_THAT(expr_fast, EquivalentTo(Ex::literal(0u)));

        expr      = a % Ex::literal(-5);
        expr_fast = fastDivision(expr, context.get());
        CHECK_THAT(
            expr_fast,
            EquivalentTo(a - (fastDivision(a / Ex::literal(-5), context.get()) * Ex::literal(-5))));

        expr      = a % Ex::literal(8u);
        expr_fast = fastDivision(expr, context.get());
        CHECK_THAT(expr_fast, EquivalentTo(a & Ex::literal(7u)));

        expr      = a % Ex::literal(128u);
        expr_fast = fastDivision(expr, context.get());
        CHECK_THAT(expr_fast, EquivalentTo(a & Ex::literal(127u)));

        auto argsAfter = context->kernel()->arguments();
        CHECK(argsBefore == argsAfter);
    }

    TEST_CASE("FastDivision ExpressionTransformation works for modulo of argument expressions.",
              "[division][expression][expression-transformation]")
    {
        using namespace rocRoller;
        namespace Ex = Expression;

        auto context = TestContext::ForDefaultTarget({.minLaunchTimeExpressionComplexity = 49});

        auto command = std::make_shared<Command>();

        auto reg = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        auto a = reg->expression();

        auto reg2 = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::UInt32, 1);
        auto a_unsigned = reg2->expression();

        auto bSignedTag = command->allocateTag();
        auto b_signed   = command
                            ->allocateArgument({DataType::Int32, PointerType::Value},
                                               bSignedTag,
                                               ArgumentType::Value)
                            ->expression();

        auto bUnsignedTag = command->allocateTag();
        auto b_unsigned   = command
                              ->allocateArgument({DataType::UInt32, PointerType::Value},
                                                 bUnsignedTag,
                                                 ArgumentType::Value)
                              ->expression();

        auto expr      = a % b_signed;
        auto expr_fast = fastDivision(expr, context.get());
        auto fastDiv   = fastDivision(a / b_signed, context.get());
        CHECK_THAT(expr_fast, EquivalentTo(a - (fastDiv * b_signed)));

        expr      = a_unsigned % b_unsigned;
        expr_fast = fastDivision(expr, context.get());
        fastDiv   = fastDivision(a_unsigned / b_unsigned, context.get());
        CHECK_THAT(expr_fast, EquivalentTo(a_unsigned - (fastDiv * b_unsigned)));
    }
}