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
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "ExpressionMatchers.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("FastArithmetic ExpressionTransformation works",
          "[expression][expression-transformation]")
{
    using namespace rocRoller;
    using Expression::literal;

    auto context = TestContext::ForDefaultTarget();

    auto r
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Int32, 1);
    r->allocateNow();
    auto v = r->expression();
    auto r2
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Int32, 1);
    r2->allocateNow();
    auto v2 = r2->expression();

    auto zero  = literal(0);
    auto one   = literal(1);
    auto a     = literal(33);
    auto b     = literal(100);
    auto c     = literal(12.f);
    auto True  = literal(true);
    auto False = literal(false);

    auto fast = Expression::FastArithmetic(context.get());

    SECTION("Negate")
    {
        CHECK_THAT(fast(-(one + one)), IdenticalTo(literal(-2)));
    }

    SECTION("Multiply")
    {
        CHECK_THAT(fast(zero * one), IdenticalTo(zero));

        CHECK_THROWS_AS(fast(c * zero), FatalError);
        CHECK_THAT(fast(c * literal(0.f)), IdenticalTo(literal(0.f)));

        CHECK_THROWS_AS(fast(c * one), FatalError);
        CHECK_THAT(fast(c * convert<DataType::Float>(one)), IdenticalTo(c));

        CHECK_THAT(fast(v * zero), IdenticalTo(zero));

        CHECK_THAT(fast(a * b), IdenticalTo(literal(3300)));

        CHECK_THAT(fast((v * one) * a), IdenticalTo(v * literal(33)));
        CHECK_THAT(fast((v * a) * a), IdenticalTo(v * literal(33 * 33)));
        CHECK_THAT(fast(v * (a * a)), IdenticalTo(v * literal(33 * 33)));
        CHECK_THAT(fast((a * v) * a), EquivalentTo(v * literal(33 * 33)));
    }

    SECTION("Add")
    {
        CHECK_THAT(fast(zero + one), IdenticalTo(one));
        CHECK_THAT(fast(v + zero), IdenticalTo(v));
        CHECK_THAT(fast(a + b), IdenticalTo(literal(133)));

        CHECK_THAT(fast((v + one) + a), IdenticalTo(v + literal(34)));
        CHECK_THAT(fast(v + (one + a)), IdenticalTo(v + literal(34)));
        CHECK_THAT(fast((one + v) + a), EquivalentTo(v + literal(34)));
    }

    SECTION("Divide")
    {
        CHECK_THAT(fast(a / one), IdenticalTo(a));
        CHECK_THAT(fast(v / one), IdenticalTo(v));
        CHECK_THAT(fast(v / v2), IdenticalTo(v / v2));
        CHECK_THAT(fast(b / v), IdenticalTo(b / v));
    }

    SECTION("Modulo")
    {
        CHECK_THAT(fast(a % one), IdenticalTo(zero));
        CHECK_THAT(fast(v % one), IdenticalTo(zero));
        CHECK_THAT(fast(v % v2), IdenticalTo(v % v2));
        CHECK_THAT(fast(b % v), IdenticalTo(b % v));
    }

    SECTION("Bitwise And")
    {
        CHECK_THAT(fast(((((v & one) & a) & a) & a) & a), EquivalentTo(v & one));
        CHECK_THAT(fast(((((v & one) & a) + a) & a) & one), IdenticalTo(((v & one) + a) & one));
        CHECK_THAT(fast((one & v) & a), EquivalentTo(v & one));
        CHECK_THAT(fast((v & one) & a), EquivalentTo(v & one));
        CHECK_THAT(fast(a & (one & v)), EquivalentTo(v & one));
        CHECK_THAT(fast(a & (v & one)), EquivalentTo(v & one));
        CHECK_THAT(fast(fast((one & a) & v)), EquivalentTo(v & one));
        CHECK_THAT(fast(one & a), IdenticalTo(one));
        CHECK_THAT(fast(one & b), IdenticalTo(zero));
        CHECK_THAT(fast(v & (a + b << one)), IdenticalTo(v & literal(266)));
        CHECK_THAT(fast(v & (zero + zero)), IdenticalTo(zero));
        CHECK_THAT(fast(v & -one), IdenticalTo(v & literal(-1)));
        CHECK_THAT(fast(v & one), EquivalentTo(v & one));
        CHECK_THAT(fast(v & v2), IdenticalTo(v & v2));
        CHECK_THAT(fast(v & zero), IdenticalTo(zero));
        CHECK_THAT(fast(v2 & (zero + zero)), IdenticalTo(zero));
    }

    SECTION("Logical And")
    {
        CHECK_THAT(fast((v < one) && False), IdenticalTo(convert<DataType::Bool64>(False)));
        CHECK_THAT(fast((v < one) && True), IdenticalTo(v < one));
    }

    SECTION("Logical Or")
    {
        CHECK_THAT(fast((v < one) || True), IdenticalTo(convert<DataType::Bool64>(True)));
        CHECK_THAT(fast((v < one) || False), IdenticalTo(v < one));
    }

    SECTION("ShiftL")
    {
        auto codegenOnly = Expression::EvaluationTimes{Expression::EvaluationTime::KernelExecute};

        CHECK_THAT(fast(v << zero), IdenticalTo(v));
        CHECK_THAT(fast((one << one) << one), IdenticalTo(literal(4)));
        CHECK_THAT(fast(v << (zero << zero)), IdenticalTo(v));
        CHECK_THAT(fast(v << (one << one)), IdenticalTo(v << literal(2)));
        CHECK_THAT(fast((v << one) << one), IdenticalTo(v << literal(2)));
        CHECK_THAT(fast(logicalShiftR((v << one), one)), IdenticalTo(v & literal<int>((~0u) >> 1)));

        CHECK_THAT(fast((v << one) << one), IdenticalTo(v << literal(2)));
        CHECK(evaluationTimes(((a << v) << one) << one) == codegenOnly);
        CHECK(evaluationTimes((a << v) << one) == codegenOnly);
        CHECK(evaluationTimes(a << v) == codegenOnly);
        CHECK_THAT(fast(((a << v) << one) << one), IdenticalTo((a << v) << literal(2)));
    }

    SECTION("Arithmetic ShiftR")
    {
        CHECK_THAT(fast(((one >> v) >> one) >> one), IdenticalTo((one >> v) >> literal(2)));
        CHECK_THAT(fast(((v >> one) >> one) >> one), IdenticalTo(v >> literal(3)));
        CHECK_THAT(fast((a >> one) >> one), IdenticalTo(literal(8)));
        CHECK_THAT(fast((v >> one) >> one), IdenticalTo(v >> literal(2)));
        CHECK_THAT(fast(literal(-2) >> one), IdenticalTo(literal(-1)));
        CHECK_THAT(fast(v >> (zero >> zero)), IdenticalTo(v));
        CHECK_THAT(fast(v >> zero), IdenticalTo(v));
    }

    SECTION("LogicalShiftR")
    {
        CHECK_THAT(fast(logicalShiftR(v, zero)), IdenticalTo(v));
        CHECK_THAT(fast(logicalShiftR(logicalShiftR(a, one), one)), IdenticalTo(literal(8)));
        CHECK_THAT(fast(logicalShiftR(literal(-2), one)), IdenticalTo(literal(-1 ^ (1 << 31))));
        CHECK_THAT(fast(logicalShiftR(v, logicalShiftR(zero, zero))), IdenticalTo(v));

        CHECK_THAT(fast(logicalShiftR(logicalShiftR(v, one), one)),
                   IdenticalTo(logicalShiftR(v, literal(2))));

        CHECK_THAT(fast(logicalShiftR(logicalShiftR(logicalShiftR(v, one), one), one)),
                   IdenticalTo(logicalShiftR(v, literal(3))));

        CHECK_THAT(fast(logicalShiftR(logicalShiftR(logicalShiftR(one, v), one), one)),
                   IdenticalTo(logicalShiftR(logicalShiftR(one, v), literal(2))));
    }

    SECTION("nullptr")
    {
        Expression::ExpressionPtr nullexpr;

        CHECK_THAT(fast(nullptr), IdenticalTo(nullptr));

        CHECK_THROWS_AS(
            fast(std::make_shared<Expression::Expression>(Expression::Multiply{nullptr, nullptr})),
            FatalError);
        CHECK_THROWS_AS(
            fast(std::make_shared<Expression::Expression>(Expression::Multiply{nullptr, zero})),
            FatalError);
        CHECK_THROWS_AS(
            fast(std::make_shared<Expression::Expression>(Expression::Multiply{zero, nullptr})),
            FatalError);
        CHECK_THROWS_AS(fast(nullptr * nullexpr), FatalError);
        CHECK_THROWS_AS(fast(nullptr * zero), FatalError);
        CHECK_THROWS_AS(fast(zero * nullptr), FatalError);
    }

    SECTION("AddShiftL")
    {
        // addShiftLeft
        CHECK_THAT(fast(a + b << one + one), IdenticalTo(literal(532)));

        CHECK_THAT(fast((b + a) << one), IdenticalTo(literal(266)));
        CHECK_THAT(fast((v + a) << one), IdenticalTo(addShiftL(v, a, one)));
        CHECK_THAT(fast((a + v) << one), IdenticalTo(addShiftL(a, v, one)));
        CHECK_THAT(fast((v + v2) << one), IdenticalTo(addShiftL(v, v2, one)));
        CHECK_THAT(fast(((v + a) << one) + v), IdenticalTo(addShiftL(v, a, one) + v));
        CHECK_THAT(fast(((b + v + a) << one) + v),
                   IdenticalTo(addShiftL(literal(133), v, one) + v));

        CHECK_THAT(fast((((v + v2) << one) + v) + v2),
                   IdenticalTo((addShiftL(v, v2, one) + v) + v2));

        CHECK_THAT(fast((b + a) << v), IdenticalTo(literal(133) << v));
    }

    SECTION("Bitwise Or")
    {
        CHECK_THAT(fast(v | (a + b << one)), IdenticalTo(v | literal(266)));
        CHECK_THAT(fast(v | -one), IdenticalTo(v | literal(-1)));
        CHECK_THAT(fast(v | one), EquivalentTo(v | one));
        CHECK_THAT(fast((v | one) | a), EquivalentTo(v | a));
        CHECK_THAT(fast(fast((one | a) | v)), EquivalentTo(v | a));
        CHECK_THAT(fast(a | (v | one)), EquivalentTo(v | a));
        CHECK_THAT(fast((one | v) | a), EquivalentTo(v | a));
        CHECK_THAT(fast(a | (one | v)), EquivalentTo(v | a));
        CHECK_THAT(fast(((((v | one) | a) | a) | a) | a), EquivalentTo(v | a));
        CHECK_THAT(fast(((((v | one) | a) + a) | a) | one), IdenticalTo(((v | a) + a) | a));
    }

    SECTION("Subtraction")
    {
        // fast does not affect non-associative ops
        CHECK_THAT(fast((v - one) - a), IdenticalTo((v - one) - a));
    }

    SECTION("Non-transformed")
    {
        CHECK_THAT(fast(-one + one), IdenticalTo(literal(0)));
        CHECK_THAT(fast(one + one), IdenticalTo(literal(2)));
    }

    SECTION("ShiftLAdd")
    {
        CHECK_THAT(fast((b << one) + a), IdenticalTo(literal(233)));
        CHECK_THAT(fast((v << one) + a), IdenticalTo(shiftLAdd(v, one, a)));
        CHECK_THAT(fast((v << one) + v), IdenticalTo(shiftLAdd(v, one, v)));

        CHECK_THAT(fast((((v * v2) << one) + v) + v2),
                   IdenticalTo(shiftLAdd((v * v2), one, v) + v2));

        CHECK_THAT(fast((b << v) + a), IdenticalTo((b << v) + a));
    }

    SECTION("FMA")
    {
        CHECK_THAT(fast((b * one) + a), IdenticalTo(literal(133)));
        CHECK_THAT(fast((v * v2) + a), IdenticalTo(multiplyAdd(v, v2, a)));
        CHECK_THAT(fast((v * one * v2) + v), IdenticalTo(multiplyAdd(v, v2, v)));

        CHECK_THAT(fast((((v + v2) * one) + v) + v2), IdenticalTo(v + v2 + v + v2));

        CHECK_THAT(fast((((v * v2) * one) + v) + v2), IdenticalTo(multiplyAdd(v, v2, v) + v2));

        CHECK_THAT(fast((((v * v2) * one) + v) + v2), IdenticalTo(multiplyAdd(v, v2, v) + v2));
    }
}

TEST_CASE("fastMultiplication and fastDivision lead into combineShifts",
          "[expression][expression-transformation]")
{
    using namespace rocRoller;
    using Expression::literal;
    auto ctx = TestContext::ForDefaultTarget();

    Expression::FastArithmetic fast(ctx.get());
    auto                       command = std::make_shared<Command>();
    auto                       argTag  = command->allocateTag();
    auto arg = command->allocateArgument(DataType::UInt32, argTag, ArgumentType::Limit);

    auto argExp = fast(arg->expression());

    SECTION("Multiply then divide masks off MSBs.")
    {
        auto exp     = (argExp * literal(4u)) / literal(4u);
        auto maskExp = argExp & literal(0b00111111111111111111111111111111u);
        CHECK_THAT(fast(exp), IdenticalTo(maskExp));
    }

    SECTION("Divide then multiply masks off LSBs.")
    {
        auto exp     = (argExp / literal(4u)) * literal(4u);
        auto maskExp = argExp & literal(0b11111111111111111111111111111100u);
        CHECK_THAT(fast(exp), IdenticalTo(maskExp));
    }

    SECTION("Repeated multiplication is combined.")
    {
        auto exp = argExp * literal(4u) * literal(8u);
        CHECK_THAT(fast(exp), IdenticalTo(argExp << literal(5u)));
    }

    SECTION("Multiplication is combined with shift.")
    {
        auto exp = (argExp * literal(4u)) << literal(2u);
        CHECK_THAT(fast(exp), IdenticalTo(argExp << literal(4u)));
    }

    SECTION("Shift is combined with multiplication.")
    {
        auto exp = (argExp << literal(4u)) * literal(4u);
        CHECK_THAT(fast(exp), IdenticalTo(argExp << literal(6u)));
    }

    SECTION("Repeated division is combined.")
    {
        auto exp = argExp / literal(2u) / literal(16u);
        CHECK_THAT(fast(exp), IdenticalTo(argExp >> literal(5u)));
    }

    SECTION("Division is combined with shift.")
    {
        auto exp = (argExp / literal(8u)) >> literal(2u);
        CHECK_THAT(fast(exp), IdenticalTo(argExp >> literal(5u)));
    }

    SECTION("Shift is combined with division.")
    {
        auto exp = (argExp >> literal(8u)) / literal(2u);
        CHECK_THAT(fast(exp), IdenticalTo(argExp >> literal(9u)));
    }
}
