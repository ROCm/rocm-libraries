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

#include <common/TestValues.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Simplify ExpressionTransformation works", "[expression][expression-transformation]")
{
    using namespace rocRoller;
    using namespace Expression;

    auto context = TestContext::ForDefaultTarget();

    auto r
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Int32, 1);
    r->allocateNow();
    auto v = r->expression();
    auto r2
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Int32, 1);
    r2->allocateNow();
    auto v2 = r2->expression();
    auto r3
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::UInt64, 1);
    r3->allocateNow();
    auto v3 = r3->expression();

    auto zero  = literal(0);
    auto one   = literal(1);
    auto a     = literal(33);
    auto b     = literal(100);
    auto c     = literal(12.f);
    auto True  = literal(true);
    auto False = literal(false);

    // negate
    SECTION("Negate")
    {
        CHECK_THAT(simplify(-(one + one)), IdenticalTo(literal(-2)));
    }

    SECTION("Multiply")
    {
        CHECK_THAT(simplify(zero * one), IdenticalTo(zero));

        CHECK_THROWS_AS(simplify(c * zero), FatalError);
        CHECK_THAT(simplify(c * literal(0.f)), IdenticalTo(literal(0.f)));

        CHECK_THROWS_AS(simplify(c * one), FatalError);
        CHECK_THAT(simplify(c * convert<DataType::Float>(one)), IdenticalTo(c));

        CHECK_THAT(simplify(v * zero), IdenticalTo(zero));

        CHECK_THAT(simplify(a * b), IdenticalTo(literal(3300)));
    }

    SECTION("Add")
    {
        CHECK_THAT(simplify(zero + one), IdenticalTo(one));
        CHECK_THAT(simplify(v + zero), IdenticalTo(v));
        CHECK_THAT(simplify(a + b), IdenticalTo(literal(133)));
    }

    SECTION("Divide")
    {
        CHECK_THAT(simplify(a / one), IdenticalTo(a));
        CHECK_THAT(simplify(v / one), IdenticalTo(v));
        CHECK_THAT(simplify(v / a), IdenticalTo(v / a));
        CHECK_THAT(simplify(b / v), IdenticalTo(b / v));
    }

    SECTION("Modulo")
    {
        CHECK_THAT(simplify(a % one), IdenticalTo(zero));
        CHECK_THAT(simplify(v % one), IdenticalTo(zero));
        CHECK_THAT(simplify(v % a), IdenticalTo(v % a));
        CHECK_THAT(simplify(b % v), IdenticalTo(b % v));
    }

    SECTION("Bitwise And")
    {
        CHECK_THAT(simplify(one & b), IdenticalTo(zero));
        CHECK_THAT(simplify(one & a), IdenticalTo(one));
        CHECK_THAT(simplify(v & zero), IdenticalTo(zero));
        CHECK_THAT(simplify(v & (zero + zero)), IdenticalTo(zero));
        CHECK_THAT(simplify(v2 & (zero + zero)), IdenticalTo(zero));
        CHECK_THAT(simplify(v & v2), IdenticalTo(v & v2));
    }

    SECTION("Logical And")
    {
        CHECK_THAT(simplify((v < one) && False), IdenticalTo(convert<DataType::Bool64>(False)));
        CHECK_THAT(simplify((v < one) && True), IdenticalTo(v < one));
    }

    SECTION("Logical Or")
    {
        CHECK_THAT(simplify((v < one) || True), IdenticalTo(convert<DataType::Bool64>(True)));
        CHECK_THAT(simplify((v < one) || False), IdenticalTo(v < one));
    }

    auto fast = FastArithmetic(context.get());
    SECTION("ShiftL")
    {
        CHECK_THAT(simplify(v << zero), IdenticalTo(v));
        CHECK_THAT(simplify((one << one) << one), IdenticalTo(literal(4)));
        CHECK_THAT(simplify(v << (zero << zero)), IdenticalTo(v));
        CHECK_THAT(simplify(v << (one << one)), IdenticalTo(v << literal(2)));
        CHECK_THAT(fast((v << one) << one), IdenticalTo(v << literal(2)));
        CHECK_THAT(fast(logicalShiftR((v << one), one)), IdenticalTo(v & literal<int>((~0u) >> 1)));
    }

    SECTION("SignedShiftR")
    {
        CHECK_THAT(simplify(v >> zero), IdenticalTo(v));
        CHECK_THAT(simplify((a >> one) >> one), IdenticalTo(literal(8)));
        CHECK_THAT(simplify(v >> (zero >> zero)), IdenticalTo(v));
        CHECK_THAT(simplify(literal(-2) >> one), IdenticalTo(literal(-1)));
        CHECK_THAT(fast((v >> one) >> one), IdenticalTo(v >> literal(2)));
    }

    SECTION("LogicalShiftR")
    {
        CHECK_THAT(simplify(logicalShiftR(v, zero)), IdenticalTo(v));
        CHECK_THAT(simplify(logicalShiftR(logicalShiftR(a, one), one)), IdenticalTo(literal(8)));
        CHECK_THAT(simplify(logicalShiftR(literal(-2), one)), IdenticalTo(literal(-1 ^ (1 << 31))));
        CHECK_THAT(simplify(logicalShiftR(v, logicalShiftR(zero, zero))), IdenticalTo(v));
    }

    SECTION("nullptr")
    {
        CHECK_THROWS_AS(simplify(zero * nullptr), FatalError);
        CHECK_THROWS_AS(simplify(nullptr * zero), FatalError);
        ExpressionPtr nullexpr;
        CHECK_THROWS_AS(simplify(nullptr * nullexpr), FatalError);

        CHECK_THAT(simplify(nullptr), IdenticalTo(nullptr));
    }

    SECTION("addShiftLeft")
    {
        // addShiftLeft
        CHECK_THAT(simplify(fuseTernary(a + b << one + one)),
                   IdenticalTo(std::make_shared<rocRoller::Expression::Expression>(
                       AddShiftL{literal(33), literal(100), literal(2)})));
    }

    SECTION("bitFieldExtract")
    {
        CHECK_THAT(simplify(bfe(DataType::Int32, v, 0, 32)), IdenticalTo(v));
        CHECK_THAT(simplify(bfe(DataType::UInt32, v, 0, 32)),
                   IdenticalTo(convert(DataType::UInt32, v)));
    }

    SECTION("bitFieldCombine")
    {
        CHECK_THAT(simplify(bfc(v2, v, 16, 8, 0)), IdenticalTo(v));
        CHECK_THAT(simplify(bfc(v2, v, 0, 0, 32)), IdenticalTo(v2));
        CHECK_THAT(simplify(bfc(v3, v, 16, 0, 32)), IdenticalTo(bfe(DataType::Int32, v3, 16, 32)));
    }

    SECTION("concatenate")
    {
        CHECK_THAT(simplify(concat({v}, {DataType::Int32})), IdenticalTo(v));
    }
}

TEST_CASE("FuseAssociative ExpressionTransformation works",
          "[expression][expression-transformation]")
{
    using namespace rocRoller;
    auto context = TestContext::ForDefaultTarget();

    auto r
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Int32, 1);
    r->allocateNow();
    auto v = r->expression();

    auto zero = Expression::literal(0);
    auto one  = Expression::literal(1);
    auto a    = Expression::literal(33);
    auto b    = Expression::literal(100);
    auto c    = Expression::literal(12.f);

    SECTION("BitwiseAnd")
    {
        CHECK_THAT(fuseAssociative(v & Expression::fuseTernary(a + b << one)),
                   IdenticalTo(v & Expression::addShiftL(a, b, one)));
        CHECK_THAT(fuseAssociative(v & -one), IdenticalTo(v & -one));
        CHECK_THAT(fuseAssociative(v & one), EquivalentTo(v & one));
        CHECK_THAT(fuseAssociative((v & one) & a), EquivalentTo(v & one));
        CHECK_THAT(fuseAssociative(simplify((one & a) & v)), EquivalentTo(v & one));
        CHECK_THAT(fuseAssociative(a & (v & one)), EquivalentTo(v & one));
        CHECK_THAT(fuseAssociative((one & v) & a), EquivalentTo(v & one));
        CHECK_THAT(fuseAssociative(a & (one & v)), EquivalentTo(v & one));
        CHECK_THAT(fuseAssociative(((((v & one) & a) & a) & a) & a), EquivalentTo(v & one));
        CHECK_THAT(fuseAssociative(((((v & one) & a) + a) & a) & one),
                   IdenticalTo(((v & one) + a) & one));
    }

    SECTION("BitwiseOr")
    {
        CHECK_THAT(fuseAssociative(v | Expression::fuseTernary(a + b << one)),
                   IdenticalTo(v | Expression::addShiftL(a, b, one)));
        CHECK_THAT(fuseAssociative(v | -one), IdenticalTo(v | -one));
        CHECK_THAT(fuseAssociative(v | one), EquivalentTo(v | one));
        CHECK_THAT(fuseAssociative((v | one) | a), EquivalentTo(v | a));
        CHECK_THAT(fuseAssociative(simplify((one | a) | v)), EquivalentTo(v | a));
        CHECK_THAT(fuseAssociative(a | (v | one)), EquivalentTo(v | a));
        CHECK_THAT(fuseAssociative((one | v) | a), EquivalentTo(v | a));
        CHECK_THAT(fuseAssociative(a | (one | v)), EquivalentTo(v | a));
        CHECK_THAT(fuseAssociative(((((v | one) | a) | a) | a) | a), EquivalentTo(v | a));
        CHECK_THAT(fuseAssociative(((((v | one) | a) + a) | a) | one),
                   IdenticalTo(((v | a) + a) | a));
    }

    SECTION("Add")
    {
        CHECK_THAT(fuseAssociative((v + one) + a), IdenticalTo(v + Expression::literal(34)));
        CHECK_THAT(fuseAssociative(v + (one + a)), IdenticalTo(v + Expression::literal(34)));
        CHECK_THAT(fuseAssociative((one + v) + a), EquivalentTo(v + Expression::literal(34)));
    }

    SECTION("Multiply")
    {
        CHECK_THAT(fuseAssociative((v * one) * a), IdenticalTo(v * Expression::literal(33)));
        CHECK_THAT(fuseAssociative((v * a) * a), IdenticalTo(v * Expression::literal(33 * 33)));
        CHECK_THAT(fuseAssociative(v * (a * a)), IdenticalTo(v * Expression::literal(33 * 33)));
        CHECK_THAT(fuseAssociative((a * v) * a), EquivalentTo(v * Expression::literal(33 * 33)));
    }

    SECTION("ShiftL")
    {
        CHECK_THAT(fuseAssociative((v << one) << one), IdenticalTo(v << Expression::literal(2)));
        auto codegenOnly = Expression::EvaluationTimes{Expression::EvaluationTime::KernelExecute};
        CHECK(evaluationTimes(((a << v) << one) << one) == codegenOnly);
        CHECK(evaluationTimes((a << v) << one) == codegenOnly);
        CHECK(evaluationTimes(a << v) == codegenOnly);
        CHECK_THAT(fuseAssociative(((a << v) << one) << one),
                   IdenticalTo((a << v) << Expression::literal(2)));
    }

    SECTION("Arithmetic ShiftR")
    {
        CHECK_THAT(fuseAssociative((v >> one) >> one), IdenticalTo(v >> Expression::literal(2)));

        CHECK_THAT(fuseAssociative(((v >> one) >> one) >> one),
                   IdenticalTo(v >> Expression::literal(3)));

        CHECK_THAT(fuseAssociative(((one >> v) >> one) >> one),
                   IdenticalTo((one >> v) >> Expression::literal(2)));
    }

    SECTION("Logical shiftR")
    {
        CHECK_THAT(fuseAssociative(logicalShiftR(logicalShiftR(v, one), one)),
                   IdenticalTo(logicalShiftR(v, Expression::literal(2))));

        CHECK_THAT(fuseAssociative(logicalShiftR(logicalShiftR(logicalShiftR(v, one), one), one)),
                   IdenticalTo(logicalShiftR(v, Expression::literal(3))));

        CHECK_THAT(fuseAssociative(logicalShiftR(logicalShiftR(logicalShiftR(one, v), one), one)),
                   IdenticalTo(logicalShiftR(logicalShiftR(one, v), Expression::literal(2))));
    }

    SECTION("Subtraction")
    {
        // fuseAssociative does not affect non-associative ops
        CHECK_THAT(fuseAssociative((v - one) - a), IdenticalTo((v - one) - a));
    }

    SECTION("nullptr")
    {
        CHECK_THROWS_AS(fuseAssociative(std::make_shared<Expression::Expression>(
                            Expression::Multiply{zero, nullptr})),
                        FatalError);
        CHECK_THROWS_AS(fuseAssociative(std::make_shared<Expression::Expression>(
                            Expression::Multiply{nullptr, zero})),
                        FatalError);
        CHECK_THROWS_AS(fuseAssociative(std::make_shared<Expression::Expression>(
                            Expression::Multiply{nullptr, nullptr})),
                        FatalError);

        CHECK_THAT(Expression::fuseAssociative(nullptr), IdenticalTo(nullptr));
    }
}

TEST_CASE("FuseTernary ExpressionTransformation works", "[expression][expression-transformation]")
{
    using namespace rocRoller;
    auto context = TestContext::ForDefaultTarget();

    auto r
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Int32, 1);
    r->allocateNow();
    auto v = r->expression();
    auto r2
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Int32, 1);
    r2->allocateNow();
    auto v2 = r2->expression();

    auto zero = Expression::literal(0);
    auto one  = Expression::literal(1);
    auto a    = Expression::literal(33);
    auto b    = Expression::literal(100);
    auto c    = Expression::literal(12.f);

    SECTION("Non-transformed")
    {
        CHECK_THAT(fuseTernary(-one + one), IdenticalTo(-one + one));
        CHECK_THAT(fuseTernary(one + one), IdenticalTo(one + one));
    }

    SECTION("AddShiftL")
    {
        CHECK_THAT(fuseTernary((b + a) << one), IdenticalTo(addShiftL(b, a, one)));
        CHECK_THAT(fuseTernary((v + a) << one), IdenticalTo(addShiftL(v, a, one)));
        CHECK_THAT(fuseTernary((a + v) << one), IdenticalTo(addShiftL(a, v, one)));
        CHECK_THAT(fuseTernary((v + v2) << one), IdenticalTo(addShiftL(v, v2, one)));
        CHECK_THAT(fuseTernary(((v + a) << one) + v), IdenticalTo(addShiftL(v, a, one) + v));
        CHECK_THAT(fuseTernary(((b + v + a) << one) + v),
                   IdenticalTo(addShiftL((b + v), a, one) + v));

        CHECK_THAT(fuseTernary((((v + v2) << one) + v) + v2),
                   IdenticalTo((addShiftL(v, v2, one) + v) + v2));

        CHECK_THAT(fuseTernary((b + a) << v), IdenticalTo((b + a) << v));
    }

    SECTION("ShiftLAdd")
    {
        CHECK_THAT(fuseTernary((b << one) + a), IdenticalTo(shiftLAdd(b, one, a)));
        CHECK_THAT(fuseTernary((v << one) + a), IdenticalTo(shiftLAdd(v, one, a)));
        CHECK_THAT(fuseTernary((v << one) + v), IdenticalTo(shiftLAdd(v, one, v)));

        CHECK_THAT(fuseTernary((((v * v2) << one) + v) + v2),
                   IdenticalTo(shiftLAdd((v * v2), one, v) + v2));

        CHECK_THAT(fuseTernary((b << v) + a), IdenticalTo((b << v) + a));
    }

    SECTION("ShiftLAddShiftL")
    {
        //
        // Original: ShiftL(Add(ShiftL(v0:I, 1:I)I, 33:I)I, 1:I)I
        // Expected: ShiftL(ShiftLAdd(v0:I, 1:I, 33:I)I, 1:I)I
        //
        CHECK_THAT(fuseTernary(((v << one) + a) << one),
                   IdenticalTo((shiftLAdd(v, one, a)) << one));
    }

    SECTION("FMA")
    {
        CHECK_THAT(fuseTernary((b * one) + a), IdenticalTo(multiplyAdd(b, one, a)));
        CHECK_THAT(fuseTernary((v * one) + a), IdenticalTo(multiplyAdd(v, one, a)));
        CHECK_THAT(fuseTernary((v * one) + v), IdenticalTo(multiplyAdd(v, one, v)));

        CHECK_THAT(fuseTernary((((v + v2) * one) + v) + v2),
                   IdenticalTo(multiplyAdd((v + v2), one, v) + v2));

        CHECK_THAT(fuseTernary((((v * v2) * one) + v) + v2),
                   IdenticalTo(multiplyAdd((v * v2), one, v) + v2));

        auto fast = Expression::FastArithmetic(context.get());
        CHECK_THAT(fast((((v * v2) * one) + v) + v2), IdenticalTo(multiplyAdd(v, v2, v) + v2));
    }

    SECTION("nullptr")
    {
        CHECK_THROWS_AS(fuseTernary(std::make_shared<Expression::Expression>(
                            Expression::Multiply{zero, nullptr})),
                        FatalError);
        CHECK_THROWS_AS(fuseTernary(std::make_shared<Expression::Expression>(
                            Expression::Multiply{nullptr, zero})),
                        FatalError);
        CHECK_THROWS_AS(fuseTernary(std::make_shared<Expression::Expression>(
                            Expression::Multiply{nullptr, nullptr})),
                        FatalError);

        CHECK_THROWS_AS(Expression::fuseTernary(nullptr), FatalError);
    }
}

TEST_CASE("FastArithmetic includes translate time evaluation",
          "[expression][expression-transformation]")
{
    using namespace rocRoller;
    using Expression::literal;
    auto context = TestContext::ForDefaultTarget();

    auto zero = literal(0.f);
    auto c    = literal(12.f);

    Expression::FastArithmetic fastArith(context.get());
    CHECK(fastArith(nullptr).get() == nullptr);
    CHECK_THAT(fastArith(c * zero), IdenticalTo(literal(0.f)));
}

TEST_CASE("ConvertPropagation", "[expression][expression-transformation]")
{
    using namespace rocRoller;
    using enum DataType;
    using Expression::literal;
    auto context = TestContext::ForDefaultTarget();

    std::vector<Expression::ExpressionPtr> r64{
        3,
        Register::Value::Placeholder(context.get(), Register::Type::Vector, Int64, 1)
            ->expression()};

    std::vector<Expression::ExpressionPtr> r32{
        3,
        Register::Value::Placeholder(context.get(), Register::Type::Vector, Int32, 1)
            ->expression()};

    Expression::FastArithmetic fastArith(context.get());
    CHECK(fastArith(nullptr).get() == nullptr);

    SECTION("basic")
    {
        // Int32(r64 + r64) -> Int32(Int32(r64) + Int32(r64))
        CHECK_THAT(convertPropagation(convert(Int32, r64[0] + r64[1])),
                   IdenticalTo(convert(Int32, convert(Int32, r64[0]) + convert(Int32, r64[1]))));

        // Int32(r64 + r64 * r64) -> Int32(Int32(r64) + Int32(r64) * Int32(r64))
        CHECK_THAT(
            convertPropagation(convert(Int32, r64[0] + r64[1] * r64[2])),
            IdenticalTo(convert(
                Int32, convert(Int32, r64[0]) + convert(Int32, r64[1]) * convert(Int32, r64[2]))));

        // r64 + Int32(r64 * r64) -> r64 + Int32(Int32(r64) * Int32(r64))
        CHECK_THAT(
            convertPropagation(r64[0] + convert(Int32, r64[1] * r64[2])),
            IdenticalTo(r64[0] + convert(Int32, convert(Int32, r64[1]) * convert(Int32, r64[2]))));

        // Int64(r64) -> no change
        CHECK_THAT(convertPropagation(convert(Int64, r64[0])), IdenticalTo(convert(Int64, r64[0])));

        // Int32(r64) -> Int32(Int32(r64))
        CHECK_THAT(convertPropagation(convert(Int32, r64[0])),
                   IdenticalTo(convert(Int32, convert(Int32, r64[0]))));

        // Int32(r64 << r64) -> Int32(Int32(r64) << r64)
        CHECK_THAT(convertPropagation(convert(Int32, r64[0] << r64[1])),
                   IdenticalTo(convert(Int32, convert(Int32, r64[0]) << r64[1])));

        // Int32((r64 + r64) << r64) -> Int32((Int32(r64) + Int32(r64)) << r64)
        CHECK_THAT(convertPropagation(convert(Int32, addShiftL(r64[0], r64[1], r64[2]))),
                   IdenticalTo(convert(
                       Int32, addShiftL(convert(Int32, r64[0]), convert(Int32, r64[1]), r64[2]))));

        // Int32((r64 << r64) + r64) -> Int32((Int32(r64) << r64) + Int32(r64))
        CHECK_THAT(convertPropagation(convert(Int32, shiftLAdd(r64[0], r64[1], r64[2]))),
                   IdenticalTo(convert(
                       Int32, shiftLAdd(convert(Int32, r64[0]), r64[1], convert(Int32, r64[2])))));
    }

    SECTION("skipped datatypes")
    {
        Expression::ExpressionPtr f1 = literal(1.0);
        CHECK_THAT(convertPropagation(convert(Float, f1 + f1)),
                   IdenticalTo(convert(Float, f1 + f1)));

        Expression::ExpressionPtr halfx2
            = Register::Value::Placeholder(context.get(), Register::Type::Vector, Halfx2, 1)
                  ->expression();
        CHECK_THAT(convertPropagation(convert(Float, halfx2 + halfx2)),
                   IdenticalTo((convert(Float, halfx2 + halfx2))));
    }

    SECTION("nested convert")
    {
        // Int32(r64 + Int64(r32 * r32)) -> Int32(Int32(r64), Int64(r32 * r32))
        CHECK_THAT(
            convertPropagation(convert(Int32, r64[0] + convert(Int64, r32[1] * r32[2]))),
            IdenticalTo(convert(Int32, convert(Int32, r64[0]) + convert(Int64, r32[1] * r32[2]))));

        // Do not propagate existing converts to larger types
        // Int32(r64 + Int64(r64 * r64)) -> Int32(Int32(r64) + Int64(r64 * r64))
        auto expr = convertPropagation(convert(Int32, r64[0] + convert(Int64, r64[1] * r64[2])));
        CHECK_THAT(
            expr,
            IdenticalTo(convert(Int32, convert(Int32, r64[0]) + convert(Int64, r64[1] * r64[2]))));
    }

    SECTION("conditional")
    {
        auto cond = Register::Value::Placeholder(context.get(), Register::Type::Vector, Bool32, 1)
                        ->expression();
        auto expr
            = convertPropagation(convert(Int32, Expression::conditional(cond, r64[0], r64[1])));
        CHECK_THAT(expr,
                   IdenticalTo(convert(Int32,
                                       Expression::conditional(
                                           cond, convert(Int32, r64[0]), convert(Int32, r64[1])))));
    }
}

TEST_CASE("launchTimeSubExpressions works", "[expression][expression-transformation]")
{
    using namespace rocRoller;
    using Expression::literal;
    auto context = TestContext::ForDefaultTarget();

    auto command = std::make_shared<Command>();

    auto arg1Tag = command->allocateTag();
    auto arg1    = command->allocateArgument(
        DataType::Int32, arg1Tag, ArgumentType::Value, DataDirection::ReadOnly, "arg1");
    auto arg2Tag = command->allocateTag();
    auto arg2    = command->allocateArgument(
        DataType::Int32, arg2Tag, ArgumentType::Value, DataDirection::ReadOnly, "arg2");
    auto arg3Tag = command->allocateTag();
    auto arg3    = command->allocateArgument(
        DataType::Int64, arg3Tag, ArgumentType::Value, DataDirection::ReadOnly, "arg3");

    auto arg1e = arg1->expression();
    auto arg2e = arg2->expression();
    auto arg3e = arg3->expression();

    auto reg1e
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Int32, 1)
              ->expression();
    auto reg2e
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Int64, 1)
              ->expression();

    auto ex1 = (arg1e * Expression::literal(5)) * arg2e * arg3e;

    auto ex1_launch = launchTimeSubExpressions(ex1, context.get());

    auto argExpr = [&]() {
        auto arg = context->kernel()->arguments().at(0);
        CHECK_THAT(arg.expression, IdenticalTo(ex1));

        auto argPtr = std::make_shared<AssemblyKernelArgument>(arg);
        return std::make_shared<Expression::Expression>(argPtr);
    }();

    auto arg1e2 = launchTimeSubExpressions(arg1e, context.get());

    CHECK_THAT(ex1_launch, IdenticalTo(argExpr));

    auto restored = restoreCommandArguments(ex1_launch);

    CHECK_THAT(restored, IdenticalTo(ex1));

    auto ex2 = ex1 + arg1e;

    auto ex2_launch = launchTimeSubExpressions(ex2, context.get());

    auto arg2Expr = [&]() {
        auto arg = context->kernel()->arguments().at(1);
        CHECK_THAT(arg.expression, IdenticalTo(arg1e));

        auto argPtr = std::make_shared<AssemblyKernelArgument>(arg);
        return std::make_shared<Expression::Expression>(argPtr);
    }();

    CHECK_THAT(ex2_launch, IdenticalTo(argExpr + arg2Expr));

    std::vector<AssemblyKernelArgument> expectedArgs{
        {"Multiply_0", DataType::Int64, DataDirection::ReadOnly, ex1, 0, 8},
        {"arg1_1", DataType::Int32, DataDirection::ReadOnly, arg1e, 8, 4}};

    CHECK(expectedArgs == context->kernel()->arguments());
}

TEST_CASE("lowerPRNG transformation works", "[expression][expression-transformation]")
{
    using namespace rocRoller;
    using Expression::literal;
    auto context = TestContext::ForDefaultTarget();

    // Test replacing random number expression with equivalent expressions
    // when PRNG instruction is not available
    auto seed
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::UInt32, 1);
    seed->allocateNow();
    auto seedExpr = seed->expression();

    auto expr = std::make_shared<Expression::Expression>(Expression::RandomNumber{seedExpr});

    CHECK_THAT(lowerPRNG(expr, context.get()),
               IdenticalTo(
                   conditional((logicalShiftR(seedExpr, literal(31u)) & literal(1u)) == literal(1u),
                               literal(197u) ^ (seedExpr << literal(1u)),
                               seedExpr << literal(1u))));
}

TEST_CASE("combineShifts works", "[expression][expression-transformation]")
{
    using namespace rocRoller;
    using Expression::literal;
    auto ctx = TestContext::ForDefaultTarget();

    Expression::FastArithmetic fast(ctx.get());
    auto                       command = std::make_shared<Command>();
    auto                       argTag  = command->allocateTag();

    auto dataType = GENERATE(DataType::Int32, DataType::UInt32);

    DYNAMIC_SECTION(dataType)
    {
        auto dataTypeInfo = DataTypeInfo::Get(dataType);

        auto arg = command->allocateArgument(dataType, argTag, ArgumentType::Limit);

        auto argExp = fast(arg->expression());

        SECTION("Shift left then right masks off MSBs.")
        {
            auto exp     = logicalShiftR((argExp << literal(4u)), literal(4u));
            auto fastExp = fast(exp);
            auto maskExp = argExp & literal(0b00001111111111111111111111111111u, dataType);
            CHECK_THAT(fastExp, IdenticalTo(maskExp));

            for(auto v : TestValues::byType(dataType))
            {
                KernelArguments args;
                args.append("", v);
                auto bytes = args.runtimeArguments();

                CAPTURE(v);

                CHECK(evaluate(fastExp, bytes) == evaluate(exp, bytes));
            }
        }

        SECTION("Shift left then right (arithmetic) masks off MSBs for unsigned.")
        {
            auto exp     = (argExp << literal(4u)) >> literal(4u);
            auto fastExp = fast(exp);
            auto maskExp = argExp & literal(0b00001111111111111111111111111111u, dataType);
            if(dataTypeInfo.isSigned)
            {
                CHECK_THAT(fastExp, IdenticalTo(exp));
            }
            else
            {
                CHECK_THAT(fastExp, IdenticalTo(maskExp));
            }

            for(auto v : TestValues::byType(dataType))
            {
                KernelArguments args;
                args.append("", v);
                auto bytes = args.runtimeArguments();

                CAPTURE(v);

                CHECK(evaluate(fastExp, bytes) == evaluate(exp, bytes));
            }
        }

        SECTION("Shift left then right by different amounts is left alone.")
        {
            auto exp     = (argExp << literal(5u)) >> literal(4u);
            auto fastExp = fast(exp);
            if(dataTypeInfo.isSigned)
                CHECK_THAT(fastExp, IdenticalTo(exp));
            else
                CHECK_THAT(fastExp,
                           IdenticalTo(logicalShiftR((argExp << literal(5u)), literal(4u))));
        }

        SECTION("Shift right then left masks off LSBs.")
        {
            auto exp     = (argExp >> literal(5u)) << literal(5u);
            auto fastExp = fast(exp);
            auto maskExp = argExp & literal(0b11111111111111111111111111100000u, dataType);
            CHECK_THAT(fastExp, IdenticalTo(maskExp));

            for(auto v : TestValues::byType(dataType))
            {
                KernelArguments args;
                args.append("", v);
                auto bytes = args.runtimeArguments();

                CAPTURE(v);

                CHECK(evaluate(fastExp, bytes) == evaluate(exp, bytes));
            }
        }

        SECTION("Shift right then left by different amounts is left alone.")
        {
            auto exp     = (argExp >> literal(2u)) << literal(5u);
            auto fastExp = fast(exp);
            if(dataTypeInfo.isSigned)
                CHECK_THAT(fastExp, IdenticalTo(exp));
            else
                CHECK_THAT(fastExp, IdenticalTo(logicalShiftR(argExp, literal(2u)) << literal(5u)));
        }

        SECTION("Direct combineShift with shift left then right (arithmetic) masks off MSBs for "
                "unsigned.")
        {
            auto exp    = (argExp << literal(4u)) >> literal(4u);
            auto newExp = combineShifts(exp);

            auto maskExp = argExp & literal(0b00001111111111111111111111111111u, dataType);
            if(dataTypeInfo.isSigned)
            {
                CHECK_THAT(newExp, IdenticalTo(exp));
            }
            else
            {
                CHECK_THAT(newExp, IdenticalTo(maskExp));
            }
        }
    }

    dataType = GENERATE(DataType::Int64, DataType::UInt64);

    DYNAMIC_SECTION(dataType)
    {
        auto dataTypeInfo = DataTypeInfo::Get(dataType);

        auto arg = command->allocateArgument(dataType, argTag, ArgumentType::Limit);

        auto argExp = fast(arg->expression());

        SECTION("Shift left then right masks off MSBs.")
        {
            auto exp     = logicalShiftR((argExp << literal(4u)), literal(4u));
            auto fastExp = fast(exp);
            auto maskExp
                = argExp
                  & literal(0b0000111111111111111111111111111111111111111111111111111111111111ull,
                            dataType);
            CHECK_THAT(fastExp, IdenticalTo(maskExp));

            for(auto v : TestValues::byType(dataType))
            {
                KernelArguments args;
                args.append("", v);
                auto bytes = args.runtimeArguments();

                CAPTURE(v);

                CHECK(evaluate(fastExp, bytes) == evaluate(exp, bytes));
            }
        }

        SECTION("Shift left then right (arithmetic) masks off MSBs for unsigned.")
        {
            auto exp     = (argExp << literal(4u)) >> literal(4u);
            auto fastExp = fast(exp);
            auto maskExp
                = argExp
                  & literal(0b0000111111111111111111111111111111111111111111111111111111111111ull,
                            dataType);
            if(dataTypeInfo.isSigned)
            {
                CHECK_THAT(fastExp, IdenticalTo(exp));
            }
            else
            {
                CHECK_THAT(fastExp, IdenticalTo(maskExp));
            }

            for(auto v : TestValues::byType(dataType))
            {
                KernelArguments args;
                args.append("", v);
                auto bytes = args.runtimeArguments();

                CAPTURE(v);

                CHECK(evaluate(fastExp, bytes) == evaluate(exp, bytes));
            }
        }

        SECTION("Shift left then right by different amounts is left alone.")
        {
            auto exp     = (argExp << literal(5u)) >> literal(4u);
            auto fastExp = fast(exp);
            if(dataTypeInfo.isSigned)
                CHECK_THAT(fastExp, IdenticalTo(exp));
            else
                CHECK_THAT(fastExp, IdenticalTo(logicalShiftR(argExp << literal(5u), literal(4u))));
        }

        SECTION("Shift right then left masks off LSBs.")
        {
            auto exp     = (argExp >> literal(5u)) << literal(5u);
            auto fastExp = fast(exp);
            auto maskExp
                = argExp
                  & literal(0b1111111111111111111111111111111111111111111111111111111111100000ull,
                            dataType);
            CHECK_THAT(fastExp, IdenticalTo(maskExp));

            for(auto v : TestValues::byType(dataType))
            {
                KernelArguments args;
                args.append("", v);
                auto bytes = args.runtimeArguments();

                CAPTURE(v);

                CHECK(evaluate(fastExp, bytes) == evaluate(exp, bytes));
            }
        }

        SECTION("Shift right then left by different amounts is left alone.")
        {
            auto exp     = (argExp >> literal(2u)) << literal(5u);
            auto fastExp = fast(exp);
            if(dataTypeInfo.isSigned)
                CHECK_THAT(fastExp, IdenticalTo(exp));
            else
                CHECK_THAT(fastExp,
                           IdenticalTo((logicalShiftR(argExp, literal(2u)) << literal(5u))));
        }

        SECTION("Direct combineShift with shift left then right (arithmetic) masks off MSBs for "
                "unsigned.")
        {
            auto exp    = (argExp << literal(4u)) >> literal(4u);
            auto newExp = combineShifts(exp);

            auto maskExp
                = argExp
                  & literal(0b0000111111111111111111111111111111111111111111111111111111111111ull,
                            dataType);

            if(dataTypeInfo.isSigned)
            {
                CHECK_THAT(newExp, IdenticalTo(exp));
            }
            else
            {
                CHECK_THAT(newExp, IdenticalTo(maskExp));
            }
        }
    }
}

TEST_CASE("splitBitFieldCombine works", "[expression][expression-transformation]")
{
    using namespace rocRoller;
    auto context = TestContext::ForDefaultTarget();

    auto zero64  = Expression::literal(0, DataType::UInt64);
    auto zero32  = Expression::literal(0, DataType::UInt32);
    auto zero128 = Expression::literal(Buffer{0, 0, 0, 0});

    auto ones64 = Expression::literal(0xffffffffffffffffull, DataType::UInt64);
    auto ones32 = Expression::literal(0xfffffffful, DataType::UInt32);
    auto four   = Expression::literal(4, DataType::UInt32);

    auto r
        = Register::Value::Placeholder(context.get(), Register::Type::Scalar, DataType::UInt32, 1);
    r->allocateNow();
    auto reg32 = r->expression();

    auto r2
        = Register::Value::Placeholder(context.get(), Register::Type::Scalar, DataType::UInt64, 1);
    r2->allocateNow();
    auto reg64 = r2->expression();

    SECTION("Combine into first dword of 64bit and fold to constant")
    {
        auto expr = bfc(ones32, zero64, 0, 16, 8);

        std::vector<Expression::ExpressionPtr> operands{
            Expression::literal(0x00ff0000ul, DataType::UInt32), zero32};
        auto expected = concat(operands, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine into second dword of 64bit and fold to constant")
    {
        auto expr = bfc(ones32, zero64, 0, 48, 8);

        std::vector<Expression::ExpressionPtr> operands{
            zero32, Expression::literal(0x00ff0000ul, DataType::UInt32)};
        auto expected = concat(operands, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine into middle of first and second dword of 64bit and fold to constant")
    {
        auto expr = bfc(ones32, zero64, 0, 24, 16);

        std::vector<Expression::ExpressionPtr> operands{
            Expression::literal(0xff000000ul, DataType::UInt32),
            Expression::literal(0x000000fful, DataType::UInt32)};
        auto expected = concat(operands, DataType::UInt64);

        // TODO: concatenate could be folded to single 64bit constant
        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine into first dword of 64bit")
    {
        auto expr = bfc(reg32, zero64, 0, 16, 8);

        auto                                   expect_1 = bfc(reg32, zero32, 0, 16, 8);
        std::vector<Expression::ExpressionPtr> operands{expect_1, zero32};
        auto                                   expected = concat(operands, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine into second dword of 64bit")
    {
        auto expr = bfc(reg32, zero64, 0, 48, 8);

        auto                                   expect1 = bfc(reg32, zero32, 0, 16, 8);
        std::vector<Expression::ExpressionPtr> operands{zero32, expect1};
        auto                                   expected = concat(operands, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine into middle of first and second dword of 64bit")
    {
        auto expr = bfc(reg32, zero64, 0, 24, 16);

        auto                                   expect1 = bfc(reg32, zero32, 0, 24, 8);
        auto                                   expect2 = bfc(reg32, zero32, 8, 0, 8);
        std::vector<Expression::ExpressionPtr> operands{expect1, expect2};
        auto                                   expected = concat(operands, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Chain two BitfieldCombines into 64bit")
    {
        auto expr  = bfc(reg32, zero64, 0, 16, 8);
        auto expr2 = bfc(ones32, expr, 0, 48, 8);

        auto                                   expect1 = bfc(reg32, zero32, 0, 16, 8);
        std::vector<Expression::ExpressionPtr> operands{
            expect1, Expression::literal(0x00ff0000ul, DataType::UInt32)};
        auto expected = concat(operands, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr2), IdenticalTo(expected));
    }

    SECTION("Chain two BitfieldCombines into 64bit, second uses the into middle of first and "
            "second dword of 64bit")
    {
        auto expr  = bfc(reg32, zero64, 0, 16, 8);
        auto expr2 = bfc(ones32, expr, 0, 24, 16);

        auto                                   expect1 = bfc(reg32, zero32, 0, 16, 8);
        auto                                   expect2 = bfc(ones32, expect1, 0, 24, 8);
        std::vector<Expression::ExpressionPtr> operands{
            expect2, Expression::literal(0x000000fful, DataType::UInt32)};
        auto expected = concat(operands, DataType::UInt64);

        // TODO: the BitfieldCombine in expect_2 could be folded into zero32
        CHECK_THAT(splitBitfieldCombine(expr2), IdenticalTo(expected));
    }

    SECTION("Chain two BitfieldCombines into 64bit and fold to constant")
    {
        auto expr  = bfc(ones32, zero64, 0, 16, 8);
        auto expr2 = bfc(ones32, expr, 0, 40, 8);

        std::vector<Expression::ExpressionPtr> operands{
            Expression::literal(0x00ff0000ul, DataType::UInt32),
            Expression::literal(0x0000ff00ul, DataType::UInt32)};
        auto expected = concat(operands, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr2), IdenticalTo(expected));
    }

    SECTION("Combine into first dword of 128bit and fold to constant")
    {
        auto expr = bfc(ones32, zero128, 0, 16, 8);

        std::vector<Expression::ExpressionPtr> operands{
            Expression::literal(0x00ff0000ul, DataType::UInt32), zero32, zero32, zero32};
        auto expected = concat(operands, {DataType::UInt32, PointerType::Buffer});

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine two dword register into of 128bit constant")
    {
        auto expr = bfc(reg64, zero128, 0, 0, 64);

        std::vector<Expression::ExpressionPtr> operands{bfe(DataType::UInt32, reg64, 0, 32),
                                                        bfe(DataType::UInt32, reg64, 32, 32),
                                                        zero32,
                                                        zero32};
        auto expected = concat(operands, {DataType::UInt32, PointerType::Buffer});

        // TODO: The two first operands could be simplified into reg64
        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine 32bit across src dword boundary register into of 128bit constant")
    {
        auto expr = bfc(reg64, zero128, 16, 0, 32);

        std::vector<Expression::ExpressionPtr> operands{
            bfe(DataType::UInt32, reg64, 16, 32), zero32, zero32, zero32};
        auto expected = concat(operands, {DataType::UInt32, PointerType::Buffer});

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine 32bit across src dword boundary register into of 128bit constant across dst "
            "dword boundary")
    {
        auto expr = bfc(reg64, zero128, 16, 16, 32);

        std::vector<Expression::ExpressionPtr> operands{
            bfc(bfe(DataType::UInt32, reg64, 16, 16), zero32, 0, 16, 16),
            bfc(bfe(DataType::UInt32, reg64, 32, 16), zero32, 0, 0, 16),
            zero32,
            zero32};
        auto expected = concat(operands, {DataType::UInt32, PointerType::Buffer});

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("BitfieldCombine chain into 128bit")
    {
        auto expr  = bfc(reg32, zero128, 0, 90, 12);
        auto expr2 = bfc(four, expr, 0, 110, 8);

        auto                                   expect1 = bfc(reg32, zero32, 0, 26, 6);
        auto                                   expect2 = bfc(reg32, zero32, 6, 0, 6);
        auto                                   expect3 = bfc(four, expect2, 0, 14, 8);
        std::vector<Expression::ExpressionPtr> operands{zero32, zero32, expect1, expect3};
        auto expected = concat(operands, {DataType::UInt32, PointerType::Buffer});

        // TODO: the 4 in BitfieldCombine in expect_3 could be folded into zero32
        CHECK_THAT(splitBitfieldCombine(expr2), IdenticalTo(expected));
    }
}

TEST_CASE("Simplify Shift ExpressionTransformation works",
          "[expression][expression-transformation]")
{
    using namespace rocRoller;
    auto context = TestContext::ForDefaultTarget();

    auto zero = Expression::literal(0);
    auto one  = Expression::literal(1);
    auto a    = Expression::literal(33);
    auto b    = Expression::literal(35);
    auto c    = Expression::literal(30);
    auto e    = Expression::literal(64);

    SECTION("Shift right by more than 31 bits")
    {
        auto r = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        r->allocateNow();
        auto v = r->expression();

        auto expr = fuseAssociative(logicalShiftR(logicalShiftR(logicalShiftR(v, one), one), a));

        CHECK_THAT(expr, IdenticalTo(logicalShiftR(v, b)));
        auto expected = Expression::literal(0, DataType::Int32);
        CHECK_THAT(simplify(expr), IdenticalTo(expected));
    }

    SECTION("Shift right by more than 63 bits")
    {
        auto r = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Int64, 1);
        r->allocateNow();
        auto v = r->expression();

        auto expr = fuseAssociative(logicalShiftR(logicalShiftR(logicalShiftR(v, one), a), c));

        CHECK_THAT(expr, IdenticalTo(logicalShiftR(v, e)));
        auto expected = Expression::literal(0, DataType::Int64);
        CHECK_THAT(simplify(expr), IdenticalTo(expected));
    }

    SECTION("Shift left by more than 31 bits")
    {
        auto r = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        r->allocateNow();
        auto v = r->expression();

        auto expr     = v << a;
        auto expected = Expression::literal(0, DataType::Int32);
        CHECK_THAT(simplify(expr), IdenticalTo(expected));
    }

    SECTION("Shift left by more than 63 bits")
    {
        auto r = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Int64, 1);
        r->allocateNow();
        auto v = r->expression();

        auto expr     = v << e;
        auto expected = Expression::literal(0, DataType::Int64);
        CHECK_THAT(simplify(expr), IdenticalTo(expected));
    }
}

TEST_CASE("LowerUnsignedArithmeticShiftR ExpressionTransformation works",
          "[expression][expression-transformation]")
{
    using namespace rocRoller;
    auto context = TestContext::ForDefaultTarget();

    auto a = Expression::literal(21);

    SECTION("Unsigned ArithmeticShiftR should be lowered to LogicalShiftR")
    {
        auto r = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::UInt32, 1);
        r->allocateNow();
        auto v = r->expression();

        auto expr = std::make_shared<rocRoller::Expression::Expression>(
            rocRoller::Expression::ArithmeticShiftR{v, a});
        CHECK_THAT(lowerUnsignedArithmeticShiftR(expr), IdenticalTo(logicalShiftR(v, a)));
    }

    SECTION("Signed ArithmeticShiftR should NOT be lowered to LogicalShiftR")
    {
        auto r = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        r->allocateNow();
        auto v = r->expression();

        auto expr = std::make_shared<rocRoller::Expression::Expression>(
            rocRoller::Expression::ArithmeticShiftR{v, a});
        CHECK_THAT(lowerUnsignedArithmeticShiftR(expr), IdenticalTo(expr));
    }
}

TEST_CASE("BitfieldCombine expression and lowering", "[expression][expression-transformation]")
{
    using namespace rocRoller;

    auto context = TestContext::ForDefaultTarget();

    Register::AllocationOptions allocOptions{.contiguousChunkWidth = Register::FULLY_CONTIGUOUS};

    auto src = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::UInt32, 1, allocOptions);
    src->allocateNow();
    auto srcExpr = src->expression();

    auto dst = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::UInt32, 1, allocOptions);
    dst->allocateNow();
    auto dstExpr = dst->expression();

    auto const srcOffset = 10u;
    auto const dstOffset = 4u;
    auto const width     = 7u;

    auto offsetDiff = Expression::literal(srcOffset - dstOffset);

    SECTION("Lowering Basic")
    {
        auto bfc = std::make_shared<Expression::Expression>(
            Expression::BitfieldCombine{srcExpr, dstExpr, "", srcOffset, dstOffset, width});

        auto expected
            = logicalShiftR(
                  (Expression::literal(Raw32(((1u << width) - 1u) << srcOffset)) & srcExpr),
                  offsetDiff)
              | (Expression::literal(Raw32(~(((1u << width) - 1u) << dstOffset))) & dstExpr);

        CHECK_THAT(lowerBitfieldCombine(bfc), IdenticalTo(expected));
    }

    SECTION("Lowering with srcIsZero")
    {
        auto bfc = std::make_shared<Expression::Expression>(
            Expression::BitfieldCombine{srcExpr, dstExpr, "", srcOffset, dstOffset, width, true});

        auto expected
            = logicalShiftR(srcExpr, offsetDiff)
              | (Expression::literal(Raw32(~(((1u << width) - 1u) << dstOffset))) & dstExpr);

        CHECK_THAT(lowerBitfieldCombine(bfc), IdenticalTo(expected));
    }

    SECTION("Lowering with dstIsZero")
    {
        auto bfc = std::make_shared<Expression::Expression>(Expression::BitfieldCombine{
            srcExpr, dstExpr, "", srcOffset, dstOffset, width, std::nullopt, true});

        auto expected = logicalShiftR((Expression::literal(Raw32(((1u << width) - 1u) << srcOffset))
                                       & srcExpr),
                                      offsetDiff)
                        | dstExpr;

        CHECK_THAT(lowerBitfieldCombine(bfc), IdenticalTo(expected));
    }

    SECTION("Lowering with srcIsZero & dstIsZero")
    {
        auto bfc = std::make_shared<Expression::Expression>(Expression::BitfieldCombine{
            srcExpr, dstExpr, "", srcOffset, dstOffset, width, true, true});

        auto expected = logicalShiftR(srcExpr, offsetDiff) | dstExpr;

        CHECK_THAT(lowerBitfieldCombine(bfc), IdenticalTo(expected));
    }
}
