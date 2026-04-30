// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "ExpressionMatchers.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <common/SourceMatcher.hpp>
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
                   IdenticalTo(reinterpret(DataType::UInt32, v)));

        auto expr  = bfe(DataType::Int32, v, 16, 16);
        auto expr2 = bfe(DataType::Int32, expr, 0, 16);
        CHECK_THAT(simplify(expr2), IdenticalTo(expr));

        expr  = bfe(DataType::Int32, v, 8, 16);
        expr2 = bfe(DataType::Int32, expr, 0, 4);
        CHECK_THAT(simplify(expr2), IdenticalTo(bfe(DataType::Int32, v, 8, 4)));

        expr  = bfe(DataType::Int32, v, 8, 16);
        expr2 = bfe(DataType::Int32, expr, 8, 16);
        CHECK_THAT(simplify(expr2), IdenticalTo(expr2));
    }

    SECTION("bitFieldCombine")
    {
        CHECK_THAT(simplify(bfc(v2, v, 16, 8, 0)), IdenticalTo(v));
        CHECK_THAT(simplify(bfc(v2, v, 0, 0, 32)), IdenticalTo(v2));
        CHECK_THAT(simplify(bfc(v3, v, 16, 0, 32)), IdenticalTo(bfe(DataType::Int32, v3, 16, 32)));

        auto expr  = bfc(v2, zero, 0, 0, 16);
        auto expr2 = bfc(v, expr, 0, 0, 16);
        CHECK_THAT(simplify(expr2), IdenticalTo(bfc(v, zero, 0, 0, 16)));

        expr  = bfc(v2, zero, 0, 8, 16);
        expr2 = bfc(v, expr, 0, 0, 32);
        CHECK_THAT(simplify(expr2), IdenticalTo(v));

        expr  = bfc(v2, zero, 0, 8, 16);
        expr2 = bfc(v, expr, 0, 9, 32);
        CHECK_THAT(simplify(expr2), IdenticalTo(expr2));
    }

    SECTION("concatenate")
    {
        CHECK_THAT(simplify(concat({v}, {DataType::Int32})), IdenticalTo(v));
        CHECK_THAT(simplify(concat({literal(1u), literal(0u)}, {DataType::UInt64})),
                   IdenticalTo(literal(1ull, DataType::UInt64)));
        CHECK_THAT(
            simplify(concat({bfe(DataType::UInt32, v3, 0, 32), bfe(DataType::UInt32, v3, 32, 32)},
                            {DataType::UInt64})),
            IdenticalTo(v3));
    }

    SECTION("convert")
    {
        CHECK_THAT(simplify(convert(DataType::Int32, v)), IdenticalTo(v));
        CHECK_THAT(simplify(convert(DataType::UInt64, v3)), IdenticalTo(v3));
    }

    SECTION("reinterpret")
    {
        CHECK_THAT(simplify(reinterpret(DataType::Int32, v)), IdenticalTo(v));
        CHECK_THAT(simplify(reinterpret(DataType::UInt64, v3)), IdenticalTo(v3));
    }

    SECTION("conditional")
    {
        auto expr = conditional(b > zero, v2, v2 + v);
        CHECK_THAT(simplify(expr), IdenticalTo(v2));
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

TEST_CASE("FastArithmetic pipeline properly simplifies expressions",
          "[expression][expression-transformation]")
{
    using namespace rocRoller;
    using Expression::literal;
    auto context = TestContext::ForDefaultTarget();

    Expression::FastArithmetic fastArith(context.get());
    auto                       transforms = fastArith.getTransforms();

    auto isSimplify = [](const Expression::ExpressionTransformType& transformFunction) {
        using ExprTransformFuncPtrType = Expression::ExpressionPtr (*)(Expression::ExpressionPtr);
        const auto* funcPtr            = transformFunction.target<ExprTransformFuncPtrType>();
        return funcPtr && *funcPtr == Expression::simplify;
    };

    // Create a version with extra simplifies after each non-simplify transform
    std::vector<Expression::ExpressionTransformType> transformsExtraSimplify;
    for(const auto& transform : transforms)
    {
        transformsExtraSimplify.push_back(transform);
        if(!isSimplify(transform))
            transformsExtraSimplify.push_back(Expression::simplify);
    }

    // Create a version with only one simplify
    std::vector<Expression::ExpressionTransformType> transformsOneSimplify;
    bool                                             hasSimplify = false;
    for(const auto& transform : transforms)
    {
        if(isSimplify(transform))
        {
            if(hasSimplify)
                continue;

            hasSimplify = true;
        }
        transformsOneSimplify.push_back(transform);
    }

    auto tag83 = Expression::dataFlowTag(83, Register::Type::Vector, DataType::UInt32);
    auto expr  = (tag83 % literal(4)) * literal(32);
    expr       = (tag83 / literal(4)) * literal(128u) + expr;
    expr       = expr * literal(4u);
    expr       = expr / literal(8u);
    expr       = expr + literal(0u);
    expr       = std::make_shared<Expression::Expression>(Expression::ToScalar{expr});
    expr       = Expression::convert(DataType::UInt32, expr);

    // One simplify is not enough
    CHECK(!Expression::identical(fastArith.applyTransforms(expr, transforms),
                                 fastArith.applyTransforms(expr, transformsOneSimplify)));

    // There are enough simplifies
    CHECK_THAT(fastArith.applyTransforms(expr, transforms),
               IdenticalTo(fastArith.applyTransforms(expr, transformsExtraSimplify)));
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
        // Int32(r64 + Int64(r32 * r32)) -> Int32(Int32(r64) + Int32(r32 * r32))
        CHECK_THAT(
            convertPropagation(convert(Int32, r64[0] + convert(Int64, r32[1] * r32[2]))),
            IdenticalTo(convert(Int32, convert(Int32, r64[0]) + convert(Int32, r32[1] * r32[2]))));

        // Do not propagate existing converts to larger types
        // Int32(r64 + Int64(r64 * r64)) -> Int32(Int32(r64) + Int64(r64 * r64))
        auto expr = convertPropagation(convert(Int32, r64[0] + convert(Int64, r64[1] * r64[2])));
        CHECK_THAT(expr,
                   IdenticalTo(convert(
                       Int32,
                       convert(Int32, r64[0])
                           + convert(Int32, convert(Int32, r64[1]) * convert(Int32, r64[2])))));
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

    SECTION("special value types")
    {
        using namespace Expression;
        const auto tag  = dataFlowTag(72, Register::Type::Vector, Int64);
        const auto expr = convertPropagation(convert(Int32, tag + r64[0]));
        CHECK_THAT(expr, IdenticalTo(convert(Int32, convert(Int32, tag) + convert(Int32, r64[0]))));
        CHECK_THAT(simplify(expr), IdenticalTo(convert(Int32, tag) + convert(Int32, r64[0])));

        CHECK_THAT(convertPropagation(convert(
                       UInt32, literal(32u, DataType::UInt32) + literal(4u, DataType::UInt64))),
                   IdenticalTo(convert(UInt32,
                                       literal(32u, DataType::UInt32)
                                           + convert(UInt32, literal(4u, DataType::UInt64)))));
        CHECK_NOTHROW(evaluate(convertPropagation(
            convert(UInt32, literal(32u, DataType::UInt32) + literal(4u, DataType::UInt64)))));
    }

    SECTION(
        "Propagation stops on Div/ArithmeticShiftR/LogicalShiftR when converting 64-bit to 32-bit")
    {
        CHECK_THAT(convertPropagation(convert(Int32, logicalShiftR(r64[0], r32[0]))),
                   IdenticalTo(convert(Int32, convert(Int32, logicalShiftR(r64[0], r32[0])))));

        CHECK_THAT(convertPropagation(convert(Int32, (r64[0] >> r32[0]))),
                   IdenticalTo(convert(Int32, convert(Int32, (r64[0] >> r32[0])))));

        CHECK_THAT(convertPropagation(convert(Int32, (r64[0] / r64[1]))),
                   IdenticalTo(convert(Int32, convert(Int32, (r64[0] / r64[1])))));
    }

    SECTION("Other expressions")
    {
        CHECK_THAT(Expression::convertPropagation(Expression::convert(Int32, -r64[0])),
                   IdenticalTo(Expression::convert(Int32, -Expression::convert(Int32, r64[0]))));

        CHECK_THAT(
            convertPropagation(convert(
                Int32,
                Expression::concat({r64[0], r64[1]}, {DataType::None, PointerType::Buffer}))),
            IdenticalTo(convert(Int32,
                                Expression::concat({convert(Int32, r64[0]), convert(Int32, r64[1])},
                                                   {DataType::None, PointerType::Buffer}))));
    }
}

TEST_CASE("Nested Reinterpret simplification", "[expression][expression-transformation]")
{
    using namespace rocRoller;
    using namespace Expression;

    auto context = TestContext::ForDefaultTarget();

    auto r_int32
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::Int32, 1);
    r_int32->allocateNow();
    auto v = r_int32->expression();

    SECTION("Double reinterpret same size types")
    {
        auto expr       = reinterpret(DataType::Int32, reinterpret(DataType::Float, v));
        auto simplified = simplify(expr);

        CHECK_THAT(simplified, IdenticalTo(v));
    }

    SECTION("Reinterpret chain")
    {
        auto expr       = reinterpret(DataType::UInt32, reinterpret(DataType::Float, v));
        auto simplified = simplify(expr);

        CHECK_THAT(simplified, IdenticalTo(reinterpret(DataType::UInt32, v)));
    }

    SECTION("Reinterpret of literal")
    {
        auto lit        = literal(10.0f);
        auto expr       = reinterpret(DataType::UInt32, lit);
        auto simplified = simplify(expr);

        CHECK(std::holds_alternative<CommandArgumentValue>(*simplified));
        CHECK(std::get<uint32_t>(evaluate(simplified)) == 1092616192);
    }
}

TEST_CASE("launchTimeSubExpressions works", "[expression][expression-transformation]")
{
    using namespace rocRoller;
    using Expression::literal;

    KernelOptions opts;
    opts->minLaunchTimeExpressionComplexity = 5;
    auto context                            = TestContext::ForDefaultTarget(opts);

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
        CHECK_THAT(arg.getExpression(), IdenticalTo(ex1));

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
        CHECK_THAT(arg.getExpression(), IdenticalTo(arg1e));

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

        auto arg = command->allocateArgument(dataType, argTag, ArgumentType::Size);

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

        auto arg = command->allocateArgument(dataType, argTag, ArgumentType::Size);

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
    auto zero32  = Expression::literal(0, DataType::Raw32);
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

    SECTION("Combine into the first dword of 64 bit dst and fold to constant")
    {
        auto expr     = bfc(ones32, zero64, 0, 16, 8);
        auto expected = Expression::literal(0x0000000000ff0000ull, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine into the second dword of 64 bit dst and fold to constant")
    {
        auto expr     = bfc(ones32, zero64, 0, 48, 8);
        auto expected = Expression::literal(0x00ff000000000000ull, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine into the first and second dwords of 64 bit dst and fold to constant")
    {
        auto expr     = bfc(ones32, zero64, 0, 24, 16);
        auto expected = Expression::literal(0x000000ffff000000ull, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine into the first dword of 64 bit dst")
    {
        auto expr = bfc(reg32, zero64, 0, 16, 8);

        auto                                   expect_1 = bfc(reg32, zero32, 0, 16, 8);
        std::vector<Expression::ExpressionPtr> operands{expect_1, zero32};
        auto                                   expected = concat(operands, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine into the second dword of 64 bit dst")
    {
        auto expr = bfc(reg32, zero64, 0, 48, 8);

        auto                                   expect1 = bfc(reg32, zero32, 0, 16, 8);
        std::vector<Expression::ExpressionPtr> operands{zero32, expect1};
        auto                                   expected = concat(operands, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine into the first and second dwords of 64 bit dst")
    {
        auto expr = bfc(reg32, zero64, 0, 24, 16);

        // zero64     00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
        // expr       00000000 00000000 00000000 XXXXXXXX XXXXXXXX 00000000 00000000 00000000
        // expect1    XXXXXXXX 00000000 00000000 00000000
        // expect2    00000000 00000000 00000000 XXXXXXXX

        auto                                   expect1 = bfc(reg32, zero32, 0, 24, 8);
        auto                                   expect2 = bfc(reg32, zero32, 8, 0, 8);
        std::vector<Expression::ExpressionPtr> operands{expect1, expect2};
        auto                                   expected = concat(operands, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Chain two BitfieldCombines into 64 bit dst")
    {
        auto expr  = bfc(reg32, zero64, 0, 16, 8);
        auto expr2 = bfc(ones32, expr, 0, 48, 8);

        // zero64     00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
        // expr       00000000 00000000 00000000 00000000 00000000 XXXXXXXX 00000000 00000000
        // expr2      00000000 11111111 00000000 00000000 00000000 XXXXXXXX 00000000 00000000
        // expect1    00000000 XXXXXXXX 00000000 00000000
        // expect2    00000000 11111111 00000000 00000000

        auto                                   expect1 = bfc(reg32, zero32, 0, 16, 8);
        std::vector<Expression::ExpressionPtr> operands{
            expect1, Expression::literal(0x00ff0000ul, DataType::Raw32)};
        auto expected = concat(operands, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr2), IdenticalTo(expected));
    }

    SECTION("Chain two BitfieldCombines into 64 bit dst, the second bfc goes into the first and "
            "second dwords of dst")
    {
        auto expr  = bfc(reg32, zero64, 0, 16, 8);
        auto expr2 = bfc(ones32, expr, 0, 24, 16);

        // zero64     00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
        // expr       00000000 00000000 00000000 00000000 00000000 XXXXXXXX 00000000 00000000
        // expr2      00000000 00000000 00000000 11111111 11111111 XXXXXXXX 00000000 00000000
        // expect1    11111111 XXXXXXXX 00000000 00000000
        // expect2    00000000 00000000 00000000 11111111

        auto expect1 = bfc(reg32, Expression::literal(0xff000000ul, DataType::Raw32), 0, 16, 8);
        std::vector<Expression::ExpressionPtr> operands{
            expect1, Expression::literal(0x000000fful, DataType::Raw32)};
        auto expected = concat(operands, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr2), IdenticalTo(expected));
    }

    SECTION("Chain two BitfieldCombines into 64 bit dst and fold to constant")
    {
        auto expr     = bfc(ones32, zero64, 0, 16, 8);
        auto expr2    = bfc(ones32, expr, 0, 40, 8);
        auto expected = Expression::literal(0x0000ff0000ff0000ull, DataType::UInt64);

        CHECK_THAT(splitBitfieldCombine(expr2), IdenticalTo(expected));
    }

    SECTION("Combine into the first dword of 128 bit dst and fold to constant")
    {
        auto expr = bfc(ones32, zero128, 0, 16, 8);

        std::vector<Expression::ExpressionPtr> operands{
            Expression::literal(0x00ff0000ul, DataType::Raw32), zero32, zero32, zero32};
        auto expected = concat(operands, {DataType::None, PointerType::Buffer});

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine 64 bit register into 128 bit dst")
    {
        auto expr = bfc(reg64, zero128, 0, 0, 64);

        // zero128    0x 00000000 00000000 00000000 00000000
        // expr       0x 00000000 00000000 XXXXXXXX XXXXXXXX

        std::vector<Expression::ExpressionPtr> operands{reg64, zero32, zero32};
        auto expected = concat(operands, {DataType::None, PointerType::Buffer});

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine 64 bit register into second, third, and fourth dwords of 128 bit dst")
    {
        auto expr = bfc(reg64, zero128, 0, 57, 45);

        // zero128    0x 00000000 00000000 00000000 00000000
        // reg64      0x XXXXXXXX XXXXXXXX (45 bits used: bits 0-44)
        // expr       0x 000000XX XXXXXXXX XXXXXXX0 00000000
        //
        // Bit layout (45 bits from reg64[0:44] inserted at dest bits 57-101):
        //   dword0 (bits 0-31):   unchanged = zero32
        //   dword1 (bits 32-63):  bits 57-63 = reg64[0:6] (7 bits at positions 25-31)
        //   dword2 (bits 64-95):  bits 64-95 = reg64[7:38] (32 bits, crosses dword boundary)
        //                         split into: bfe(reg64, 7, 25) | (bfe(reg64, 32, 7) << 25)
        //   dword3 (bits 96-127): bits 96-101 = reg64[39:44] (6 bits at positions 0-5)

        auto dword2_lower = bfe(DataType::Raw32, reg64, 7, 25);
        auto dword2_upper = bfe(DataType::Raw32, reg64, 32, 7);
        auto dword2       = dword2_lower | (dword2_upper << Expression::literal(25u));

        std::vector<Expression::ExpressionPtr> operands{
            zero32,
            bfc(bfe(DataType::Raw32, reg64, 0, 7), zero32, 0, 25, 7),
            dword2,
            bfc(bfe(DataType::Raw32, reg64, 39, 6), zero32, 0, 0, 6)};
        auto expected = concat(operands, {DataType::None, PointerType::Buffer});

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine into the first dword of 128 bit dst, across src dword boundary")
    {
        auto expr = bfc(reg64, zero128, 16, 0, 32);

        // zero128    0x 00000000 00000000 00000000 00000000
        // reg64      0x 0000XXXX XXXX0000
        // expr       0x 00000000 00000000 00000000 XXXXXXXX
        //
        // bfe(reg64, 16, 32) extracts bits 16-47, crosses dword boundary at bit 32
        // Split into: bfe(reg64, 16, 16) | (bfe(reg64, 32, 16) << 16)

        auto dword0_lower = bfe(DataType::Raw32, reg64, 16, 16);
        auto dword0_upper = bfe(DataType::Raw32, reg64, 32, 16);
        auto dword0       = dword0_lower | (dword0_upper << Expression::literal(16u));

        std::vector<Expression::ExpressionPtr> operands{dword0, zero32, zero32, zero32};
        auto expected = concat(operands, {DataType::None, PointerType::Buffer});

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("Combine into the first and second dwords of 128 bit dst, across src dword boundary")
    {
        auto expr = bfc(reg64, zero128, 16, 16, 32);

        // zero128    0x 00000000 00000000 00000000 00000000
        // reg64      0x 0000XXXX XXXX0000
        // expr       0x 00000000 00000000 0000XXXX XXXX0000

        std::vector<Expression::ExpressionPtr> operands{
            bfc(bfe(DataType::Raw32, reg64, 16, 16), zero32, 0, 16, 16),
            bfc(bfe(DataType::Raw32, reg64, 32, 16), zero32, 0, 0, 16),
            zero32,
            zero32};
        auto expected = concat(operands, {DataType::None, PointerType::Buffer});

        CHECK_THAT(splitBitfieldCombine(expr), IdenticalTo(expected));
    }

    SECTION("BitfieldCombine chain into 128 bit")
    {
        auto expr  = bfc(reg32, zero128, 0, 90, 12);
        auto expr2 = bfc(four, expr, 0, 110, 8);

        // zero128  00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
        // expr     00000000 00000000 00000000 00XXXXXX XXXXXX00 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
        // four     00000000 00000000 00000000 00000100
        // expr2    00000000 00000001 00000000 00XXXXXX XXXXXX00 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
        // expect1  XXXXXX00 00000000 00000000 00000000
        // expect2  00000000 00000001 00000000 00XXXXXX

        // four is combined with an offset of 110, so its bit 1 goes to position 110 + 2. The other 7 bits from four are all zeros.

        auto expect1 = bfc(reg32, zero32, 0, 26, 6);
        auto expect2 = bfc(reg32, Expression::literal(0x00010000ul, DataType::Raw32), 6, 0, 6);
        std::vector<Expression::ExpressionPtr> operands{zero32, zero32, expect1, expect2};
        auto expected = concat(operands, {DataType::None, PointerType::Buffer});

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

TEST_CASE("SplitConcatenate ExpressionTransformation works",
          "[expression][expression-transformation]")
{
    using namespace rocRoller;
    using namespace Expression;
    auto context = TestContext::ForDefaultTarget();

    auto r
        = Register::Value::Placeholder(context.get(), Register::Type::Vector, DataType::UInt64, 1);
    r->allocateNow();
    auto v = r->expression();

    SECTION("Split uint64_t literal into two Raw32 operands")
    {
        auto                       uint64Literal = literal(0x0123456789ABCDEFull, DataType::UInt64);
        std::vector<ExpressionPtr> operands{uint64Literal, v};
        auto concatExpr = Concatenate{{operands}, {DataType::None, PointerType::Buffer}};

        Concatenate result = splitConcatenate(concatExpr);

        std::vector<ExpressionPtr> expectedOperands{
            literal(Raw32(0x89ABCDEF)), literal(Raw32(0x01234567)), v};
        auto expectedExpr = Concatenate{{expectedOperands}, {DataType::None, PointerType::Buffer}};

        CHECK_THAT(std::make_shared<rocRoller::Expression::Expression>(result),
                   IdenticalTo(std::make_shared<rocRoller::Expression::Expression>(expectedExpr)));
    }

    SECTION("Multiple uint64_t literals are all split")
    {
        auto uint64Literal1 = literal(0xFFFFFFFF00000000ull, DataType::UInt64);
        auto uint64Literal2 = literal(0x00000000FFFFFFFFull, DataType::UInt64);

        std::vector<ExpressionPtr> operands{uint64Literal1, uint64Literal2};
        auto concatExpr = Concatenate{{operands}, {DataType::None, PointerType::Buffer}};

        Concatenate result = splitConcatenate(concatExpr);

        std::vector<ExpressionPtr> expectedOperands{literal(Raw32(0x00000000)),
                                                    literal(Raw32(0xFFFFFFFF)),
                                                    literal(Raw32(0xFFFFFFFF)),
                                                    literal(Raw32(0x00000000))};
        auto expectedExpr = Concatenate{{expectedOperands}, {DataType::None, PointerType::Buffer}};

        CHECK_THAT(std::make_shared<rocRoller::Expression::Expression>(result),
                   IdenticalTo(std::make_shared<rocRoller::Expression::Expression>(expectedExpr)));
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

    SECTION("Lowering with width=32 (full width)")
    {
        auto const fullWidth   = 32u;
        auto const srcOffset32 = 0u;
        auto const dstOffset32 = 0u;

        auto bfc = std::make_shared<Expression::Expression>(
            Expression::BitfieldCombine{srcExpr, dstExpr, "", srcOffset32, dstOffset32, fullWidth});

        // When width=32, srcMask should be 0xFFFFFFFF (all bits)
        // and dstMask should be 0x00000000 (clear all bits)
        auto expected = (Expression::literal(Raw32(0xFFFFFFFFu)) & srcExpr)
                        | (Expression::literal(Raw32(0x00000000u)) & dstExpr);

        CHECK_THAT(lowerBitfieldCombine(bfc), IdenticalTo(expected));
    }
}

TEST_CASE("Code gen with ConvertPropagation", "[expression][expression-transformation][codegen]")
{
    using namespace rocRoller;
    auto context = TestContext::ForDefaultTarget();

    const auto srcDatatype = GENERATE(DataType::Int64, DataType::UInt64);
    const auto dstDatatype = GENERATE(DataType::Int32, DataType::UInt32);

    auto r64a
        = std::make_shared<Register::Value>(context.get(), Register::Type::Vector, srcDatatype, 1);
    r64a->allocateNow();
    auto r64b
        = std::make_shared<Register::Value>(context.get(), Register::Type::Vector, srcDatatype, 1);
    r64b->allocateNow();

    Register::ValuePtr d32;

    auto kb = [&]() -> Generator<Instruction> {
        co_yield Expression::generate(
            d32, convert(dstDatatype, r64a->expression() + r64b->expression()), context.get());
    };
    context.get()->schedule(kb());

    std::string expected;
    if(DataTypeInfo::Get(dstDatatype).isSigned)
        expected = R"(
            v_add_i32 v4, v0, v2
        )";
    else
        expected = R"(
            v_add_u32 v4, v0, v2
        )";

    INFO(context.output());
    CHECK(NormalizedSource(context.output()) == NormalizedSource(expected));
}
