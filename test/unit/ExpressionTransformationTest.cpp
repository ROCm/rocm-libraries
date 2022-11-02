
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

#include "GenericContextFixture.hpp"

class ExpressionTransformationTest : public GenericContextFixture
{
};

using namespace rocRoller;

TEST_F(ExpressionTransformationTest, Simplify)
{
    auto r = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);
    r->allocateNow();
    auto v = r->expression();

    auto zero = Expression::literal(0);
    auto one  = Expression::literal(1);
    auto a    = Expression::literal(33);
    auto b    = Expression::literal(100);
    auto c    = Expression::literal(12.f);

    // negate
    EXPECT_EQ(Expression::toString(simplify(-(one + one))), "Negate(2i)");

    // multiply
    EXPECT_EQ(Expression::toString(simplify(zero * one)), "0i");
    EXPECT_EQ(Expression::toString(simplify(c * zero)), "0i");
    EXPECT_EQ(Expression::toString(simplify(c * one)), "12.0000f");
    EXPECT_EQ(Expression::toString(simplify(v * zero)), "0i");
    EXPECT_EQ(Expression::toString(simplify(v * zero)), "0i");
    EXPECT_EQ(Expression::toString(simplify(a * b)), "3300i");

    // add
    EXPECT_EQ(Expression::toString(simplify(zero + one)), "1i");
    EXPECT_EQ(Expression::toString(simplify(zero + one)), "1i");
    EXPECT_EQ(Expression::toString(simplify(v + zero)), "v0:I");
    EXPECT_EQ(Expression::toString(simplify(a + b)), "133i");

    // divide
    EXPECT_EQ(Expression::toString(simplify(a / one)), "33i");
    EXPECT_EQ(Expression::toString(simplify(v / one)), "v0:I");
    EXPECT_EQ(Expression::toString(simplify(v / a)), "Divide(v0:I, 33i)");
    EXPECT_EQ(Expression::toString(simplify(b / v)), "Divide(100i, v0:I)");

    // mod
    EXPECT_EQ(Expression::toString(simplify(a % one)), "0i");
    EXPECT_EQ(Expression::toString(simplify(v % one)), "0i");
    EXPECT_EQ(Expression::toString(simplify(v % a)), "Modulo(v0:I, 33i)");
    EXPECT_EQ(Expression::toString(simplify(b % v)), "Modulo(100i, v0:I)");

    // bitwiseAnd
    EXPECT_EQ(Expression::toString(simplify(one & b)), "0i");
    EXPECT_EQ(Expression::toString(simplify(one & a)), "1i");
    EXPECT_EQ(Expression::toString(simplify(v & zero)), "0i");
    EXPECT_EQ(Expression::toString(simplify(v & (zero + zero))), "0i");

    // shiftL
    EXPECT_EQ(Expression::toString(simplify(one << zero)), "1i");

    EXPECT_THROW(
        simplify(std::make_shared<Expression::Expression>(Expression::Multiply{zero, nullptr})),
        FatalError);
    EXPECT_THROW(
        simplify(std::make_shared<Expression::Expression>(Expression::Multiply{nullptr, zero})),
        FatalError);
    EXPECT_THROW(
        simplify(std::make_shared<Expression::Expression>(Expression::Multiply{nullptr, nullptr})),
        FatalError);

    // addShiftLeft
    EXPECT_EQ(Expression::toString(simplify(Expression::fuseTernary(a + b << one + one))),
              "AddShiftL(33i, 100i, 2i)");

    EXPECT_EQ(Expression::simplify(nullptr), nullptr);
}

TEST_F(ExpressionTransformationTest, FuseAssociative)
{
    auto r = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);
    r->allocateNow();
    auto v = r->expression();

    auto zero = Expression::literal(0);
    auto one  = Expression::literal(1);
    auto a    = Expression::literal(33);
    auto b    = Expression::literal(100);
    auto c    = Expression::literal(12.f);

    EXPECT_EQ(Expression::toString(fuseAssociative(v & Expression::fuseTernary(a + b << one))),
              "BitwiseAnd(v0:I, AddShiftL(33i, 100i, 1i))");
    EXPECT_EQ(Expression::toString(fuseAssociative(v & -one)), "BitwiseAnd(v0:I, Negate(1i))");
    EXPECT_EQ(Expression::toString(fuseAssociative(v & one)), "BitwiseAnd(v0:I, 1i)");
    EXPECT_EQ(Expression::toString(fuseAssociative((v & one) & a)), "BitwiseAnd(v0:I, 1i)");
    EXPECT_EQ(Expression::toString(fuseAssociative(simplify((one & a) & v))),
              "BitwiseAnd(1i, v0:I)");
    EXPECT_EQ(Expression::toString(fuseAssociative(a & (v & one))), "BitwiseAnd(v0:I, 1i)");
    EXPECT_EQ(Expression::toString(fuseAssociative((one & v) & a)), "BitwiseAnd(1i, v0:I)");
    EXPECT_EQ(Expression::toString(fuseAssociative(a & (one & v))), "BitwiseAnd(1i, v0:I)");
    EXPECT_EQ(Expression::toString(fuseAssociative(((((v & one) & a) & a) & a) & a)),
              "BitwiseAnd(v0:I, 1i)");
    EXPECT_EQ(Expression::toString(fuseAssociative(((((v & one) & a) + a) & a) & one)),
              "BitwiseAnd(Add(BitwiseAnd(v0:I, 1i), 33i), 1i)");

    EXPECT_EQ(Expression::toString(fuseAssociative((v + one) + a)), "Add(v0:I, 34i)");

    EXPECT_EQ(
        Expression::toString(fuseAssociative((v - one) - a)),
        "Subtract(Subtract(v0:I, 1i), 33i)"); // fuseAssociative does not affect non-associative ops

    EXPECT_THROW(fuseAssociative(
                     std::make_shared<Expression::Expression>(Expression::Multiply{zero, nullptr})),
                 FatalError);
    EXPECT_THROW(fuseAssociative(
                     std::make_shared<Expression::Expression>(Expression::Multiply{nullptr, zero})),
                 FatalError);
    EXPECT_THROW(fuseAssociative(std::make_shared<Expression::Expression>(
                     Expression::Multiply{nullptr, nullptr})),
                 FatalError);

    EXPECT_EQ(Expression::fuseAssociative(nullptr), nullptr);
}

TEST_F(ExpressionTransformationTest, FuseTernary)
{
    auto r = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);
    r->allocateNow();
    auto v = r->expression();

    auto zero = Expression::literal(0);
    auto one  = Expression::literal(1);
    auto a    = Expression::literal(33);
    auto b    = Expression::literal(100);
    auto c    = Expression::literal(12.f);

    EXPECT_EQ(Expression::toString(fuseTernary(-one + one)), "Add(Negate(1i), 1i)");
    EXPECT_EQ(Expression::toString(fuseTernary(one + one)), "Add(1i, 1i)");
    EXPECT_EQ(Expression::toString(fuseTernary((b + a) << one)), "AddShiftL(100i, 33i, 1i)");
    EXPECT_EQ(Expression::toString(fuseTernary((b << one) + a)), "ShiftLAdd(100i, 1i, 33i)");
    EXPECT_EQ(Expression::toString(fuseTernary((simplify((b + a) << one)))), "266i");
    EXPECT_EQ(Expression::toString(fuseTernary(simplify((b << one) + a))), "233i");

    EXPECT_THROW(
        fuseTernary(std::make_shared<Expression::Expression>(Expression::Multiply{zero, nullptr})),
        FatalError);
    EXPECT_THROW(
        fuseTernary(std::make_shared<Expression::Expression>(Expression::Multiply{nullptr, zero})),
        FatalError);
    EXPECT_THROW(fuseTernary(std::make_shared<Expression::Expression>(
                     Expression::Multiply{nullptr, nullptr})),
                 FatalError);

    EXPECT_THROW(Expression::fuseTernary(nullptr), FatalError);
}

TEST_F(ExpressionTransformationTest, Fast)
{
    auto zero = Expression::literal(0);
    auto one  = Expression::literal(1);
    auto a    = Expression::literal(33);
    auto b    = Expression::literal(100);
    auto c    = Expression::literal(12.f);

    Expression::FastArithmetic fastArith(m_context);
    EXPECT_EQ(fastArith(nullptr), nullptr);
    EXPECT_EQ(Expression::toString(fastArith(c * zero)), "0i");
}
