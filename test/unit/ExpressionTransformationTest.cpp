#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/Operations/Command.hpp>

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

    auto zero  = Expression::literal(0);
    auto one   = Expression::literal(1);
    auto a     = Expression::literal(33);
    auto b     = Expression::literal(100);
    auto c     = Expression::literal(12.f);
    auto True  = Expression::literal(true);
    auto False = Expression::literal(false);

    // negate
    EXPECT_EQ(Expression::toString(simplify(-(one + one))), "Negate(2i)");

    // multiply
    EXPECT_EQ(Expression::toString(simplify(zero * one)), "0i");
    EXPECT_EQ(Expression::toString(simplify(c * zero)), "0.00000f");
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

    // bitwise and
    EXPECT_EQ(Expression::toString(simplify(one & b)), "0i");
    EXPECT_EQ(Expression::toString(simplify(one & a)), "1i");
    EXPECT_EQ(Expression::toString(simplify(v & zero)), "0i");
    EXPECT_EQ(Expression::toString(simplify(v & (zero + zero))), "0i");

    // logical and
    EXPECT_EQ(Expression::toString(simplify((v < one) && False)), "0b");
    EXPECT_EQ(Expression::toString(simplify((v < one) && True)), "LessThan(v0:I, 1i)");

    // logical or
    EXPECT_EQ(Expression::toString(simplify((v < one) || True)), "1b");
    EXPECT_EQ(Expression::toString(simplify((v < one) || False)), "LessThan(v0:I, 1i)");

    // shiftL
    EXPECT_EQ(Expression::toString(simplify(v << zero)), "v0:I");
    EXPECT_EQ(Expression::toString(simplify((one << one) << one)), "4i");
    EXPECT_EQ(Expression::toString(simplify(v << (zero << zero))), "v0:I");

    // signedShiftR
    EXPECT_EQ(Expression::toString(simplify(v >> zero)), "v0:I");
    EXPECT_EQ(Expression::toString(simplify((a >> one) >> one)), "8i");
    EXPECT_EQ(Expression::toString(simplify(v >> (zero >> zero))), "v0:I");

    // shiftR
    EXPECT_EQ(Expression::toString(simplify(logicalShiftR(v, zero))), "v0:I");
    EXPECT_EQ(Expression::toString(simplify(logicalShiftR(logicalShiftR(a, one), one))), "8i");
    EXPECT_EQ(Expression::toString(simplify(logicalShiftR(v, logicalShiftR(zero, zero)))), "v0:I");

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

    EXPECT_EQ(Expression::simplify(nullptr).get(), nullptr);
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

    // shiftL
    EXPECT_EQ(Expression::toString(fuseAssociative((v << one) << one)), "ShiftL(v0:I, 2i)");

    // signedShiftR
    EXPECT_EQ(Expression::toString(fuseAssociative(((v >> one) >> one) >> one)),
              "ArithmeticShiftR(v0:I, 3i)");

    // shiftR
    EXPECT_EQ(Expression::toString(fuseAssociative(logicalShiftR(logicalShiftR(v, one), one))),
              "LogicalShiftR(v0:I, 2i)");

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

    EXPECT_EQ(Expression::fuseAssociative(nullptr).get(), nullptr);
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

    EXPECT_THROW(Expression::fuseTernary(nullptr).get(), FatalError);
}

TEST_F(ExpressionTransformationTest, Fast)
{
    auto zero = Expression::literal(0);
    auto one  = Expression::literal(1);
    auto a    = Expression::literal(33);
    auto b    = Expression::literal(100);
    auto c    = Expression::literal(12.f);

    Expression::FastArithmetic fastArith(m_context);
    EXPECT_EQ(fastArith(nullptr).get(), nullptr);
    EXPECT_EQ(Expression::toString(fastArith(c * zero)), "0.00000f");
}

TEST_F(ExpressionTransformationTest, LaunchTimeSubExpressions)
{
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

    auto reg1e = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1)
                     ->expression();
    auto reg2e = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int64, 1)
                     ->expression();

    auto ex1 = (arg1e * Expression::literal(5)) * arg2e * arg3e;

    auto origStr = toString(ex1);

    auto ex1_launch = launchTimeSubExpressions(ex1, m_context);

    EXPECT_EQ(toString(ex1_launch), "Multiply_0");

    auto restored = restoreCommandArguments(ex1_launch);

    EXPECT_EQ(origStr, toString(restored));

    auto ex2 = ex1 + arg1e;

    auto ex2_launch = launchTimeSubExpressions(ex2, m_context);
    EXPECT_EQ(toString(ex2_launch), "Add(Multiply_0, arg1_1)");

    std::vector<AssemblyKernelArgument> expectedArgs{
        {"Multiply_0", DataType::Int64, DataDirection::ReadOnly, ex1, 0, 8},
        {"arg1_1", DataType::Int32, DataDirection::ReadOnly, arg1e, 8, 4}};

    EXPECT_EQ(expectedArgs, m_context->kernel()->arguments());
}
