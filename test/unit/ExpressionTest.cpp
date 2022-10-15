
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic.hpp>
#include <rocRoller/CodeGen/Arithmetic/Double.hpp>
#include <rocRoller/CodeGen/Arithmetic/Int32.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/CoordinateTransform/Dimension.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "CodeGen/Instruction.hpp"
#include "Expression_fwd.hpp"
#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "TestValues.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace ExpressionTest
{

    struct ExpressionTest : public GenericContextFixture
    {
        virtual std::string targetArchitecture() override
        {
            // MFMA, 64 lanes per wavefront
            return "gfx90a";
        }

        void testSerialization(Expression::ExpressionPtr expr);
    };

    TEST_F(ExpressionTest, Basic)
    {
        auto a = Expression::literal(1);
        auto b = Expression::literal(2);

        auto rc = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rc->allocateNow();

        auto expr1  = a + b;
        auto expr2  = b * expr1;
        auto expr3  = b * expr1 - rc->expression();
        auto expr4  = expr1 > expr2;
        auto expr5  = expr3 < expr4;
        auto expr6  = expr4 >= expr5;
        auto expr7  = expr5 <= expr6;
        auto expr8  = expr6 == expr7;
        auto expr9  = -expr2;
        auto expr10 = Expression::fuse(expr1 << b);
        auto expr11 = Expression::fuse((a << b) + b);

        auto sexpr1  = Expression::toString(expr1);
        auto sexpr2  = Expression::toString(expr2);
        auto sexpr3  = Expression::toString(expr3);
        auto sexpr4  = Expression::toString(expr4);
        auto sexpr5  = Expression::toString(expr5);
        auto sexpr6  = Expression::toString(expr6);
        auto sexpr7  = Expression::toString(expr7);
        auto sexpr8  = Expression::toString(expr8);
        auto sexpr9  = Expression::toString(expr9);
        auto sexpr10 = Expression::toString(expr10);
        auto sexpr11 = Expression::toString(expr11);

        EXPECT_EQ(sexpr1, "Add(1i, 2i)");
        EXPECT_EQ(sexpr2, "Multiply(2i, Add(1i, 2i))");
        EXPECT_EQ(sexpr3, "Subtract(Multiply(2i, Add(1i, 2i)), v0:I)");
        EXPECT_EQ(sexpr4, "GreaterThan(" + sexpr1 + ", " + sexpr2 + ")");
        EXPECT_EQ(sexpr5, "LessThan(" + sexpr3 + ", " + sexpr4 + ")");
        EXPECT_EQ(sexpr6, "GreaterThanEqual(" + sexpr4 + ", " + sexpr5 + ")");
        EXPECT_EQ(sexpr7, "LessThanEqual(" + sexpr5 + ", " + sexpr6 + ")");
        EXPECT_EQ(sexpr8, "Equal(" + sexpr6 + ", " + sexpr7 + ")");
        EXPECT_EQ(sexpr9, "Negate(" + sexpr2 + ")");
        EXPECT_EQ(sexpr10, "FusedAddShift(1i, 2i, 2i)");
        EXPECT_EQ(sexpr11, "FusedShiftAdd(1i, 2i, 2i)");

        Expression::EvaluationTimes expectedTimes{Expression::EvaluationTime::KernelExecute};
        EXPECT_EQ(expectedTimes, Expression::evaluationTimes(expr8));
        EXPECT_EQ(expectedTimes, Expression::evaluationTimes(expr10));
    }

    void ExpressionTest::testSerialization(Expression::ExpressionPtr expr)
    {
        auto yamlText = Expression::toYAML(expr);

        EXPECT_NE("", yamlText) << toString(expr);

        auto deserialized = Expression::fromYAML(yamlText);
        ASSERT_NE(nullptr, deserialized.get()) << yamlText << toString(expr);

        EXPECT_EQ(Expression::toString(deserialized), Expression::toString(expr));
        EXPECT_EQ(true, Expression::identical(deserialized, expr));
    }

    TEST_F(ExpressionTest, Serialization)
    {
        auto a = Expression::literal(1);
        auto b = Expression::literal(2);

        auto c = Register::Value::Literal(4.2f);
        auto d = Register::Value::Literal(Half(4.2f));

        auto kernelArg                   = std::make_shared<AssemblyKernelArgument>();
        kernelArg->name                  = "KernelArg1";
        kernelArg->variableType.dataType = DataType::Int32;
        kernelArg->expression            = Expression::literal(10);
        kernelArg->offset                = 1;
        kernelArg->size                  = 5;

        Expression::DataFlowTag dataFlow;
        dataFlow.tag              = 50;
        dataFlow.regType          = Register::Type::Vector;
        dataFlow.varType.dataType = DataType::Float;

        auto waveTile = std::make_shared<KernelGraph::CoordinateTransform::WaveTile>(5);

        auto expr1  = a + b;
        auto expr2  = b * expr1;
        auto expr3  = b * expr1 - c->expression();
        auto expr4  = expr1 > (expr2 + d->expression());
        auto expr5  = expr3 < expr4;
        auto expr6  = expr4 >= expr5;
        auto expr7  = expr5 <= expr6;
        auto expr8  = expr6 == expr7;
        auto expr9  = -expr2;
        auto expr10 = Expression::fuse(expr1 << b);
        auto expr11 = Expression::fuse((a << b) + b);
        auto expr12 = b >> std::make_shared<Expression::Expression>(kernelArg);
        auto expr13 = std::make_shared<Expression::Expression>(dataFlow) / a;
        auto expr14 = std::make_shared<Expression::Expression>(waveTile) + b;

        testSerialization(expr1);
        testSerialization(expr2);
        testSerialization(expr3);
        testSerialization(expr4);
        testSerialization(expr5);
        testSerialization(expr6);
        testSerialization(expr7);
        testSerialization(expr8);
        testSerialization(expr9);
        testSerialization(expr10);
        testSerialization(expr11);
        // TODO: Enable test when KernelArgumentPtr can be serialized
        //testSerialization(expr12);
        testSerialization(expr13);
        // TODO: Enable test when WaveTilePtr can be serialized
        //testSerialization(expr14);

        auto reg = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        reg->allocateNow();

        EXPECT_ANY_THROW(testSerialization(reg->expression()));
    }

    TEST_F(ExpressionTest, IdenticalTest)
    {
        auto a    = Expression::literal(1u);
        auto ap   = Expression::literal(1);
        auto b    = Expression::literal(2u);
        auto zero = Expression::literal(0u);

        auto rc = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rc->allocateNow();

        auto rd = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);
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

        EXPECT_TRUE(identical(expr1, expr2));
        EXPECT_FALSE(identical(expr1, expr3));
        EXPECT_FALSE(identical(ap + b, expr3));

        auto expr4 = c + d;
        auto expr5 = c + d + zero;

        EXPECT_FALSE(identical(expr1, expr4));

        EXPECT_FALSE(identical(expr4, expr5));
        EXPECT_TRUE(identical(expr4, simplify(expr5)));

        auto expr6 = e / f % d;
        auto expr7 = a + f;

        EXPECT_FALSE(identical(expr6, expr7));
        EXPECT_FALSE(identical(e, f));

        auto A_tile = std::make_shared<KernelGraph::CoordinateTransform::WaveTile>(0);
        auto B_tile = std::make_shared<KernelGraph::CoordinateTransform::WaveTile>(1);

        int M = 32;
        int N = 32;
        int K = 2;

        A_tile->sizes = {M, K};
        A_tile->vgpr  = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, M * K / 64);
        A_tile->vgpr->allocateNow();

        B_tile->sizes = {K, N};
        B_tile->vgpr  = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, K * N / 64);
        B_tile->vgpr->allocateNow();

        auto A = std::make_shared<Expression::Expression>(A_tile);
        auto B = std::make_shared<Expression::Expression>(B_tile);

        auto expr8 = A * B;
        auto expr9 = a * b;

        EXPECT_FALSE(identical(expr8, expr9));
        EXPECT_TRUE(identical(A * B, expr8));

        EXPECT_TRUE(Expression::identical(nullptr, nullptr));
        EXPECT_FALSE(identical(nullptr, a));
        EXPECT_FALSE(identical(a, nullptr));
    }

    TEST_F(ExpressionTest, BasicInstructions)
    {
        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rb->setName("rb");
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        auto expr1 = a + b;
        auto expr2 = b * expr1;

        Register::ValuePtr dest;
        m_context->schedule(Expression::generate(dest, expr2, m_context));

        // Explicitly copy the result into another register.
        auto dest2 = dest->placeholder();
        dest2->allocateNow();
        auto regIndexBefore = Generated(dest2->registerIndices())[0];

        m_context->schedule(Expression::generate(dest2, dest->expression(), m_context));
        auto regIndexAfter = Generated(dest2->registerIndices())[0];
        EXPECT_EQ(regIndexBefore, regIndexAfter);

        m_context->schedule(Expression::generate(dest2, expr2, m_context));
        regIndexAfter = Generated(dest2->registerIndices())[0];
        EXPECT_EQ(regIndexBefore, regIndexAfter);

        std::string expected = R"(
            v_add_i32 v2, v0, v1
            v_mul_lo_u32 v3, v1, v2

            // Note that v2 is reused
            v_mov_b32 v2, v3

            // Still storing into v2
            v_add_i32 v4, v0, v1
            v_mul_lo_u32 v2, v1, v4
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    TEST_F(ExpressionTest, ExpressionCommentsBasic)
    {
        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        auto expr1 = a + b;
        auto expr2 = b * expr1;

        setComment(expr1, "The Addition");
        appendComment(expr1, " extra comment");
        setComment(expr2, "The Multiplication");

        Register::ValuePtr dest;
        m_context->schedule(Expression::generate(dest, expr2, m_context));

        std::string expected = R"(
            // {The Multiplication: Multiply(v1:I, {The Addition extra comment: Add(v0:I, v1:I)})}
            // BEGIN: The Multiplication
            // BEGIN: The Addition extra comment
            // Allocated : 1 VGPR (Value: Int32): v2
            v_add_i32 v2, v0, v1
            // END: The Addition extra comment
            // Allocated : 1 VGPR (Value: Int32): v3
            v_mul_lo_u32 v3, v1, v2
            // Freeing : 1 VGPR (Value: Int32): v2
            // END: The Multiplication
        )";

        EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
    }

    TEST_F(ExpressionTest, ExpressionCommentsErrors)
    {
        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto a = ra->expression();
        EXPECT_THROW(setComment(a, "The a input"), FatalError);
        EXPECT_THROW(appendComment(a, "extra comment"), FatalError);
        EXPECT_EQ(getComment(a), "");

        Expression::ExpressionPtr expr1;
        EXPECT_THROW(setComment(expr1, "The first expression"), FatalError);
        EXPECT_THROW(appendComment(expr1, "extra"), FatalError);
        EXPECT_EQ(getComment(expr1), "");
    }

    TEST_F(ExpressionTest, GenerateInvalid)
    {
        Register::ValuePtr result;

        m_context->kernelOptions().preloadKernelArguments = false;
        m_context->schedule(m_context->kernel()->preamble());
        m_context->schedule(m_context->kernel()->prolog());

        auto reg = m_context->kernel()->workitemIndex()[0]->expression();

        {
            auto exp = Expression::magicMultiple(reg);
            EXPECT_ANY_THROW(m_context->schedule(Expression::generate(result, exp, m_context)))
                << output();
        }

        EXPECT_ANY_THROW(m_context->schedule(
            Expression::generate(result, Expression::magicShifts(reg), m_context)))
            << output();
        EXPECT_ANY_THROW(m_context->schedule(
            Expression::generate(result, Expression::magicSign(reg), m_context)))
            << output();

        std::shared_ptr<CommandArgument> arg;
        auto                             argExp = std::make_shared<Expression::Expression>(arg);
        EXPECT_ANY_THROW(m_context->schedule(Expression::generate(result, argExp, m_context)));
    }

    TEST_F(ExpressionTest, MatrixMultiply01)
    {
        int M = 32;
        int N = 32;
        int K = 2;
        int B = 1;

        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, M * K * B / 64);
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, K * N * B / 64);
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        auto expr = std::make_shared<Expression::Expression>(
            Expression::MatrixMultiply(a, b, M, N, K, B));

        Register::ValuePtr rc;
        m_context->schedule(Expression::generate(rc, expr, m_context));

        EXPECT_EQ(rc->regType(), Register::Type::Accumulator);
        EXPECT_EQ(rc->valueCount(), 16);

        auto result = R"(
            v_accvgpr_write a0, 0x0
            v_accvgpr_write a1, 0x0
            v_accvgpr_write a2, 0x0
            v_accvgpr_write a3, 0x0
            v_accvgpr_write a4, 0x0
            v_accvgpr_write a5, 0x0
            v_accvgpr_write a6, 0x0
            v_accvgpr_write a7, 0x0
            v_accvgpr_write a8, 0x0
            v_accvgpr_write a9, 0x0
            v_accvgpr_write a10, 0x0
            v_accvgpr_write a11, 0x0
            v_accvgpr_write a12, 0x0
            v_accvgpr_write a13, 0x0
            v_accvgpr_write a14, 0x0
            v_accvgpr_write a15, 0x0
            v_mfma_f32_32x32x2f32 a[0:15], v0, v1, a[0:15]
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(result));
    }

    TEST_F(ExpressionTest, MatrixMultiply02)
    {
        int M = 32;
        int N = 32;
        int K = 2;

        auto A_tile = std::make_shared<KernelGraph::CoordinateTransform::WaveTile>(0);
        auto B_tile = std::make_shared<KernelGraph::CoordinateTransform::WaveTile>(1);

        A_tile->sizes = {M, K};
        A_tile->vgpr  = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, M * K / 64);
        A_tile->vgpr->allocateNow();

        B_tile->sizes = {K, N};
        B_tile->vgpr  = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, K * N / 64);
        B_tile->vgpr->allocateNow();

        auto A = std::make_shared<Expression::Expression>(A_tile);
        auto B = std::make_shared<Expression::Expression>(B_tile);

        auto expr = A * B;

        Register::ValuePtr rc;
        m_context->schedule(Expression::generate(rc, expr, m_context));

        EXPECT_EQ(rc->regType(), Register::Type::Accumulator);
        EXPECT_EQ(rc->valueCount(), 16);

        auto result = R"(
            v_accvgpr_write a0, 0x0
            v_accvgpr_write a1, 0x0
            v_accvgpr_write a2, 0x0
            v_accvgpr_write a3, 0x0
            v_accvgpr_write a4, 0x0
            v_accvgpr_write a5, 0x0
            v_accvgpr_write a6, 0x0
            v_accvgpr_write a7, 0x0
            v_accvgpr_write a8, 0x0
            v_accvgpr_write a9, 0x0
            v_accvgpr_write a10, 0x0
            v_accvgpr_write a11, 0x0
            v_accvgpr_write a12, 0x0
            v_accvgpr_write a13, 0x0
            v_accvgpr_write a14, 0x0
            v_accvgpr_write a15, 0x0
            v_mfma_f32_32x32x2f32 a[0:15], v0, v1, a[0:15]
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(result));
    }

    TEST_F(ExpressionTest, ResultType)
    {
        auto vgprInt32
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1)
                  ->expression();
        auto vgprInt64
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int64, 1)
                  ->expression();
        auto vgprFloat
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Float, 1)
                  ->expression();
        auto sgprFloat
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Float, 1)
                  ->expression();
        auto vgprDouble
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Double, 1)
                  ->expression();
        auto sgprDouble
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Double, 1)
                  ->expression();

        auto sgprInt32
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Int32, 1)
                  ->expression();
        auto sgprInt64
            = Register::Value::Placeholder(m_context, Register::Type::Scalar, DataType::Int64, 1)
                  ->expression();

        auto agprFloat = Register::Value::Placeholder(
                             m_context, Register::Type::Accumulator, DataType::Float, 1)
                             ->expression();
        auto agprDouble = Register::Value::Placeholder(
                              m_context, Register::Type::Accumulator, DataType::Double, 1)
                              ->expression();

        auto litInt32  = Expression::literal<int32_t>(5);
        auto litInt64  = Expression::literal<int64_t>(5);
        auto litFloat  = Expression::literal(5.0f);
        auto litDouble = Expression::literal(5.0);

        Expression::ResultType rVgprInt32{Register::Type::Vector, DataType::Int32};
        Expression::ResultType rVgprInt64{Register::Type::Vector, DataType::Int64};
        Expression::ResultType rSgprInt32{Register::Type::Scalar, DataType::Int32};
        Expression::ResultType rSgprInt64{Register::Type::Scalar, DataType::Int64};
        Expression::ResultType rVgprFloat{Register::Type::Vector, DataType::Float};
        Expression::ResultType rSgprFloat{Register::Type::Scalar, DataType::Float};
        Expression::ResultType rVgprDouble{Register::Type::Vector, DataType::Double};
        Expression::ResultType rSgprDouble{Register::Type::Scalar, DataType::Double};
        Expression::ResultType rAgprFloat{Register::Type::Accumulator, DataType::Float};
        Expression::ResultType rAgprDouble{Register::Type::Accumulator, DataType::Double};

        EXPECT_EQ(rSgprInt64, resultType(sgprInt64));

        EXPECT_EQ(rVgprInt32, resultType(vgprInt32));
        EXPECT_EQ(rVgprInt64, resultType(vgprInt64));
        EXPECT_EQ(rVgprFloat, resultType(vgprFloat));
        EXPECT_EQ(rSgprFloat, resultType(sgprFloat));
        EXPECT_EQ(rVgprDouble, resultType(vgprDouble));
        EXPECT_EQ(rSgprDouble, resultType(sgprDouble));
        EXPECT_EQ(rAgprDouble, resultType(agprDouble));
        EXPECT_EQ(rAgprDouble, resultType(agprDouble));

        EXPECT_EQ(rVgprInt32, resultType(vgprInt32 + vgprInt32));
        EXPECT_EQ(rVgprInt32, resultType(vgprInt32 + sgprInt32));
        EXPECT_EQ(rVgprInt32, resultType(sgprInt32 - vgprInt32));
        EXPECT_EQ(rSgprInt32, resultType(sgprInt32 * sgprInt32));

        EXPECT_EQ(rVgprInt64, resultType(vgprInt64 + vgprInt32));
        EXPECT_EQ(rVgprInt64, resultType(vgprInt32 + vgprInt64));
        EXPECT_EQ(rVgprInt64, resultType(vgprInt64 + vgprInt64));

        EXPECT_EQ(rVgprFloat, resultType(vgprFloat + vgprFloat));
        EXPECT_EQ(rVgprFloat, resultType(vgprFloat - sgprFloat));
        EXPECT_EQ(rVgprFloat, resultType(litFloat * vgprFloat));
        EXPECT_EQ(rVgprFloat, resultType(vgprFloat * litFloat));

        EXPECT_EQ(rSgprInt32, resultType(sgprInt32 + sgprInt32));
        EXPECT_EQ(rSgprInt32, resultType(sgprInt32 + litInt32));
        EXPECT_EQ(rSgprInt32, resultType(litInt32 + sgprInt32));
        EXPECT_EQ(rSgprInt64, resultType(litInt32 + sgprInt64));
        EXPECT_EQ(rSgprInt64, resultType(sgprInt64 + litInt32));
        EXPECT_EQ(rSgprInt64, resultType(sgprInt64 + sgprInt32));

        Expression::ResultType rSgprBool32{Register::Type::Scalar, DataType::Bool32};
        Expression::ResultType rSpecialBool{Register::Type::Special, DataType::Bool};

        EXPECT_EQ(rSgprBool32, resultType(vgprFloat > vgprFloat));
        EXPECT_EQ(rSgprBool32, resultType(sgprFloat < vgprFloat));
        EXPECT_EQ(rSgprBool32, resultType(sgprDouble <= vgprDouble));
        EXPECT_EQ(rSgprBool32, resultType(sgprInt32 <= vgprInt32));
        EXPECT_EQ(rSgprBool32, resultType(litInt32 > vgprInt64));
        EXPECT_EQ(rSpecialBool, resultType(litInt32 <= sgprInt64));
        EXPECT_EQ(rSpecialBool, resultType(litInt32 >= sgprInt32));

        EXPECT_ANY_THROW(resultType(sgprDouble <= vgprFloat));
        EXPECT_ANY_THROW(resultType(vgprInt32 > vgprFloat));
    }

    TEST_F(ExpressionTest, EvaluateNoArgs)
    {
        auto a = std::make_shared<Expression::Expression>(1.0);
        auto b = std::make_shared<Expression::Expression>(2.0);

        auto expr1 = a + b;
        auto expr2 = b * expr1;

        auto expectedTimes = Expression::EvaluationTimes::All();
        EXPECT_EQ(expectedTimes, Expression::evaluationTimes(expr2));

        EXPECT_EQ(3.0, std::get<double>(Expression::evaluate(expr1)));
        EXPECT_EQ(6.0, std::get<double>(Expression::evaluate(expr2)));
    }

    TEST_F(ExpressionTest, EvaluateArgs)
    {
        VariableType doubleVal{DataType::Double, PointerType::Value};
        auto         ca = std::make_shared<CommandArgument>(nullptr, doubleVal, 0);
        auto         cb = std::make_shared<CommandArgument>(nullptr, doubleVal, 8);

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto expr1 = a + b;
        auto expr2 = b * expr1;
        auto expr3 = -expr2;

        struct
        {
            double a = 1.0;
            double b = 2.0;
        } args;
        RuntimeArguments runtimeArgs((uint8_t*)&args, sizeof(args));

        Expression::ResultType expected{Register::Type::Literal, DataType::Double};
        EXPECT_EQ(expected, resultType(expr2));
        EXPECT_EQ(6.0, std::get<double>(Expression::evaluate(expr2, runtimeArgs)));

        args.a = 2.0;
        EXPECT_EQ(8.0, std::get<double>(Expression::evaluate(expr2, runtimeArgs)));
        EXPECT_EQ(-8.0, std::get<double>(Expression::evaluate(expr3, runtimeArgs)));

        args.b = 1.5;
        EXPECT_EQ(5.25, std::get<double>(Expression::evaluate(expr2, runtimeArgs)));

        // Don't send in the runtimeArgs, can't evaluate the arguments.
        EXPECT_THROW(Expression::evaluate(expr2), std::runtime_error);

        Expression::EvaluationTimes expectedTimes{Expression::EvaluationTime::KernelLaunch};
        EXPECT_EQ(expectedTimes, Expression::evaluationTimes(expr2));
    }

    TEST_F(ExpressionTest, EvaluateMixedTypes)
    {
        auto one   = std::make_shared<Expression::Expression>(1.0);
        auto two   = std::make_shared<Expression::Expression>(2.0f);
        auto five  = std::make_shared<Expression::Expression>(5);
        auto seven = std::make_shared<Expression::Expression>(7.0);

        auto ptrNull = std::make_shared<Expression::Expression>((float*)nullptr);

        float x        = 3.0f;
        auto  ptrValid = std::make_shared<Expression::Expression>(&x);

        double y              = 9.0;
        auto   ptrDoubleValid = std::make_shared<Expression::Expression>(&y);

        // double + float -> double
        auto expr1 = one + two;
        // float * double -> double
        auto exprSix = two * expr1;

        // double - int -> double
        auto exprOne = exprSix - five;

        // float + int -> float
        auto exprSeven = two + five;

        EXPECT_EQ(6.0, std::get<double>(Expression::evaluate(exprSix)));
        EXPECT_EQ(1.0, std::get<double>(Expression::evaluate(exprOne)));
        EXPECT_EQ(7.0f, std::get<float>(Expression::evaluate(exprSeven)));

        Expression::ResultType litDouble{Register::Type::Literal, DataType::Double};
        Expression::ResultType litFloat{Register::Type::Literal, DataType::Float};
        Expression::ResultType litBool{Register::Type::Literal, DataType::Bool};

        EXPECT_EQ(litDouble, resultType(exprSix));
        // Result type not (yet?) defined for mixed integral/floating point types.
        EXPECT_ANY_THROW(resultType(exprOne));
        EXPECT_ANY_THROW(resultType(exprSeven));

        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(exprSix > exprOne)));
        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(exprSix >= exprOne)));
        EXPECT_EQ(false, std::get<bool>(Expression::evaluate(exprSix < exprOne)));
        EXPECT_EQ(false, std::get<bool>(Expression::evaluate(exprSix <= exprOne)));
        // EXPECT_EQ(true,  std::get<bool>(Expression::evaluate(exprSix != exprOne)));

        EXPECT_ANY_THROW(resultType(exprSix > exprOne));
        EXPECT_ANY_THROW(resultType(exprSix >= exprOne));
        EXPECT_ANY_THROW(resultType(exprSix < exprOne));
        EXPECT_ANY_THROW(resultType(exprSix <= exprOne));
        EXPECT_EQ(litBool, resultType(one > seven));

        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(exprSix < exprSeven)));
        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(exprSix <= exprSeven)));
        EXPECT_EQ(false, std::get<bool>(Expression::evaluate(exprSix > exprSeven)));
        EXPECT_EQ(false, std::get<bool>(Expression::evaluate(exprSix >= exprSeven)));

        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(one <= exprOne)));
        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(one == exprOne)));
        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(one >= exprOne)));
        // EXPECT_EQ(false,  std::get<bool>(Expression::evaluate(one != exprOne)));

        auto trueExp = std::make_shared<Expression::Expression>(true);
        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(trueExp == (one >= exprOne))));
        EXPECT_EQ(false, std::get<bool>(Expression::evaluate(trueExp == (one < exprOne))));

        // Pointer + double -> error.
        {
            auto exprThrow = ptrValid + exprOne;
            EXPECT_THROW(Expression::evaluate(exprThrow), std::runtime_error);
            EXPECT_ANY_THROW(resultType(exprThrow));
        }

        // Pointer * int -> error.
        {
            auto exprThrow = ptrValid * five;
            EXPECT_THROW(Expression::evaluate(exprThrow), std::runtime_error);
        }

        // Pointer + pointer -> error
        {
            auto exprThrow = ptrValid + ptrDoubleValid;
            EXPECT_THROW(Expression::evaluate(exprThrow), std::runtime_error);
            EXPECT_ANY_THROW(resultType(exprThrow));
        }

        // (float *) -  (double *) -> error
        {
            auto exprThrow = ptrValid - ptrDoubleValid;
            EXPECT_THROW(Expression::evaluate(exprThrow), std::runtime_error);
            EXPECT_ANY_THROW(resultType(exprThrow));
        }

        {
            auto exprThrow = ptrNull + five;
            // nullptr + int -> error;
            EXPECT_THROW(Expression::evaluate(exprThrow), std::runtime_error);
        }

        {
            auto exprThrow = -ptrNull;
            // -pointer -> error;
            EXPECT_THROW(Expression::evaluate(exprThrow), std::runtime_error);
        }

        {
            auto exprThrow = five + ptrNull;
            // Nullptr + int -> error;
            EXPECT_THROW(Expression::evaluate(exprThrow), std::runtime_error);
        }

        auto   exprXPlus5          = ptrValid + five;
        float* dontDereferenceThis = std::get<float*>(Expression::evaluate(exprXPlus5));
        auto   ptrDifference       = dontDereferenceThis - (&x);
        EXPECT_EQ(5, ptrDifference);

        auto expr10PlusX    = five + exprXPlus5;
        dontDereferenceThis = std::get<float*>(Expression::evaluate(expr10PlusX));
        ptrDifference       = dontDereferenceThis - (&x);
        EXPECT_EQ(10, ptrDifference);

        auto expr5PtrDiff = expr10PlusX - exprXPlus5;
        EXPECT_EQ(5, std::get<int64_t>(Expression::evaluate(expr5PtrDiff)));

        EXPECT_EQ(true, std::get<bool>(Expression::evaluate(expr10PlusX > ptrValid)));
        EXPECT_EQ(false, std::get<bool>(Expression::evaluate(expr10PlusX < ptrValid)));
    }

    TEST_F(ExpressionTest, EqualityTest)
    {
        auto ra = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        ra->setName("ra");
        ra->allocateNow();

        auto rb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        rb->setName("rb");
        rb->allocateNow();

        auto a = ra->expression();
        auto b = rb->expression();

        auto expr1 = a + b;
        auto expr2 = b * expr1;
        auto expr3 = expr1 == expr2;

        Register::ValuePtr destReg;
        m_context->schedule(Expression::generate(destReg, expr3, m_context));

        auto result = R"(
            v_add_i32 v2, v0, v1
            v_add_i32 v3, v0, v1
            v_mul_lo_u32 v4, v1, v3
            v_cmp_eq_i32 s[0:1], v2, v4
        )";

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(result));
    }

    TEST_F(ExpressionTest, EvaluateConditional)
    {
        auto command = std::make_shared<Command>();
        auto ca      = command->allocateArgument({DataType::Double, PointerType::Value});
        auto cb      = command->allocateArgument({DataType::Double, PointerType::Value});

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto vals_gt = a > b;
        auto vals_lt = a < b;
        auto vals_ge = a >= b;
        auto vals_le = a <= b;
        auto vals_eq = a == b;

        auto expr_gt = a > (a + b);
        auto expr_lt = a < (a + b);
        auto expr_ge = a >= (a + b);
        auto expr_le = a <= (a + b);
        auto expr_eq = a == (a + b);

        for(double aVal : TestValues::doubleValues)
        {
            for(double bVal : TestValues::doubleValues)
            {
                KernelArguments runtimeArgs;
                runtimeArgs.append("a", aVal);
                runtimeArgs.append("b", bVal);
                auto args = runtimeArgs.runtimeArguments();

                EXPECT_EQ(aVal > bVal, std::get<bool>(Expression::evaluate(vals_gt, args)));
                EXPECT_EQ(aVal < bVal, std::get<bool>(Expression::evaluate(vals_lt, args)));
                EXPECT_EQ(aVal >= bVal, std::get<bool>(Expression::evaluate(vals_ge, args)));
                EXPECT_EQ(aVal <= bVal, std::get<bool>(Expression::evaluate(vals_le, args)));
                EXPECT_EQ(aVal == bVal, std::get<bool>(Expression::evaluate(vals_eq, args)));

                EXPECT_EQ(aVal > (aVal + bVal),
                          std::get<bool>(Expression::evaluate(expr_gt, args)));
                EXPECT_EQ(aVal < (aVal + bVal),
                          std::get<bool>(Expression::evaluate(expr_lt, args)));
                EXPECT_EQ(aVal >= (aVal + bVal),
                          std::get<bool>(Expression::evaluate(expr_ge, args)));
                EXPECT_EQ(aVal <= (aVal + bVal),
                          std::get<bool>(Expression::evaluate(expr_le, args)));
                EXPECT_EQ(aVal == (aVal + bVal),
                          std::get<bool>(Expression::evaluate(expr_eq, args)));
            }
        }
    }

    TEST_F(ExpressionTest, EvaluateShifts)
    {
        auto command = std::make_shared<Command>();
        auto ca      = command->allocateArgument({DataType::Int32, PointerType::Value});
        auto cb      = command->allocateArgument({DataType::Int32, PointerType::Value});

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto vals_shiftL       = a << b;
        auto vals_shiftR       = shiftR(a, b);
        auto vals_signedShiftR = a >> b;

        auto expr_shiftL       = (a + b) << b;
        auto expr_shiftR       = shiftR(a + b, b);
        auto expr_signedShiftR = (a + b) >> b;

        for(auto aVal : TestValues::int32Values)
        {
            for(auto bVal : TestValues::shiftValues)
            {
                KernelArguments runtimeArgs;
                runtimeArgs.append("a", aVal);
                runtimeArgs.append("b", bVal);
                auto args = runtimeArgs.runtimeArguments();

                EXPECT_EQ(aVal << bVal, std::get<int>(Expression::evaluate(vals_shiftL, args)));
                EXPECT_EQ(static_cast<unsigned int>(aVal) >> bVal,
                          std::get<int>(Expression::evaluate(vals_shiftR, args)));
                EXPECT_EQ(aVal >> bVal,
                          std::get<int>(Expression::evaluate(vals_signedShiftR, args)));

                EXPECT_EQ((aVal + bVal) << bVal,
                          std::get<int>(Expression::evaluate(expr_shiftL, args)));
                EXPECT_EQ(static_cast<unsigned int>(aVal + bVal) >> bVal,
                          std::get<int>(Expression::evaluate(expr_shiftR, args)));
                EXPECT_EQ((aVal + bVal) >> bVal,
                          std::get<int>(Expression::evaluate(expr_signedShiftR, args)));
            }
        }
    }

    TEST_F(ExpressionTest, EvaluateBitwiseOps)
    {
        auto command = std::make_shared<Command>();
        auto ca      = command->allocateArgument({DataType::Int32, PointerType::Value});
        auto cb      = command->allocateArgument({DataType::Int32, PointerType::Value});

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto vals_and = a & b;

        auto expr_and = (a + b) & b;
        auto expr_or  = (a + b) | b;

        for(auto aVal : TestValues::int32Values)
        {
            for(auto bVal : TestValues::int32Values)
            {
                KernelArguments runtimeArgs;
                runtimeArgs.append("a", aVal);
                runtimeArgs.append("b", bVal);
                auto args = runtimeArgs.runtimeArguments();

                EXPECT_EQ(aVal & bVal, std::get<int>(Expression::evaluate(vals_and, args)));
                EXPECT_EQ((aVal + bVal) & bVal,
                          std::get<int>(Expression::evaluate(expr_and, args)));
                EXPECT_EQ((aVal + bVal) | bVal, std::get<int>(Expression::evaluate(expr_or, args)));
            }
        }
    }

    TEST_F(ExpressionTest, EvaluateMultiplyHigh)
    {
        auto command = std::make_shared<Command>();
        auto ca      = command->allocateArgument({DataType::Int32, PointerType::Value});
        auto cb      = command->allocateArgument({DataType::Int32, PointerType::Value});

        auto a = std::make_shared<Expression::Expression>(ca);
        auto b = std::make_shared<Expression::Expression>(cb);

        auto expr1 = multiplyHigh(a, b);

        auto expr2 = multiplyHigh(a + b, b);

        std::vector<int> a_values = {-21474836,
                                     -146000,
                                     -1,
                                     0,
                                     1,
                                     2,
                                     4,
                                     5,
                                     7,
                                     12,
                                     19,
                                     33,
                                     63,
                                     906,
                                     3017123,
                                     800000,
                                     1234456,
                                     4022112};
        for(auto aVal : a_values)
        {
            for(auto bVal : a_values)
            {
                KernelArguments runtimeArgs;
                runtimeArgs.append("a", aVal);
                runtimeArgs.append("b", bVal);
                auto args = runtimeArgs.runtimeArguments();

                EXPECT_EQ((aVal * (int64_t)bVal) >> 32,
                          std::get<int>(Expression::evaluate(expr1, args)));

                EXPECT_EQ(((aVal + bVal) * (int64_t)bVal) >> 32,
                          std::get<int>(Expression::evaluate(expr2, args)));
            }
        }
    }

    TEST_F(ExpressionTest, EvaluateConvertExpressions)
    {
        using namespace Expression;

        float  a = 1.25f;
        Half   b = 1.1111;
        double c = 5.2619;

        auto a_exp = literal(a);
        auto b_exp = literal(b);
        auto c_exp = literal(c);

        auto exp1 = convert<DataType::Half>(a_exp);
        auto exp2 = convert<DataType::Half>(b_exp);
        auto exp3 = convert<DataType::Half>(c_exp);

        EXPECT_EQ(resultVariableType(exp1).dataType, DataType::Half);
        EXPECT_EQ(resultVariableType(exp2).dataType, DataType::Half);
        EXPECT_EQ(resultVariableType(exp3).dataType, DataType::Half);

        EXPECT_EQ(std::get<Half>(evaluate(exp1)), static_cast<Half>(a));
        EXPECT_EQ(std::get<Half>(evaluate(exp2)), b);
        EXPECT_EQ(std::get<Half>(evaluate(exp3)), static_cast<Half>(c));

        auto exp4 = convert<DataType::Float>(a_exp);
        auto exp5 = convert<DataType::Float>(b_exp);
        auto exp6 = convert<DataType::Float>(c_exp);

        EXPECT_EQ(resultVariableType(exp4).dataType, DataType::Float);
        EXPECT_EQ(resultVariableType(exp5).dataType, DataType::Float);
        EXPECT_EQ(resultVariableType(exp6).dataType, DataType::Float);

        EXPECT_EQ(std::get<float>(evaluate(exp4)), a);
        EXPECT_EQ(std::get<float>(evaluate(exp5)), static_cast<float>(b));
        EXPECT_EQ(std::get<float>(evaluate(exp6)), static_cast<float>(c));
    }

    TEST_F(ExpressionTest, VariantTest)
    {
        int32_t  x1          = 3;
        auto     intPtr      = Expression::literal(&x1);
        int64_t  x2          = 3L;
        auto     intLongPtr  = Expression::literal(&x2);
        uint32_t x3          = 3u;
        auto     uintPtr     = Expression::literal(&x3);
        uint64_t x4          = 3UL;
        auto     uintLongPtr = Expression::literal(&x4);
        float    x5          = 3.0f;
        auto     floatPtr    = Expression::literal(&x5);
        double   x6          = 3.0;
        auto     doublePtr   = Expression::literal(&x6);

        auto intExpr    = Expression::literal(1);
        auto uintExpr   = Expression::literal(1u);
        auto floatExpr  = Expression::literal(1.0f);
        auto doubleExpr = Expression::literal(1.0);
        auto boolExpr   = Expression::literal(true);

        auto v_a
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Double, 1);
        v_a->allocateNow();

        Expression::Expression    value    = Register::Value::Literal(1);
        Expression::ExpressionPtr valuePtr = std::make_shared<Expression::Expression>(value);

        Expression::Expression    tag    = Expression::DataFlowTag();
        Expression::ExpressionPtr tagPtr = std::make_shared<Expression::Expression>(tag);
        Expression::Expression    waveTile
            = std::make_shared<KernelGraph::CoordinateTransform::WaveTile>(1);
        Expression::ExpressionPtr waveTilePtr = std::make_shared<Expression::Expression>(waveTile);

        std::vector<Expression::ExpressionPtr> exprs = {
            intExpr,
            uintExpr,
            floatExpr,
            doubleExpr,
            boolExpr,
            intExpr + intExpr,
            intExpr - intExpr,
            intExpr * intExpr,
            intExpr / intExpr,
            intExpr % intExpr,
            intExpr << intExpr,
            intExpr >> intExpr,
            shiftR(intExpr, intExpr),
            intExpr & intExpr,
            intExpr ^ intExpr,
            intExpr > intExpr,
            intExpr < intExpr,
            intExpr >= intExpr,
            intExpr <= intExpr,
            intExpr == intExpr,
            -intExpr,
            intPtr,
            intLongPtr,
            uintPtr,
            uintLongPtr,
            floatPtr,
            doublePtr,
            valuePtr,
        };

        auto testFunc = [](auto const& expr) {
            EXPECT_NO_THROW(Expression::toString(expr));
            EXPECT_NO_THROW(Expression::evaluationTimes(expr));
        };

        for(auto const& expr : exprs)
        {
            testFunc(expr);
            EXPECT_NO_THROW(Expression::evaluate(expr));
            EXPECT_NO_THROW(Expression::fastDivision(expr, m_context));
        }

        testFunc(v_a);
        EXPECT_THROW(Expression::evaluate(v_a), FatalError);

        testFunc(tag);
        EXPECT_THROW(Expression::evaluate(tag), FatalError);

        testFunc(tagPtr);
        EXPECT_THROW(Expression::evaluate(tagPtr), FatalError);
        EXPECT_NO_THROW(Expression::fastDivision(tagPtr, m_context));

        testFunc(value);
        EXPECT_NO_THROW(Expression::evaluate(value));

        testFunc(waveTile);
        EXPECT_THROW(Expression::evaluate(waveTile), FatalError);

        testFunc(waveTilePtr);
        EXPECT_THROW(Expression::evaluate(waveTilePtr), FatalError);
        EXPECT_NO_THROW(Expression::fastDivision(waveTilePtr, m_context));
    }

    class ARCH_ExpressionTest : public GPUContextFixture
    {
    };

    TEST_P(ARCH_ExpressionTest, ExpressionTreeDouble)
    {
        auto v_a
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Double, 1);

        auto v_b
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Double, 1);

        auto a = v_a->expression();
        auto b = v_b->expression();

        auto expr = -b * (a + b);

        auto k = m_context->kernel();

        k->setKernelName("ExpressionTreeDouble");

        k->addArgument(
            {"result", {DataType::Double, PointerType::PointerGlobal}, DataDirection::WriteOnly});
        k->addArgument({"a", DataType::Double});
        k->addArgument({"b", DataType::Double});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            co_yield v_a->allocate();
            co_yield v_b->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

            co_yield m_context->copier()->copy(v_a, s_a, "Move pointer");
            co_yield m_context->copier()->copy(v_b, s_b, "Move pointer");

            Register::ValuePtr v_c;
            co_yield Expression::generate(v_c, expr, m_context);

            co_yield m_context->mem()->storeFlat(v_result, v_c, "", 8);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
            GTEST_SKIP() << "Skipping GPU tests for " << GetParam();

        // Only execute the kernels if running on the architecture that the kernel was built for,
        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {
            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            auto d_result = make_shared_device<double>();

            double a = 192.0;
            double b = 12981.0;

            KernelArguments kargs;
            kargs.append("result", d_result.get());
            kargs.append("a", a);
            kargs.append("b", b);
            KernelInvocation invocation;

            executableKernel->executeKernel(kargs, invocation);

            std::vector<double> result(3);
            ASSERT_THAT(hipMemcpy(result.data(), d_result.get(), sizeof(double), hipMemcpyDefault),
                        HasHipSuccess(0));

            EXPECT_EQ(result[0], -b * (a + b));
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_P(ARCH_ExpressionTest, ExpressionFusedAddShift)
    {
        auto v_a
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_b
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::UInt32, 1);

        auto a = v_a->expression();
        auto b = v_b->expression();

        auto expr = Expression::fuse((a + b) << b);

        auto k = m_context->kernel();

        k->setKernelName("ExpressionFusedAddShift");

        k->addArgument(
            {"result", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::WriteOnly});
        k->addArgument({"a", DataType::Int32});
        k->addArgument({"b", DataType::UInt32});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a, s_b;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield v_a->allocate();
            co_yield v_b->allocate();
            co_yield v_c->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

            co_yield m_context->copier()->copy(v_a, s_a, "Move pointer");
            co_yield m_context->copier()->copy(v_b, s_b, "Move pointer");

            co_yield Expression::generate(v_c, expr, m_context);

            co_yield m_context->mem()->storeFlat(v_result, v_c, "", 4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
            GTEST_SKIP() << "Skipping GPU tests for " << GetParam();

        // Only execute the kernels if running on the architecture that the kernel was built for,
        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {
            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            auto d_result = make_shared_device<int>();

            int          a = 12;
            unsigned int b = 2u;

            KernelArguments kargs;
            kargs.append("result", d_result.get());
            kargs.append("a", a);
            kargs.append("b", b);
            KernelInvocation invocation;

            executableKernel->executeKernel(kargs, invocation);

            std::vector<int> result(3);
            ASSERT_THAT(hipMemcpy(result.data(), d_result.get(), sizeof(int), hipMemcpyDefault),
                        HasHipSuccess(0));

            EXPECT_EQ(result[0], (a + b) << b);
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        ARCH_ExpressionTests,
        ARCH_ExpressionTest,
        ::testing::ValuesIn(rocRoller::GPUArchitectureLibrary::getAllSupportedISAs()));

    class GPU_ExpressionTest : public CurrentGPUContextFixture
    {
    };

    TEST_F(GPU_ExpressionTest, MatrixMultiply)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        auto const VGPR = [&](VariableType type, int count) {
            return Register::Value::Placeholder(m_context, Register::Type::Vector, type, count);
        };
        auto const FloatPointer = VariableType(DataType::Float, PointerType::PointerGlobal);

        unsigned int const M      = 32;
        unsigned int const N      = 32;
        unsigned int const K      = 2;
        unsigned int const nbatch = 1;

        auto const L = [](auto x) { return Expression::literal(x); };

        auto kernel = m_context->kernel();

        kernel->setKernelName("MatrixMultiply");

        auto workitems = Expression::literal(64u);
        auto one       = Expression::literal(1u);

        kernel->setWorkgroupSize({64, 1, 1});
        kernel->setWorkitemCount({workitems, one, one});

        kernel->addArgument({"d", FloatPointer, DataDirection::WriteOnly});
        kernel->addArgument({"a", FloatPointer});
        kernel->addArgument({"b", FloatPointer});

        m_context->schedule(kernel->preamble());
        m_context->schedule(kernel->prolog());

        // all matrices stored in column-major order
        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_d, s_a, s_b;
            co_yield m_context->argLoader()->getValue("d", s_d);
            co_yield m_context->argLoader()->getValue("a", s_a);
            co_yield m_context->argLoader()->getValue("b", s_b);

            auto v_a = VGPR(DataType::Float, 1);
            co_yield v_a->allocate();

            auto v_b = VGPR(DataType::Float, 1);
            co_yield v_b->allocate();

            auto thread = m_context->kernel()->workitemIndex()[0];
            // TODO: this gives an error currently
            // co_yield exprInt32->generate(
            //     fastDivision(thread->expression()
            //                      % Expression::literal(m_context->kernel()->wavefront_size()),
            //                  m_context),
            //     m_context);
            // auto lane = exprInt32->getResult();
            auto lane = thread;

            auto A    = v_a->expression();
            auto B    = v_b->expression();
            auto LANE = lane->expression();
            auto SA   = s_a->expression();
            auto SB   = s_b->expression();
            auto SD   = s_d->expression();

            // Load A
            co_yield_(Instruction::Comment("Load A"));
            {
                // todo: incorporate wave number
                auto g = LANE;
                auto b = g * L(4u);

                Register::ValuePtr byteOffset;
                co_yield Expression::generate(byteOffset, fastDivision(b, m_context), m_context);

                Register::ValuePtr ptr;
                co_yield Expression::generate(ptr, SA + byteOffset->expression(), m_context);

                co_yield m_context->mem()->loadFlat(v_a, ptr, "", 4);
            }

            // Load B
            co_yield_(Instruction::Comment("Load B"));
            {
                // todo: incorporate wave number
                auto i = LANE / L(32u);
                auto j = LANE % L(32u);
                auto g = j * L(2u) + i;
                auto b = g * L(4u);

                Register::ValuePtr byteOffset;
                co_yield Expression::generate(byteOffset, fastDivision(b, m_context), m_context);

                Register::ValuePtr ptr;
                co_yield Expression::generate(ptr, SB + byteOffset->expression(), m_context);

                co_yield m_context->mem()->loadFlat(v_b, ptr, "", 4);
            }

            auto expr = std::make_shared<Expression::Expression>(
                Expression::MatrixMultiply(A, B, M, N, K, nbatch));

            Register::ValuePtr v_d;
            co_yield Expression::generate(v_d, expr, m_context);
            co_yield Instruction::Nop(16, "MFMA hazard");

            // Store D
            co_yield_(Instruction::Comment("Store D"));
            {
                for(int a = 0; a < 16; ++a)
                {
                    // todo: incorporate wave number
                    auto i = LANE / L(32u) * L(4u) + L(8 * (a / 4) + a % 4);
                    auto j = LANE % L(32u);
                    auto g = j * L(32u) + i;
                    auto b = g * L(4u);

                    Register::ValuePtr byteOffset;
                    co_yield Expression::generate(
                        byteOffset, fastDivision(b, m_context), m_context);

                    Register::ValuePtr ptr;
                    co_yield Expression::generate(ptr, SD + byteOffset->expression(), m_context);

                    auto val = VGPR(DataType::Float, 1);
                    co_yield m_context->copier()->copy(val,
                                                       v_d->element({a}));
                    co_yield_(Instruction(
                        "s_nop", {Register::Value::Literal(2u)}, {}, {}, "MFMA hazard"));
                    co_yield m_context->mem()->storeFlat(ptr, val, "", 4);
                }
            }
        };

        m_context->schedule(kb());
        m_context->schedule(kernel->postamble());
        m_context->schedule(kernel->amdgpu_metadata());

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
            GTEST_SKIP() << "Skipping GPU tests for " << ARCH_ExpressionTest::GetParam();

        // Only execute the kernels if running on the architecture that the kernel was built for,
        // otherwise just assemble the instructions.
        if(isLocalDevice())
        {
            auto random = RandomGenerator(12679u);

            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            auto a = random.vector<float>(M * K * nbatch, -1.f, 1.f);
            auto b = random.vector<float>(K * N * nbatch, -1.f, 1.f);

            auto d_a = make_shared_device(a);
            auto d_b = make_shared_device(b);
            auto d_d = make_shared_device<float>(M * N * nbatch);

            KernelArguments kargs;
            kargs.append("d", d_d.get());
            kargs.append("a", d_a.get());
            kargs.append("b", d_b.get());
            KernelInvocation invocation;

            invocation.workgroupSize = {64, 1, 1};
            invocation.workitemCount = {64, 1, 1};

            executableKernel->executeKernel(kargs, invocation);

            std::vector<float> d(M * N * nbatch);
            ASSERT_THAT(
                hipMemcpy(d.data(), d_d.get(), M * N * nbatch * sizeof(float), hipMemcpyDefault),
                HasHipSuccess(0));

            std::vector<float> D(M * N * nbatch, 0.f);
            std::vector<float> C(M * N * nbatch, 0.f);
            CPUMM(D, C, a, b, M, N, K, 1.0, 0.0, false);

            double rnorm = relativeNorm(d, D);
            ASSERT_LT(rnorm, 2.e-6);
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }
}
