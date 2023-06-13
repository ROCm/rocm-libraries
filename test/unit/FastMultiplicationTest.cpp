
#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "TestValues.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace FastMultiplicationTest
{
    class FastMultiplicationTest : public GenericContextFixture
    {
    };

    TEST_F(FastMultiplicationTest, MultiplicationByConstantExpressions)
    {
        auto ExpectCommutative = [&](std::shared_ptr<Expression::Expression> lhs,
                                     std::shared_ptr<Expression::Expression> rhs,
                                     std::string                             result) {
            EXPECT_EQ(Expression::toString(fastMultiplication(lhs * rhs)), result);
            EXPECT_EQ(Expression::toString(fastMultiplication(rhs * lhs)), result);
        };

        auto command = std::make_shared<Command>();

        auto a = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));

        ExpectCommutative(a, std::make_shared<Expression::Expression>(0), "0i");

        ExpectCommutative(
            a, std::make_shared<Expression::Expression>(1), "CommandArgument(user_Int32_Value_0)");

        ExpectCommutative(a, std::make_shared<Expression::Expression>(0u), "0i");

        ExpectCommutative(a, Expression::literal(0u), "0i");

        ExpectCommutative(
            a, std::make_shared<Expression::Expression>(1u), "CommandArgument(user_Int32_Value_0)");

        ExpectCommutative(a,
                          std::make_shared<Expression::Expression>(8u),
                          "ShiftL(CommandArgument(user_Int32_Value_0), 3j)");

        ExpectCommutative(a,
                          std::make_shared<Expression::Expression>(16),
                          "ShiftL(CommandArgument(user_Int32_Value_0), 4j)");

        ExpectCommutative(
            a, Expression::literal(64u), "ShiftL(CommandArgument(user_Int32_Value_0), 6j)");

        ExpectCommutative(a,
                          std::make_shared<Expression::Expression>(256u),
                          "ShiftL(CommandArgument(user_Int32_Value_0), 8j)");

        auto expr      = a * std::make_shared<Expression::Expression>(-5);
        auto expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(CommandArgument(user_Int32_Value_0), -5i)");

        expr      = std::make_shared<Expression::Expression>(-5) * a;
        expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(-5i, CommandArgument(user_Int32_Value_0))");

        expr      = a * std::make_shared<Expression::Expression>(-4);
        expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(CommandArgument(user_Int32_Value_0), -4i)");

        expr      = std::make_shared<Expression::Expression>(-4) * a;
        expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(-4i, CommandArgument(user_Int32_Value_0))");

        expr      = a * std::make_shared<Expression::Expression>(11u);
        expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(CommandArgument(user_Int32_Value_0), 11j)");

        expr      = std::make_shared<Expression::Expression>(11u) * a;
        expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(11j, CommandArgument(user_Int32_Value_0))");

        expr      = a * std::make_shared<Expression::Expression>(3.14);
        expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(CommandArgument(user_Int32_Value_0), 3.14000d)");

        expr      = a * Expression::literal(3.14);
        expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(CommandArgument(user_Int32_Value_0), 3.14000d)");
    }

    TEST_F(FastMultiplicationTest, MultiplicationByArgumentExpressions)
    {
        auto command = std::make_shared<Command>();

        auto a = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));

        auto b_unsigned = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::UInt32, PointerType::Value}));

        auto expr      = a * b_unsigned;
        auto expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(
            Expression::toString(expr_fast),
            "Multiply(CommandArgument(user_Int32_Value_0), CommandArgument(user_UInt32_Value_1))");
    }

    class FastMultiplicationTestByConstantCurrentGPU : public CurrentGPUContextFixture,
                                                       public ::testing::WithParamInterface<int>
    {
    };

    void executeFastMultiplicationByConstant(int multiplier, ContextPtr m_context)
    {
        auto command = std::make_shared<Command>();

        auto result_arg = command->allocateArgument({DataType::Int32, PointerType::PointerGlobal});
        auto a_arg      = command->allocateArgument({DataType::Int32, PointerType::Value});

        auto result_exp = std::make_shared<Expression::Expression>(result_arg);
        auto a_exp      = std::make_shared<Expression::Expression>(a_arg);

        auto k = m_context->kernel();

        k->addArgument({"result",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"a", DataType::Int32, DataDirection::ReadOnly, a_exp});

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield v_result->allocate();

            co_yield(m_context->copier()->copy(v_result, s_result));

            auto a = s_a->expression();

            std::shared_ptr<Expression::Expression> expr;
            expr = fastMultiplication(a * Expression::literal(multiplier));

            Register::ValuePtr s_c;
            co_yield Expression::generate(s_c, expr, m_context);

            co_yield(m_context->copier()->copy(v_c, s_c));
            co_yield m_context->mem()->storeFlat(v_result, v_c, 0, 4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
            = m_context->instructions()->getExecutableKernel();

        auto d_result = make_shared_device<int>();

        CommandKernel commandKernel(m_context);

        auto values = TestValues::int32Values;

        for(int a : values)
        {
            KernelArguments runtimeArgs;
            runtimeArgs.append("result", d_result.get());
            runtimeArgs.append("a", a);

            commandKernel.launchKernel(runtimeArgs.runtimeArguments());

            int result;
            ASSERT_THAT(hipMemcpy(&result, d_result.get(), sizeof(int), hipMemcpyDefault),
                        HasHipSuccess(0));

            EXPECT_EQ(result, a * multiplier) << "A:" << a << " multiplier: " << multiplier;
        }
    }

    void executeFastMultiplicationByConstantInt64(int multiplier, ContextPtr m_context)
    {
        auto command = std::make_shared<Command>();

        auto result_arg = command->allocateArgument({DataType::Int64, PointerType::PointerGlobal});
        auto a_arg      = command->allocateArgument({DataType::Int64, PointerType::Value});

        auto result_exp = std::make_shared<Expression::Expression>(result_arg);
        auto a_exp      = std::make_shared<Expression::Expression>(a_arg);

        auto k = m_context->kernel();

        k->addArgument({"result",
                        {DataType::Int64, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument({"a", DataType::Int64, DataDirection::ReadOnly, a_exp});

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int64, 1);

            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            co_yield v_result->allocate();

            co_yield(m_context->copier()->copy(v_result, s_result));

            auto a = s_a->expression();

            std::shared_ptr<Expression::Expression> expr;
            expr = fastMultiplication(a * Expression::literal(multiplier));

            Register::ValuePtr s_c;
            co_yield Expression::generate(s_c, expr, m_context);

            co_yield(m_context->copier()->copy(v_c, s_c));
            co_yield m_context->mem()->storeFlat(v_result, v_c, 0, 8);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
            = m_context->instructions()->getExecutableKernel();

        auto d_result = make_shared_device<int64_t>();

        CommandKernel commandKernel(m_context);

        auto values = TestValues::int64Values;

        for(auto a : values)
        {
            KernelArguments runtimeArgs;
            runtimeArgs.append("result", d_result.get());
            runtimeArgs.append("a", a);

            commandKernel.launchKernel(runtimeArgs.runtimeArguments());

            int64_t result;
            ASSERT_THAT(hipMemcpy(&result, d_result.get(), sizeof(int64_t), hipMemcpyDefault),
                        HasHipSuccess(0));

            EXPECT_EQ(result, a * multiplier) << "A:" << a << " multiplier: " << multiplier;
        }
    }

    TEST_P(FastMultiplicationTestByConstantCurrentGPU, GPU_FastMultiplicationByConstant)
    {
        executeFastMultiplicationByConstant(GetParam(), m_context);
    }

    TEST_P(FastMultiplicationTestByConstantCurrentGPU, GPU_FastMultiplicationByConstantInt64)
    {
        executeFastMultiplicationByConstantInt64(GetParam(), m_context);
    }

    INSTANTIATE_TEST_SUITE_P(
        FastMultiplicationTestByConstantCurrentGPUs,
        FastMultiplicationTestByConstantCurrentGPU,
        ::testing::Values(-12345, -128, -3, -2, -1, 0, 1, 2, 3, 5, 8, 128, 12345));

}
