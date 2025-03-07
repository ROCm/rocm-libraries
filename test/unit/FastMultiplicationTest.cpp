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
#include "Utilities.hpp"
#include <common/TestValues.hpp>

using namespace rocRoller;

namespace FastMultiplicationTest
{
    class FastMultiplicationTest : public GenericContextFixture
    {
    };

    TEST_F(FastMultiplicationTest, MultiplicationByConstantExpressions)
    {
#define ExpectCommutative(lhs, rhs, result)                                \
    EXPECT_EQ(Expression::toString(fastMultiplication(lhs* rhs)), result); \
    EXPECT_EQ(Expression::toString(fastMultiplication(rhs* lhs)), result)

        auto command = std::make_shared<Command>();

        auto aTag = command->allocateTag();
        auto a    = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value));

        auto bTag = command->allocateTag();
        auto b    = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Float, PointerType::Value}, bTag, ArgumentType::Value));

        ExpectCommutative(a, std::make_shared<Expression::Expression>(0), "0:I");

        ExpectCommutative(
            a, std::make_shared<Expression::Expression>(1), "CommandArgument(user_Int32_Value_0)I");

        ExpectCommutative(a, std::make_shared<Expression::Expression>(0u), "0:I");

        ExpectCommutative(a, Expression::literal(0u), "0:I");

        ExpectCommutative(a,
                          std::make_shared<Expression::Expression>(1u),
                          "Convert(CommandArgument(user_Int32_Value_0)I)U32");

        ExpectCommutative(a,
                          std::make_shared<Expression::Expression>(8u),
                          "ShiftL(Convert(CommandArgument(user_Int32_Value_0)I)U32, 3:U32)U32");

        ExpectCommutative(a,
                          std::make_shared<Expression::Expression>(16),
                          "ShiftL(CommandArgument(user_Int32_Value_0)I, 4:U32)I");

        ExpectCommutative(a,
                          Expression::literal(64u),
                          "ShiftL(Convert(CommandArgument(user_Int32_Value_0)I)U32, 6:U32)U32");

        ExpectCommutative(a,
                          std::make_shared<Expression::Expression>(256),
                          "ShiftL(CommandArgument(user_Int32_Value_0)I, 8:U32)I");
#undef ExpectCommutative

        auto expr      = a * std::make_shared<Expression::Expression>(-5);
        auto expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(CommandArgument(user_Int32_Value_0)I, -5:I)I");

        expr      = std::make_shared<Expression::Expression>(-5) * a;
        expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(-5:I, CommandArgument(user_Int32_Value_0)I)I");

        expr      = a * std::make_shared<Expression::Expression>(-4);
        expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(CommandArgument(user_Int32_Value_0)I, -4:I)I");

        expr      = std::make_shared<Expression::Expression>(-4) * a;
        expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(-4:I, CommandArgument(user_Int32_Value_0)I)I");

        expr      = a * std::make_shared<Expression::Expression>(11u);
        expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(CommandArgument(user_Int32_Value_0)I, 11:U32)U32");

        expr      = std::make_shared<Expression::Expression>(11u) * a;
        expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(11:U32, CommandArgument(user_Int32_Value_0)I)U32");

        expr      = b * std::make_shared<Expression::Expression>(3.14);
        expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(CommandArgument(user_Float_Value_1)S, 3.14000:D)D");

        expr      = b * Expression::literal(3.14);
        expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(CommandArgument(user_Float_Value_1)S, 3.14000:D)D");
    }

    TEST_F(FastMultiplicationTest, MultiplicationByArgumentExpressions)
    {
        auto command = std::make_shared<Command>();

        auto aTag = command->allocateTag();
        auto a    = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value));

        auto bTag       = command->allocateTag();
        auto b_unsigned = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::UInt32, PointerType::Value}, bTag, ArgumentType::Value));

        auto expr      = a * b_unsigned;
        auto expr_fast = rocRoller::Expression::fastMultiplication(expr);
        EXPECT_EQ(Expression::toString(expr_fast),
                  "Multiply(CommandArgument(user_Int32_Value_0)I, "
                  "CommandArgument(user_UInt32_Value_1)U32)U32");
    }

    class FastMultiplicationTestByConstantCurrentGPU : public CurrentGPUContextFixture,
                                                       public ::testing::WithParamInterface<int>
    {
    };

    void executeFastMultiplicationByConstant(int multiplier, ContextPtr m_context)
    {
        auto command = std::make_shared<Command>();

        auto resultTag  = command->allocateTag();
        auto result_arg = command->allocateArgument(
            {DataType::Int32, PointerType::PointerGlobal}, resultTag, ArgumentType::Value);
        auto aTag  = command->allocateTag();
        auto a_arg = command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value);

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

            auto v_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               DataType::Raw32,
                                               2,
                                               Register::AllocationOptions::FullyContiguous());

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
            co_yield m_context->mem()->storeGlobal(v_result, v_c, 0, 4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
            = m_context->instructions()->getExecutableKernel();

        auto d_result = make_shared_device<int>();

        CommandKernel commandKernel;
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        auto values = TestValues::int32Values;

        for(int a : values)
        {
            CommandArguments commandArgs = command->createArguments();

            commandArgs.setArgument(resultTag, ArgumentType::Value, d_result.get());
            commandArgs.setArgument(aTag, ArgumentType::Value, a);

            commandKernel.launchKernel(commandArgs.runtimeArguments());

            int result;
            ASSERT_THAT(hipMemcpy(&result, d_result.get(), sizeof(int), hipMemcpyDefault),
                        HasHipSuccess(0));

            EXPECT_EQ(result, a * multiplier) << "A:" << a << " multiplier: " << multiplier;
        }
    }

    void executeFastMultiplicationByConstantInt64(int multiplier, ContextPtr m_context)
    {
        auto command = std::make_shared<Command>();

        auto resultTag  = command->allocateTag();
        auto result_arg = command->allocateArgument(
            {DataType::Int64, PointerType::PointerGlobal}, resultTag, ArgumentType::Value);
        auto aTag  = command->allocateTag();
        auto a_arg = command->allocateArgument(
            {DataType::Int64, PointerType::Value}, aTag, ArgumentType::Value);

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

            auto v_c = Register::Value::Placeholder(m_context,
                                                    Register::Type::Vector,
                                                    DataType::Raw32,
                                                    2,
                                                    Register::AllocationOptions::FullyContiguous());

            co_yield v_result->allocate();

            co_yield(m_context->copier()->copy(v_result, s_result));

            auto a = s_a->expression();

            std::shared_ptr<Expression::Expression> expr;
            expr = fastMultiplication(a * Expression::literal(multiplier));

            Register::ValuePtr s_c;
            co_yield Expression::generate(s_c, expr, m_context);

            co_yield(m_context->copier()->copy(v_c, s_c));
            co_yield m_context->mem()->storeGlobal(v_result, v_c, 0, 8);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
            = m_context->instructions()->getExecutableKernel();

        auto d_result = make_shared_device<int64_t>();

        CommandKernel commandKernel;
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        auto values = TestValues::int64Values;

        for(auto a : values)
        {
            CommandArguments commandArgs = command->createArguments();

            commandArgs.setArgument(resultTag, ArgumentType::Value, d_result.get());
            commandArgs.setArgument(aTag, ArgumentType::Value, a);

            commandKernel.launchKernel(commandArgs.runtimeArguments());

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
