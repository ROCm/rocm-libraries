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

#include <cmath>

using namespace rocRoller;

namespace LowerExponentialTest
{
    class LowerExponentialTest : public GenericContextFixture
    {
    };

    TEST_F(LowerExponentialTest, ExponentialByArgumentExpressions)
    {
        auto ExpectCommutative
            = [&](std::shared_ptr<Expression::Expression> arg, std::string result) {
                  EXPECT_EQ(Expression::toString(lowerExponential(exp(arg))), result);
              };

        auto command = std::make_shared<Command>();
        auto aTag    = command->allocateTag();
        auto a       = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Float, PointerType::Value}, aTag, ArgumentType::Value));

        ExpectCommutative(
            a, "Exponential2(Multiply(1.44270:S, CommandArgument(user_Float_Value_0)S)S)S");
    }

    class ExecuteLowerExponentialCurrentGPU : public CurrentGPUContextFixture
    {
    public:
        void executeLowerExponentialTest(float a)
        {

            auto command = std::make_shared<Command>();

            auto resultTag  = command->allocateTag();
            auto result_arg = command->allocateArgument(
                {DataType::Float, PointerType::PointerGlobal}, resultTag, ArgumentType::Value);
            auto aTag  = command->allocateTag();
            auto a_arg = command->allocateArgument(
                {DataType::Float, PointerType::Value}, aTag, ArgumentType::Value);

            auto result_exp = std::make_shared<Expression::Expression>(result_arg);
            auto a_exp      = std::make_shared<Expression::Expression>(a_arg);

            auto k = m_context->kernel();

            k->addArgument({"result",
                            {DataType::Float, PointerType::PointerGlobal},
                            DataDirection::WriteOnly,
                            result_exp});

            k->addArgument({"a", DataType::Float, DataDirection::ReadOnly, a_exp});

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
                                                   {DataType::Float, PointerType::PointerGlobal},
                                                   1);

                auto v_a = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Float, 1);
                auto v_c = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Float, 1);

                co_yield m_context->copier()->copy(v_a, s_a, "Move s_a pointer to v_a");

                co_yield v_result->allocate();

                co_yield m_context->copier()->copy(v_result, s_result, "Move pointer v_result");

                auto a = v_a->expression();

                std::shared_ptr<Expression::Expression> expr;
                expr = exp(a);

                Register::ValuePtr s_c;
                co_yield Expression::generate(s_c, expr, m_context);

                co_yield m_context->copier()->copy(v_c, s_c, "Move result to vgpr v_c to store.");
                co_yield m_context->mem()->storeGlobal(
                    v_result, v_c, 0, v_c->variableType().getElementSize());
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());

            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            auto d_result = make_shared_device<float>();

            CommandKernel commandKernel;
            commandKernel.setContext(m_context);
            commandKernel.generateKernel();

            CommandArguments commandArgs = command->createArguments();

            commandArgs.setArgument(resultTag, ArgumentType::Value, d_result.get());
            commandArgs.setArgument(aTag, ArgumentType::Value, a);

            commandKernel.launchKernel(commandArgs.runtimeArguments());

            float result;
            ASSERT_THAT(hipMemcpy(&result, d_result.get(), sizeof(float), hipMemcpyDefault),
                        HasHipSuccess(0));

            float exp_a = std::exp(a);
            if((exp_a == 0) || (exp_a == INFINITY))
            {
                EXPECT_EQ(result, exp_a)
                    << "a: " << a << ", result: " << result << ", exp(a): " << exp_a;
                ;
            }
            else
            {
                EXPECT_LT(std::abs((result - exp_a) / exp_a), 0.000255)
                    << "a: " << a << ", result: " << result << ", exp(a): " << exp_a
                    << "diff: " << std::abs((result - exp_a) / exp_a);
                ;
            }
        }
    };

    class LowerExponentialTestCurrentGPU : public ExecuteLowerExponentialCurrentGPU,
                                           public ::testing::WithParamInterface<float>
    {
    };

    TEST_P(LowerExponentialTestCurrentGPU, GPU_LowerExponentialTest)
    {
        executeLowerExponentialTest(GetParam());
    }

    INSTANTIATE_TEST_SUITE_P(LowerExponentialTestCurrentGPUs,
                             LowerExponentialTestCurrentGPU,
                             ::testing::ValuesIn(TestValues::floatValues));
}
