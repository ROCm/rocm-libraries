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

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GPUContextFixture.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class GPU_LabelTest : public CurrentGPUContextFixture,
                          public ::testing::WithParamInterface<bool>
    {
    };

    /**
      * This test is making sure that a label register can be used to insert a label and then branch to that label.
      **/
    TEST_P(GPU_LabelTest, LabelBranchTest)
    {
        bool useVCC = GetParam();
        ASSERT_EQ(true, isLocalDevice());

        auto command = std::make_shared<Command>();

        VariableType floatPtr{DataType::Float, PointerType::PointerGlobal};
        VariableType floatVal{DataType::Float, PointerType::Value};
        VariableType uintVal{DataType::UInt32, PointerType::Value};

        auto ptrTag   = command->allocateTag();
        auto ptr_arg  = command->allocateArgument(floatPtr, ptrTag, ArgumentType::Value);
        auto valTag   = command->allocateTag();
        auto val_arg  = command->allocateArgument(floatVal, valTag, ArgumentType::Value);
        auto sizeTag  = command->allocateTag();
        auto size_arg = command->allocateArgument(uintVal, sizeTag, ArgumentType::Limit);

        auto ptr_exp  = std::make_shared<Expression::Expression>(ptr_arg);
        auto val_exp  = std::make_shared<Expression::Expression>(val_arg);
        auto size_exp = std::make_shared<Expression::Expression>(size_arg);

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        auto k = m_context->kernel();

        k->setKernelDimensions(1);

        k->addArgument({"ptr",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        ptr_exp});
        k->addArgument({"val", {DataType::Float}, DataDirection::ReadOnly, val_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({size_exp, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_ptr, s_value;
            co_yield m_context->argLoader()->getValue("ptr", s_ptr);
            co_yield m_context->argLoader()->getValue("val", s_value);

            auto v_ptr   = Register::Value::Placeholder(m_context,
                                                      Register::Type::Vector,
                                                      {DataType::Float, PointerType::PointerGlobal},
                                                      1);
            auto v_value = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);
            auto v_target = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Float, 1);

            co_yield v_ptr->allocate();

            co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");

            co_yield v_value->allocate();

            co_yield m_context->copier()->copy(v_value, s_value, "Move value");

            //===========Begin Core Test Block======================================
            // Here we will double the input value until it is greater than or equal to 100.f.
            co_yield m_context->copier()->copy(
                v_target, Register::Value::Literal(100.f), "Move value");

            auto loop_start = Register::Value::Label("main_loop"); // Using a label register.
            co_yield Instruction::Label(loop_start); // Inserting the label.

            Register::ValuePtr s_condition;
            if(useVCC)
                s_condition = m_context->getVCC();
            else
                s_condition = Register::Value::WavefrontPlaceholder(m_context);

            // Double the input value.
            co_yield Expression::generate(
                v_value, v_value->expression() + v_value->expression(), m_context);
            // Compare against the stop value.
            co_yield Expression::generate(
                s_condition, v_value->expression() < v_target->expression(), m_context);

            co_yield m_context->brancher()->branchIfNonZero(
                loop_start, s_condition, "//Conditionally branching to the label register.");
            //===========End Core Test Block======================================

            co_yield m_context->mem()->storeGlobal(v_ptr, v_value, 0, 4);
        };

        m_context->schedule(kb());

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel;
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        auto         ptr  = make_shared_device<float>();
        float        val  = 6.0f;
        unsigned int size = 1;

        ASSERT_THAT(hipMemset(ptr.get(), 0, sizeof(float)), HasHipSuccess(0));

        CommandArguments commandArgs = command->createArguments();

        commandArgs.setArgument(ptrTag, ArgumentType::Value, ptr.get());
        commandArgs.setArgument(valTag, ArgumentType::Value, val);
        commandArgs.setArgument(sizeTag, ArgumentType::Limit, size);

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        float resultValue = 0.0f;
        ASSERT_THAT(hipMemcpy(&resultValue, ptr.get(), sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        EXPECT_GT(resultValue, 100.0f);
    }

    INSTANTIATE_TEST_SUITE_P(GPU_LabelTests, GPU_LabelTest, ::testing::Values(false, true));

}
