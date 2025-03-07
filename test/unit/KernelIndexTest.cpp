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
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GPUContextFixture.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace rocRollerTest
{

    struct KernelIndexParam
    {
        unsigned int                dimensions;
        unsigned int                indexDimension;
        std::array<unsigned int, 3> arraySize;
    };

    std::ostream& operator<<(std::ostream& st, KernelIndexParam const& params)
    {
        return st << params.dimensions << "D," << params.indexDimension << ",["
                  << params.arraySize[0] << "," << params.arraySize[1] << "," << params.arraySize[2]
                  << "]";
    }

    // Parameters: Axis of index to use, dimensions of the array.
    class GPU_KernelIndexTest : public CurrentGPUContextFixture,
                                public ::testing::WithParamInterface<KernelIndexParam>
    {
    };

    TEST_P(GPU_KernelIndexTest, WorkitemIndices)
    {
        auto params = GetParam();

        ASSERT_LE(params.dimensions, 3);
        ASSERT_LT(params.indexDimension, params.dimensions);

        if(params.dimensions < 3)
        {
            ASSERT_EQ(params.arraySize[2], 1);
        }

        if(params.dimensions < 2)
        {
            ASSERT_EQ(params.arraySize[1], 1);
        }

        unsigned int arrayElements
            = params.arraySize[0] * params.arraySize[1] * params.arraySize[2];
        unsigned int allocSize = RoundUpToMultiple(arrayElements, 256u);

        auto command = std::make_shared<Command>();

        VariableType intPtr{DataType::Int32, PointerType::PointerGlobal};
        VariableType intVal{DataType::Int32, PointerType::Value};
        VariableType uintVal{DataType::UInt32, PointerType::Value};

        auto ptrTag    = command->allocateTag();
        auto ptr_arg   = command->allocateArgument(intPtr, ptrTag, ArgumentType::Value);
        auto valTag    = command->allocateTag();
        auto val_arg   = command->allocateArgument(intVal, valTag, ArgumentType::Value);
        auto size0Tag  = command->allocateTag();
        auto size0_arg = command->allocateArgument(uintVal, size0Tag, ArgumentType::Size);
        auto size1Tag  = command->allocateTag();
        auto size1_arg = command->allocateArgument(uintVal, size1Tag, ArgumentType::Size);
        auto size2Tag  = command->allocateTag();
        auto size2_arg = command->allocateArgument(uintVal, size2Tag, ArgumentType::Size);

        auto ptr_exp   = std::make_shared<Expression::Expression>(ptr_arg);
        auto val_exp   = std::make_shared<Expression::Expression>(val_arg);
        auto size0_exp = std::make_shared<Expression::Expression>(size0_arg);
        auto size1_exp = std::make_shared<Expression::Expression>(size1_arg);
        auto size2_exp = std::make_shared<Expression::Expression>(size2_arg);

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        auto k = m_context->kernel();

        k->setKernelDimensions(params.dimensions);

        k->addArgument({"ptr",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        ptr_exp});
        k->addArgument({"val", {DataType::Int32}, DataDirection::ReadOnly, val_exp});

        k->setWorkgroupSize(params.arraySize);
        k->setWorkitemCount({size0_exp, size1_exp, size2_exp});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            co_yield Instruction::Comment("ptr[idx] = threadIdx[indexDimension] * value;");

            Register::ValuePtr s_ptr, s_value;
            co_yield m_context->argLoader()->getValue("ptr", s_ptr);
            co_yield m_context->argLoader()->getValue("val", s_value);

            std::array<Register::ValuePtr, 3> workitemIndex = k->workitemIndex();

            auto valueIndex = workitemIndex[params.indexDimension];

            Register::ValuePtr v_value;
            Register::ValuePtr v_offset;

            {
                std::array<Expression::ExpressionPtr, 3> index;
                for(int i = 0; i < 3; i++)
                    index[i] = std::make_shared<Expression::Expression>(workitemIndex[i]);

                std::array<Expression::ExpressionPtr, 3> sizes;
                for(int i = 0; i < 3; i++)
                    sizes[i] = Expression::literal(params.arraySize[i]);

                auto elementSize = std::make_shared<Expression::Expression>(
                    Register::Value::Literal<int32_t>(sizeof(unsigned int)));

                Expression::ExpressionPtr offset;

                offset = index[0];
                if(params.dimensions > 1)
                    offset = offset + index[1] * sizes[0];
                if(params.dimensions > 2)
                    offset = offset + index[2] * sizes[0] * sizes[1];

                offset = offset * elementSize;

                co_yield Expression::generate(v_offset, offset, m_context);
            }

            {
                auto valueIndexExp = std::make_shared<Expression::Expression>(valueIndex);

                auto exp = valueIndexExp * std::make_shared<Expression::Expression>(s_value);
                co_yield Expression::generate(v_value, exp, m_context);
            }

            co_yield_(Instruction(
                "global_store_dword", {}, {v_offset, v_value, s_ptr}, {}, "Store value"));
        };

        m_context->schedule(kb());

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel;
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        auto ptr = make_shared_device<int>(allocSize);
        int  val = 5;

        ASSERT_THAT(hipMemset(ptr.get(), 0, allocSize * sizeof(int)), HasHipSuccess(0));

        CommandArguments commandArgs = command->createArguments();

        commandArgs.setArgument(ptrTag, ArgumentType::Value, ptr.get());
        commandArgs.setArgument(valTag, ArgumentType::Value, val);
        commandArgs.setArgument(size0Tag, ArgumentType::Size, params.arraySize[0]);
        commandArgs.setArgument(size1Tag, ArgumentType::Size, params.arraySize[1]);
        commandArgs.setArgument(size2Tag, ArgumentType::Size, params.arraySize[2]);

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        std::vector<unsigned int> hostBuffer(allocSize);
        ASSERT_THAT(
            hipMemcpy(hostBuffer.data(), ptr.get(), allocSize * sizeof(int), hipMemcpyDefault),
            HasHipSuccess(0));

        for(unsigned int i = 0; i < allocSize; i++)
        {
            if(i >= arrayElements)
            {
                EXPECT_EQ(hostBuffer[i], 0) << i;
            }
            else
            {
                std::array<unsigned int, 3> indices{
                    i % params.arraySize[0], i / params.arraySize[0], 1};
                if(params.dimensions > 2)
                {
                    indices[2] = indices[1] / params.arraySize[1];
                    indices[1] = indices[1] % params.arraySize[1];
                }
                EXPECT_EQ(hostBuffer[i], indices[params.indexDimension] * val) << i;
            }
        }

        if(HasFailure())
        {
            std::ostringstream msg;
            for(unsigned int i = 0; i < params.arraySize[2]; i++)
            {
                for(unsigned int j = 0; j < params.arraySize[1]; j++)
                {
                    for(unsigned int k = 0; k < params.arraySize[0]; k++)
                    {
                        auto idx = k + (j * params.arraySize[0])
                                   + (i * params.arraySize[0] * params.arraySize[1]);
                        msg << hostBuffer[idx] << " ";
                    }
                    msg << std::endl;
                }
            }
            FAIL() << msg.str();
        }
    }

    TEST_P(GPU_KernelIndexTest, WorkgroupIndices)
    {
        auto params = GetParam();

        ASSERT_LE(params.dimensions, 3);
        ASSERT_LT(params.indexDimension, params.dimensions);

        if(params.dimensions < 3)
        {
            ASSERT_EQ(params.arraySize[2], 1);
        }

        if(params.dimensions < 2)
        {
            ASSERT_EQ(params.arraySize[1], 1);
        }

        unsigned int arrayElements
            = params.arraySize[0] * params.arraySize[1] * params.arraySize[2];
        unsigned int allocSize = RoundUpToMultiple(arrayElements, 256u);

        auto command = std::make_shared<Command>();

        VariableType intPtr{DataType::Int32, PointerType::PointerGlobal};
        VariableType intVal{DataType::Int32, PointerType::Value};
        VariableType uintVal{DataType::UInt32, PointerType::Value};

        auto ptrTag    = command->allocateTag();
        auto ptr_arg   = command->allocateArgument(intPtr, ptrTag, ArgumentType::Value);
        auto valTag    = command->allocateTag();
        auto val_arg   = command->allocateArgument(intVal, valTag, ArgumentType::Value);
        auto size0Tag  = command->allocateTag();
        auto size0_arg = command->allocateArgument(uintVal, size0Tag, ArgumentType::Size);
        auto size1Tag  = command->allocateTag();
        auto size1_arg = command->allocateArgument(uintVal, size1Tag, ArgumentType::Size);
        auto size2Tag  = command->allocateTag();
        auto size2_arg = command->allocateArgument(uintVal, size2Tag, ArgumentType::Size);

        auto ptr_exp   = std::make_shared<Expression::Expression>(ptr_arg);
        auto val_exp   = std::make_shared<Expression::Expression>(val_arg);
        auto size0_exp = std::make_shared<Expression::Expression>(size0_arg);
        auto size1_exp = std::make_shared<Expression::Expression>(size1_arg);
        auto size2_exp = std::make_shared<Expression::Expression>(size2_arg);

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        auto k = m_context->kernel();

        k->setKernelDimensions(params.dimensions);

        k->addArgument({"ptr",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        ptr_exp});
        k->addArgument({"val", {DataType::Int32}, DataDirection::ReadOnly, val_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({size0_exp, size1_exp, size2_exp});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_ptr, s_value;
            co_yield m_context->argLoader()->getValue("ptr", s_ptr);
            co_yield m_context->argLoader()->getValue("val", s_value);

            std::array<Register::ValuePtr, 3> workgroupIndex = k->workgroupIndex();
            for(unsigned int i = 0; i < params.dimensions; i++)
            {
                auto vgpr = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);
                co_yield m_context->copier()->copy(vgpr, workgroupIndex[i], "");
                workgroupIndex[i] = vgpr;
            }

            auto valueIndex = workgroupIndex[params.indexDimension];

            Register::ValuePtr v_value;
            Register::ValuePtr v_offset;

            {
                std::array<Expression::ExpressionPtr, 3> index;
                for(unsigned int i = 0; i < params.dimensions; i++)
                    index[i] = workgroupIndex[i]->expression();

                std::array<Expression::ExpressionPtr, 3> sizes;
                for(unsigned int i = 0; i < params.dimensions; i++)
                    sizes[i] = Expression::literal(params.arraySize[i]);

                auto elementSize = Expression::literal<int32_t>(sizeof(unsigned int));

                Expression::ExpressionPtr offset;

                offset = index[0];
                if(params.dimensions > 1)
                    offset = offset + index[1] * sizes[0];
                if(params.dimensions > 2)
                    offset = offset + index[2] * sizes[0] * sizes[1];

                offset = offset * elementSize;

                co_yield Expression::generate(v_offset, offset, m_context);
            }

            {
                auto valueIndexExp = std::make_shared<Expression::Expression>(valueIndex);

                auto exp = valueIndexExp * std::make_shared<Expression::Expression>(s_value);
                co_yield Expression::generate(v_value, exp, m_context);
            }

            co_yield_(Instruction(
                "global_store_dword", {}, {v_offset, v_value, s_ptr}, {}, "Store value"));
        };

        m_context->schedule(kb());

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel;
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        auto ptr = make_shared_device<int>(allocSize);
        int  val = 5;

        ASSERT_THAT(hipMemset(ptr.get(), 0, allocSize * sizeof(int)), HasHipSuccess(0));

        CommandArguments commandArgs = command->createArguments();

        commandArgs.setArgument(ptrTag, ArgumentType::Value, ptr.get());
        commandArgs.setArgument(valTag, ArgumentType::Value, val);
        commandArgs.setArgument(size0Tag, ArgumentType::Size, params.arraySize[0]);
        commandArgs.setArgument(size1Tag, ArgumentType::Size, params.arraySize[1]);
        commandArgs.setArgument(size2Tag, ArgumentType::Size, params.arraySize[2]);

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        std::vector<unsigned int> hostBuffer(allocSize);
        ASSERT_THAT(
            hipMemcpy(hostBuffer.data(), ptr.get(), allocSize * sizeof(int), hipMemcpyDefault),
            HasHipSuccess(0));

        for(unsigned int i = 0; i < allocSize; i++)
        {
            if(i >= arrayElements)
            {
                EXPECT_EQ(hostBuffer[i], 0) << i;
            }
            else
            {
                std::array<unsigned int, 3> indices{
                    i % params.arraySize[0], i / params.arraySize[0], 1};
                if(params.dimensions > 2)
                {
                    indices[2] = indices[1] / params.arraySize[1];
                    indices[1] = indices[1] % params.arraySize[1];
                }
                EXPECT_EQ(hostBuffer[i], indices[params.indexDimension] * val) << i;
            }
        }

        if(HasFailure())
        {
            std::ostringstream msg;
            for(unsigned int i = 0; i < params.arraySize[2]; i++)
            {
                for(unsigned int j = 0; j < params.arraySize[1]; j++)
                {
                    for(unsigned int k = 0; k < params.arraySize[0]; k++)
                    {
                        auto idx = k + (j * params.arraySize[0])
                                   + (i * params.arraySize[0] * params.arraySize[1]);
                        msg << hostBuffer[idx] << " ";
                    }
                    msg << std::endl;
                }
            }
            FAIL() << msg.str();
        }
    }

    TEST_P(GPU_KernelIndexTest, BothIndices)
    {
        auto params = GetParam();

        ASSERT_LE(params.dimensions, 3);
        ASSERT_LT(params.indexDimension, params.dimensions);

        if(params.dimensions < 3)
        {
            ASSERT_EQ(params.arraySize[2], 1);
        }

        if(params.dimensions < 2)
        {
            ASSERT_EQ(params.arraySize[1], 1);
        }

        unsigned int arrayElements
            = params.arraySize[0] * params.arraySize[1] * params.arraySize[2];
        unsigned int allocSize = RoundUpToMultiple(arrayElements, 256u);

        std::array<unsigned int, 3> workgroupSize{1, 1, 1};
        for(unsigned int i = 0; i < params.dimensions; i++)
            workgroupSize[i] = std::max(RoundUpToMultiple(params.arraySize[i] / 8u, 8u), 1u);

        auto command = std::make_shared<Command>();

        VariableType intPtr{DataType::Int32, PointerType::PointerGlobal};
        VariableType intVal{DataType::Int32, PointerType::Value};
        VariableType uintVal{DataType::UInt32, PointerType::Value};

        auto ptrTag    = command->allocateTag();
        auto ptr_arg   = command->allocateArgument(intPtr, ptrTag, ArgumentType::Value);
        auto valTag    = command->allocateTag();
        auto val_arg   = command->allocateArgument(intVal, valTag, ArgumentType::Value);
        auto size0Tag  = command->allocateTag();
        auto size0_arg = command->allocateArgument(uintVal, size0Tag, ArgumentType::Size);
        auto size1Tag  = command->allocateTag();
        auto size1_arg = command->allocateArgument(uintVal, size1Tag, ArgumentType::Size);
        auto size2Tag  = command->allocateTag();
        auto size2_arg = command->allocateArgument(uintVal, size2Tag, ArgumentType::Size);

        auto ptr_exp   = std::make_shared<Expression::Expression>(ptr_arg);
        auto val_exp   = std::make_shared<Expression::Expression>(val_arg);
        auto size0_exp = std::make_shared<Expression::Expression>(size0_arg);
        auto size1_exp = std::make_shared<Expression::Expression>(size1_arg);
        auto size2_exp = std::make_shared<Expression::Expression>(size2_arg);

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        auto k = m_context->kernel();

        k->setKernelDimensions(params.dimensions);

        k->addArgument({"ptr",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        ptr_exp});
        k->addArgument({"val", {DataType::Int32}, DataDirection::ReadOnly, val_exp});

        k->setWorkgroupSize(workgroupSize);
        k->setWorkitemCount({size0_exp, size1_exp, size2_exp});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_ptr, s_value;
            co_yield m_context->argLoader()->getValue("ptr", s_ptr);
            co_yield m_context->argLoader()->getValue("val", s_value);

            auto workgroupIndex = k->workgroupIndex();
            for(unsigned int i = 0; i < params.dimensions; i++)
            {
                auto vgpr = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);
                co_yield m_context->copier()->copy(vgpr, workgroupIndex[i], "");
                workgroupIndex[i] = vgpr;
            }

            auto workitemIndex = k->workitemIndex();

            std::array<Register::ValuePtr, 3> arrayIndex;

            for(unsigned int i = 0; i < params.dimensions; i++)
            {
                auto dimExp
                    = workgroupIndex[i]->expression() * Expression::literal(workgroupSize[i])
                      + workitemIndex[i]->expression();

                co_yield Expression::generate(arrayIndex[i], dimExp, m_context);
            }

            auto valueIndex = arrayIndex[params.indexDimension];

            Register::ValuePtr v_value;
            Register::ValuePtr v_offset;

            {
                std::array<Expression::ExpressionPtr, 3> index;
                for(unsigned int i = 0; i < params.dimensions; i++)
                    index[i] = arrayIndex[i]->expression();

                std::array<Expression::ExpressionPtr, 3> sizes;
                for(unsigned int i = 0; i < params.dimensions; i++)
                    sizes[i] = Expression::literal(params.arraySize[i]);

                auto elementSize = Expression::literal<int32_t>(sizeof(unsigned int));

                Expression::ExpressionPtr offset;

                offset = index[0];
                if(params.dimensions > 1)
                    offset = offset + index[1] * sizes[0];
                if(params.dimensions > 2)
                    offset = offset + index[2] * sizes[0] * sizes[1];

                offset = offset * elementSize;

                co_yield Instruction::Comment("Offset Calculation");
                co_yield Expression::generate(v_offset, offset, m_context);
            }

            {
                auto exp = valueIndex->expression() * s_value->expression();
                co_yield Instruction::Comment("Value Calculation");
                co_yield Expression::generate(v_value, exp, m_context);
            }

            co_yield_(Instruction(
                "global_store_dword", {}, {v_offset, v_value, s_ptr}, {}, "Store value"));
        };

        m_context->schedule(kb());

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel;
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        auto ptr = make_shared_device<int>(allocSize);
        int  val = 5;

        ASSERT_THAT(hipMemset(ptr.get(), 0, allocSize * sizeof(int)), HasHipSuccess(0));

        CommandArguments commandArgs = command->createArguments();

        commandArgs.setArgument(ptrTag, ArgumentType::Value, ptr.get());
        commandArgs.setArgument(valTag, ArgumentType::Value, val);
        commandArgs.setArgument(size0Tag, ArgumentType::Size, params.arraySize[0]);
        commandArgs.setArgument(size1Tag, ArgumentType::Size, params.arraySize[1]);
        commandArgs.setArgument(size2Tag, ArgumentType::Size, params.arraySize[2]);

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        std::vector<unsigned int> hostBuffer(allocSize);
        ASSERT_THAT(
            hipMemcpy(hostBuffer.data(), ptr.get(), allocSize * sizeof(int), hipMemcpyDefault),
            HasHipSuccess(0));

        for(unsigned int i = 0; i < allocSize; i++)
        {
            if(i >= arrayElements)
            {
                EXPECT_EQ(hostBuffer[i], 0) << i;
            }
            else
            {
                std::array<unsigned int, 3> indices{
                    i % params.arraySize[0], i / params.arraySize[0], 1};
                if(params.dimensions > 2)
                {
                    indices[2] = indices[1] / params.arraySize[1];
                    indices[1] = indices[1] % params.arraySize[1];
                }
                EXPECT_EQ(hostBuffer[i], indices[params.indexDimension] * val) << i;
            }
        }

        if(HasFailure())
        {
            std::ostringstream msg;
            for(unsigned int i = 0; i < params.arraySize[2]; i++)
            {
                for(unsigned int j = 0; j < params.arraySize[1]; j++)
                {
                    for(unsigned int k = 0; k < params.arraySize[0]; k++)
                    {
                        auto idx = k + (j * params.arraySize[0])
                                   + (i * params.arraySize[0] * params.arraySize[1]);
                        msg << hostBuffer[idx] << " ";
                    }
                    msg << std::endl;
                }
            }
            FAIL() << msg.str();
        }
    }

    std::vector<KernelIndexParam> KernelIndexParams()
    {
        std::vector<KernelIndexParam> rv;

        // 1D
        rv.push_back({1, 0, {64, 1, 1}});
        rv.push_back({1, 0, {16, 1, 1}});
        rv.push_back({1, 0, {128, 1, 1}});
        rv.push_back({1, 0, {129, 1, 1}});
        rv.push_back({1, 0, {256, 1, 1}});

        // 2D
        {
            std::vector<std::array<unsigned int, 3>> sizes{
                {16, 4, 1}, {8, 8, 1}, {64, 4, 1}, {32, 8, 1}, {16, 16, 1}, {32, 32, 1}};

            for(auto const& size : sizes)
            {
                rv.push_back({2, 0, size});
                rv.push_back({2, 1, size});
            }
        }

        // 3D
        {
            std::vector<std::array<unsigned int, 3>> sizes{
                {8, 4, 2}, {2, 4, 8}, {8, 4, 8}, {32, 8, 4}, {16, 16, 4}, {1, 32, 32}};

            for(auto const& size : sizes)
            {
                rv.push_back({3, 0, size});
                rv.push_back({3, 1, size});
                rv.push_back({3, 2, size});
            }
        }

        return rv;
    }

    INSTANTIATE_TEST_SUITE_P(GPU_KernelIndexTest,
                             GPU_KernelIndexTest,
                             ::testing::ValuesIn(KernelIndexParams()));

}
