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

#include <memory>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/BufferInstructionOptions.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "../GPUContextFixture.hpp"
#include "../Utilities.hpp"

using namespace rocRoller;

namespace ConvertInstructionsTest
{
    struct ConvertInstructionsTest : public GPUContextFixtureParam<rocRoller::DataType>
    {
    public:
        template <typename T>
        void convertFromF32()
        {
            auto numElements = 293u;

            // Float : 4 bytes
            auto inputBytesPerElement = 4u;
            auto outputDataType       = TypeInfo<T>::Var.dataType;
            // Half : 2 bytes, FP8 : 1 byte  and so on.
            auto outputBytesPerElement = DataTypeInfo::Get(outputDataType).elementBytes;

            // number of workitems per workgroup
            auto workitemsPerWorkgroup = 256u;

            // total number of workitems
            auto totalWorkitems = std::make_shared<Expression::Expression>(numElements);
            auto one            = std::make_shared<Expression::Expression>(1u);
            auto zero           = std::make_shared<Expression::Expression>(0u);

            // rocRoller sets the workgroup and workitem info for kernel
            auto k = m_context->kernel();
            k->setKernelName("ConvertFromF32Test");
            k->setKernelDimensions(1);
            k->setWorkgroupSize({workitemsPerWorkgroup, 1, 1});
            k->setWorkitemCount({totalWorkitems, one, one});
            k->setDynamicSharedMemBytes(zero);

            // rocRoller creates the command and kernel arguments
            auto command    = std::make_shared<Command>();
            auto inputTag   = command->allocateTag();
            auto inputExpr  = std::make_shared<Expression::Expression>(command->allocateArgument(
                {DataType::Float, PointerType::PointerGlobal}, inputTag, ArgumentType::Value));
            auto outputTag  = command->allocateTag();
            auto outputExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
                {outputDataType, PointerType::PointerGlobal}, outputTag, ArgumentType::Value));

            // input argument datatype (pointer to the input vector)
            k->addArgument({"input",
                            {DataType::Float, PointerType::PointerGlobal},
                            DataDirection::ReadOnly,
                            inputExpr});
            // output argument datatype (pointer to the output vector)
            k->addArgument({"output",
                            {outputDataType, PointerType::PointerGlobal},
                            DataDirection::WriteOnly,
                            outputExpr});

            // rocRoller schedules preamble and prolog (standard)
            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            // Convert Kernel instructions
            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_input, s_output;
                co_yield m_context->argLoader()->getValue("input", s_input);
                co_yield m_context->argLoader()->getValue("output", s_output);

                // workitem ID
                auto vgprWorkitem = m_context->kernel()->workitemIndex()[0];
                // workgroup ID
                auto sgprWorkgroup = m_context->kernel()->workgroupIndex()[0];

                // vgpr to hold our input/output vector values
                auto vgprInputOutput = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Float, 1);
                // vgpr to hold our index calculations
                auto vgprIndex = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);
                // vgpr to hold the number of workitems per workgroup
                auto vgprWorkitemCount = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);

                // rocRoller allocates the vgprs
                co_yield vgprInputOutput->allocate();
                co_yield vgprIndex->allocate();
                co_yield m_context->copier()->fill(vgprWorkitemCount,
                                                   Register::Value::Literal(workitemsPerWorkgroup));

                // index = (workgroupID * 256 + workitemID) * 4
                co_yield generateOp<Expression::MultiplyAdd>(
                    vgprIndex, sgprWorkgroup, vgprWorkitemCount, vgprWorkitem);
                co_yield generateOp<Expression::Multiply>(
                    vgprIndex, vgprIndex, Register::Value::Literal(inputBytesPerElement));

                //rocRoller does the intial setup for the buffer (global memory)
                auto bufDesc = std::make_shared<rocRoller::BufferDescriptor>(m_context);
                co_yield bufDesc->setup();
                auto bufInstOpts = rocRoller::BufferInstructionOptions();

                // set the buffer base pointer to the starting address of the input vector
                co_yield bufDesc->setBasePointer(s_input);
                co_yield m_context->mem()->loadBuffer(
                    vgprInputOutput, vgprIndex, 0, bufDesc, bufInstOpts, inputBytesPerElement);

                // datatype convert instruction
                co_yield_(
                    Instruction("v_cvt_f16_f32", {vgprInputOutput}, {vgprInputOutput}, {}, ""));

                // index = (workgroupID * 256 + workitemID) * 2
                co_yield generateOp<Expression::MultiplyAdd>(
                    vgprIndex, sgprWorkgroup, vgprWorkitemCount, vgprWorkitem);
                co_yield generateOp<Expression::Multiply>(
                    vgprIndex, vgprIndex, Register::Value::Literal(outputBytesPerElement));

                // set the buffer base pointer to the starting address of the output vector
                co_yield bufDesc->setBasePointer(s_output);
                co_yield m_context->mem()->storeBuffer(
                    vgprInputOutput, vgprIndex, 0, bufDesc, bufInstOpts, outputBytesPerElement);
            };

            // rocRoller schedules kernel instructions and postamble (standard)
            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());

            if(!isLocalDevice())
            {
                std::vector<char> assembledKernel = m_context->instructions()->assemble();
                EXPECT_GT(assembledKernel.size(), 0);
            }
            else
            {
                // Host side
                RandomGenerator    random(9861u);
                auto               nHalf = random.vector<T>(numElements, -65504.f, 65504.f);
                std::vector<float> input(numElements, 0.0);
                for(int i = 0; i < numElements; i++)
                    input[i] = nHalf[i];
                std::vector<T> output(numElements, 0.0);

                auto deviceInput  = make_shared_device(input);
                auto deviceOutput = make_shared_device(output);

                // rocRoller creates the commandKernel and launch the kernel
                CommandKernel commandKernel;
                commandKernel.setContext(m_context);
                CommandArguments commandArgs = command->createArguments();
                commandArgs.setArgument(outputTag, ArgumentType::Value, deviceOutput.get());
                commandArgs.setArgument(inputTag, ArgumentType::Value, deviceInput.get());
                commandKernel.launchKernel(commandArgs.runtimeArguments());

                // copy output data from buffer (global memory) into the host output vector
                ASSERT_THAT(hipMemcpy(output.data(),
                                      deviceOutput.get(),
                                      sizeof(T) * numElements,
                                      hipMemcpyDeviceToHost),
                            HasHipSuccess(0));

                for(int i = 0; i < numElements; i++)
                {
                    EXPECT_EQ(output[i], nHalf[i]);
                }
            }
        }
    };

    TEST_P(ConvertInstructionsTest, ConvertFromF32Test)
    {
        auto outputType = std::get<rocRoller::DataType>(GetParam());

        if(outputType == rocRoller::DataType::Half)
            convertFromF32<Half>();
    }

    INSTANTIATE_TEST_SUITE_P(
        ConvertInstructionsTests,
        ConvertInstructionsTest,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX90A},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX90A, {.sramecc = true}},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.sramecc = true}},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.xnack = true}}),
            ::testing::Values(rocRoller::DataType::Half)));
}
