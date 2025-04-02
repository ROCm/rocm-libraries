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

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <common/Utilities.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/Utilities/Random.hpp>

#include <catch2/catch_test_case_info.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/interfaces/catch_interfaces_testcase.hpp>

using namespace rocRoller;

namespace ExpressionTest
{
    struct RandomNumberExpressionKernel : public AssemblyTestKernel
    {
        RandomNumberExpressionKernel(rocRoller::ContextPtr context, unsigned numWorkItems)
            : AssemblyTestKernel(context)
            , m_numWorkItems(numWorkItems)
        {
            auto const& arch = m_context->targetArchitecture().target();
            if(!arch.isCDNAGPU())
                SKIP("Test not yet supported on " << arch);
        }

        void generate() override
        {
            // total number of workitems
            auto totalWorkitems = std::make_shared<Expression::Expression>(m_numWorkItems);
            auto one            = std::make_shared<Expression::Expression>(1u);
            auto zero           = std::make_shared<Expression::Expression>(0u);
            auto kernel         = m_context->kernel();

            kernel->setKernelName("RandomNumberGenerationTest");
            kernel->setKernelDimensions(1u);

            auto const workitemsPerWorkgroup = m_numWorkItems;
            kernel->setWorkgroupSize({workitemsPerWorkgroup, 1u, 1u});
            kernel->setWorkitemCount({totalWorkitems, one, one});
            kernel->setDynamicSharedMemBytes(zero);

            auto dataType = DataType::UInt32;

            kernel->addArgument(
                {"result", {dataType, PointerType::PointerGlobal}, DataDirection::WriteOnly});

            kernel->addArgument(
                {"seeds", {dataType, PointerType::PointerGlobal}, DataDirection::ReadOnly});

            // rocRoller schedules preamble and prolog (standard)
            m_context->schedule(kernel->preamble());
            m_context->schedule(kernel->prolog());

            // UInt32 = 4 bytes
            auto inputBytesPerElement  = 4u;
            auto outputBytesPerElement = inputBytesPerElement;

            // Random Number Generation Kernel instructions
            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_input, s_output;

                co_yield m_context->argLoader()->getValue("result", s_output);
                co_yield m_context->argLoader()->getValue("seeds", s_input);

                // workitem ID
                auto vgprWorkitem = m_context->kernel()->workitemIndex()[0];
                // workgroup ID
                auto sgprWorkgroup = m_context->kernel()->workgroupIndex()[0];

                auto vgpr_input = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);

                auto vgpr_output = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);

                // vgpr to hold our index calculations
                auto vgprIndex = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);

                // vgpr to hold the number of workitems per workgroup
                auto vgprWorkitemCount = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);

                co_yield vgprIndex->allocate();
                co_yield m_context->copier()->fill(vgprWorkitemCount,
                                                   Register::Value::Literal(workitemsPerWorkgroup));

                // index = (workgroupID * 256 + workitemID) * 4
                co_yield generateOp<Expression::MultiplyAdd>(
                    vgprIndex, sgprWorkgroup, vgprWorkitemCount, vgprWorkitem);
                co_yield generateOp<Expression::Multiply>(
                    vgprIndex, vgprIndex, Register::Value::Literal(inputBytesPerElement));

                // Load first input value
                //rocRoller does the intial setup for the buffer (global memory)
                auto bufDesc = std::make_shared<rocRoller::BufferDescriptor>(m_context);
                co_yield bufDesc->setup();
                auto bufInstOpts = rocRoller::BufferInstructionOptions();

                // set the buffer base pointer to the starting address of the input vector
                co_yield bufDesc->setBasePointer(s_input);
                co_yield m_context->mem()->loadBuffer(
                    vgpr_input, vgprIndex, 0, bufDesc, bufInstOpts, inputBytesPerElement);

                // Add TID (vgprWorkitem) to seed
                co_yield generateOp<Expression::Add>(vgpr_input, vgpr_input, vgprWorkitem);

                // Generate random numbers based on the input seed
                auto seedExpr = vgpr_input->expression();
                auto randomNumberExpr
                    = std::make_shared<Expression::Expression>(Expression::RandomNumber{seedExpr});
                co_yield Expression::generate(vgpr_output, randomNumberExpr, m_context);

                // index = (workgroupID * 256 + workitemID) * 4
                co_yield generateOp<Expression::MultiplyAdd>(
                    vgprIndex, sgprWorkgroup, vgprWorkitemCount, vgprWorkitem);
                co_yield generateOp<Expression::Multiply>(
                    vgprIndex, vgprIndex, Register::Value::Literal(outputBytesPerElement));

                // set the buffer base pointer to the starting address of the output vector
                co_yield bufDesc->setBasePointer(s_output);
                // Write the result into global memory
                co_yield m_context->mem()->storeBuffer(
                    vgpr_output, vgprIndex, 0, bufDesc, bufInstOpts, outputBytesPerElement);
            };

            // rocRoller schedules kernel instructions and postamble (standard)
            m_context->schedule(kb());
            m_context->schedule(kernel->postamble());
            m_context->schedule(kernel->amdgpu_metadata());
        }

    protected:
        unsigned const m_numWorkItems;
    };

    TEST_CASE("Run random number expression kernel", "[expression][random-number][gpu]")
    {
        auto const numWorkItems = 256u;

        auto                         context = TestContext::ForTestDevice();
        RandomNumberExpressionKernel kernel(context.get(), numWorkItems);
        RandomGenerator              random(12345u);

        // Testing random seeds and unique seeds across workitems
        auto                  random_seeds = random.vector<uint32_t>(numWorkItems, 0u, 123456u);
        std::vector<uint32_t> unique_seeds(numWorkItems, 12345u);

        for(auto const& seeds : {random_seeds, unique_seeds})
        {
            auto                  d_seeds = make_shared_device(seeds);
            std::vector<uint32_t> gpuResult(numWorkItems);
            auto                  d_result = make_shared_device(gpuResult);

            kernel({{numWorkItems, 1u, 1u}, {numWorkItems, 1u, 1u}, 0u},
                   d_result.get(),
                   d_seeds.get());

            CHECK(kernel.getAssembledKernel().size() > 0);

            std::vector<uint32_t> h_result(seeds.size());
            for(size_t i = 0; i < seeds.size(); i++)
            {
                h_result[i] = LFSRRandomNumberGenerator(seeds[i] + i);
            }
            REQUIRE_THAT(d_result, HasDeviceVectorEqualTo(h_result));
        }

        if(context->targetArchitecture().HasCapability(GPUCapability::HasPRNG))
        {
            auto instructions = context->instructions()->toString();
            CHECK(instructions.find("v_prng_b32") != std::string::npos);
        }
    }
}
