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
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GPUContextFixture.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class LdsMicrokernel : public CurrentGPUContextFixture
    {
    };

    TEST_F(LdsMicrokernel, GPU_LdsMicrokernel)
    {
        Settings::getInstance()->set(Settings::KernelAssembler, AssemblerType::Subprocess);

        ASSERT_EQ(true, isLocalDevice());

        auto command = std::make_shared<Command>();

        auto k = m_context->kernel();

        k->setKernelDimensions(1);

        const auto one  = std::make_shared<Expression::Expression>(1u);
        const auto zero = std::make_shared<Expression::Expression>(0u);

        const auto workgroupSize = 64u;
        auto       workitemCount = Expression::literal(256u * workgroupSize); // 256 CU on MI350X
        k->setWorkgroupSize({workgroupSize, 1, 1});
        k->setWorkitemCount({workitemCount, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto strideMultiplier = 1;
        if(const char* env_p = std::getenv("BYTE_STRIDE"))
            strideMultiplier = atoi(env_p); // adjust for bank conflict

        auto instrDwords = 1;
        if(const char* env_p = std::getenv("INSTR_WIDTH"))
            instrDwords = atoi(env_p); // e.g. 1 for ds_read_b32

        bool write = false;
        if(const char* env_p = std::getenv("WRITE"))
            write = atoi(env_p) == 1 ? true : false;

        int ITERS = 16;
        if(const char* env_p = std::getenv("ITERS"))
            ITERS = atoi(env_p);

        auto kb = [&]() -> Generator<Instruction> {
            const auto loops
                = 0; // Loop due to suspecting instruction cache causing latency, doesn't seem to change anything from testing
            const auto alignment = instrDwords;
            const auto regCount  = 256 - 8; // leave a few

            auto dst = Register::Value::Placeholder(
                m_context,
                Register::Type::Vector,
                DataType::Raw32,
                regCount,
                Register::AllocationOptions{
                    .contiguousChunkWidth = Register::FULLY_CONTIGUOUS,
                    .alignment            = alignment,
                });
            co_yield dst->allocate();

            auto lds = Register::Value::AllocateLDS(
                m_context,
                DataType::Raw32,
                m_context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize) / 4);
            auto ldsWithOffset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);
            auto workitemIndex = m_context->kernel()->workitemIndex()[0];
            co_yield Expression::generate(
                ldsWithOffset,
                Expression::literal(lds->getLDSAllocation()->offset())
                    + workitemIndex->expression()
                          * Expression::literal((4 * strideMultiplier * alignment)
                                                % lds->getLDSAllocation()->size()),
                m_context);

            // auto i = Register::Value::Placeholder(
            //     m_context, Register::Type::Scalar, DataType::UInt64, 1);
            // co_yield Expression::generate(i, Expression::literal(loops + 1), m_context);

            // auto label = m_context->labelAllocator()->label("label");
            // co_yield Instruction::Label(label);

            auto getSubset = [](size_t n, size_t m, size_t i) -> std::pair<size_t, size_t> {
                // If run out of n, wrap around
                size_t num_complete_chunks = n / m;
                if(num_complete_chunks == 0)
                {
                    return {0, 0};
                }
                size_t chunk_index = i % num_complete_chunks;
                size_t start       = chunk_index * m;
                return {start, start + m};
            };

            const int ALREADY_FINISHED = ITERS - 3;

            // auto addrs = Register::Value::Placeholder(
            //     m_context, Register::Type::Vector, DataType::UInt32, 4);
            // for(int i = 0; i < 4; ++i)
            // {
            //     auto subset = addrs->subset({i});
            //     co_yield Expression::generate(
            //         subset, ldsWithOffset->expression() + Expression::literal(i * 16), m_context);
            // }

            co_yield m_context->mem()->barrier({});

            for(int i = 0; i < ITERS; ++i)
            {
                const auto [start, end] = getSubset(regCount, instrDwords, i);
                const auto numBytes     = instrDwords * 4;

                if(write)
                {
                    co_yield m_context->mem()->storeLocal(
                        ldsWithOffset, dst->subset(Generated(iota(start, end))), 0, numBytes);
                }
                else
                {
                    co_yield m_context->mem()->loadLocal(
                        dst->subset(Generated(iota(start, end))), ldsWithOffset, 0, numBytes);
                }
            }

            for(int i = ITERS - 1 - ALREADY_FINISHED; i >= 0; --i)
            {
                co_yield Instruction::Wait(WaitCount::DSCnt(m_context->targetArchitecture(), i));
            }
            // co_yield Expression::generate(i, i->expression() - Expression::literal(1), m_context);
            // co_yield m_context->brancher()->branchIfNonZero(label, i);
        };

        m_context->schedule(kb());

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel;
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        CommandArguments commandArgs = command->createArguments();

        commandKernel.launchKernel(commandArgs.runtimeArguments());
    }

}
