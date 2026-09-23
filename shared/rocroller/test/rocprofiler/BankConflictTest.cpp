// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Scheduling/LDSModel.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/MEMObserver.hpp>
#include <rocRoller/Scheduling/RoundRobinScheduler.hpp>
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>

#include <common/Scheduling.hpp>

#include "../catch/TestContext.hpp"
#include "../catch/TestKernels.hpp"
#include "Agent.hpp"
#include "Utils.hpp"

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>

using namespace rocRoller;

class LDSBankConflictTestKernel : public AssemblyTestKernel
{
public:
    LDSBankConflictTestKernel(ContextPtr context,
                              uint32_t   workgroupSize,
                              size_t     instrDwords,
                              size_t     strideMultiplier,
                              bool       write)
        : AssemblyTestKernel(context)
        , m_workgroupSize(workgroupSize)
        , m_instrDwords(instrDwords)
        , m_strideMultiplier(strideMultiplier)
        , m_write(write)
    {
        auto k = m_context->kernel();
        k->setKernelDimensions(1);

        const auto one  = std::make_shared<Expression::Expression>(1u);
        const auto zero = std::make_shared<Expression::Expression>(0u);

        auto workitemCount = Expression::literal(m_workgroupSize * 256 * 32);
        k->setWorkgroupSize({m_workgroupSize, 1, 1});
        k->setWorkitemCount({workitemCount, one, one});
        k->setDynamicSharedMemBytes(zero);
    }

    void operator()()
    {
        KernelInvocation invocation{{m_workgroupSize * 256 * 32, 1, 1}, {m_workgroupSize, 1, 1}, 0};
        AssemblyTestKernel::operator()(invocation);
    }

protected:
    void generate() override
    {
        auto k = m_context->kernel();

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            const auto alignment = static_cast<int>(m_instrDwords);
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
                          * Expression::literal((4 * m_strideMultiplier * alignment)
                                                    % lds->getLDSAllocation()->size(),
                                                resultType(workitemIndex->expression()).varType),
                m_context);

            co_yield m_context->mem()->barrier({});

            for(int i = 0; i < ITERS; ++i)
            {
                const auto [start, end] = getAlignedSubset(regCount, m_instrDwords, i);
                const auto numBytes     = m_instrDwords * 4;

                if(m_write)
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
            co_yield Instruction::Wait(WaitCount::DSCnt(m_context->targetArchitecture(), 1));
            co_yield Instruction::Wait(WaitCount::DSCnt(m_context->targetArchitecture(), 0));
        };

        m_context->schedule(kb());

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

private:
    static constexpr int ITERS = 16;

    uint32_t m_workgroupSize;
    size_t   m_instrDwords;
    size_t   m_strideMultiplier;
    bool     m_write;
};

TEST_CASE("LDS bank model with bank conflicts", "[rocprofiler][gpu][lds-model]")
{
    using namespace Scheduling::LDSModel;

    constexpr auto workgroupSize = 64u;

    const std::vector<int>  instrSizes      = {1, 2, 4}; // b32, b64, b128
    const std::vector<int>  strides         = {1, 2, 4, 8, 16, 32, 64, 128}; // between threads
    const std::vector<bool> writeOperations = {false, true};

    for(auto instrDwords : instrSizes)
    {
        for(auto strideMultiplier : strides)
        {
            for(auto write : writeOperations)
            {
                const auto name = "lds_microkernel_" + std::to_string(instrDwords * 32) + "b_stride"
                                  + std::to_string(strideMultiplier) + "_"
                                  + (write ? "write" : "read");

                DYNAMIC_SECTION(name)
                {
                    rocRoller::profiler::reset();

                    auto context = TestContext::ForTestDevice({}, name);

                    if(not context->targetArchitecture().target().isCDNA4GPU())
                    {
                        SKIP("LDS Bank Model only implemented for CDNA 4 GPUs");
                    }

                    LDSBankConflictTestKernel kernel(
                        context.get(), workgroupSize, instrDwords, strideMultiplier, write);

                    std::vector<std::vector<rocRoller::profiler::InstructionProfile>> allLatencies;

                    for(int run = 0; run < NUM_RUNS; ++run)
                    {
                        const auto latencies
                            = rocRoller::profiler::loopUntilDispatchData([&]() { kernel(); });
                        allLatencies.push_back(latencies);

                        INFO("Run " << (run + 1) << ": " << toString(latencies));
                        Log::debug("Run " + std::to_string(run + 1) + ": " + toString(latencies));

                        REQUIRE(latencies.size() == 21);
                    }

                    GPUArchitectureGFX gfx = context->targetArchitecture().target().gfx;

                    auto baseAddresses = generateLDSAddresses(64, strideMultiplier, instrDwords);

                    RuntimeLDSInstruction ldsinstr;
                    ldsinstr.memoryOp.direction = write ? LdsDirection::Write : LdsDirection::Read;
                    ldsinstr.dwords             = instrDwords;
                    ldsinstr.baseAddresses      = baseAddresses;

                    uint predictedCycles = getInstructionCycles(ldsinstr, gfx);

                    uint issueCycles
                        = getInstructionIssueCycles(ldsinstr.memoryOp, ldsinstr.dwords);
                    uint dataCycles = getInstructionDataCycles(ldsinstr, gfx);

                    std::stringstream info;

                    info << fmt::format("dwords {}, stride {}, {}\n",
                                        instrDwords,
                                        strideMultiplier,
                                        write ? "write" : "read");

                    std::vector<uint64_t> ldsInstrCyclesPerRun;
                    std::vector<uint64_t> sWaitcntCyclesPerRun;

                    for(int run = 0; run < NUM_RUNS; ++run)
                    {
                        uint64_t maxLdsInstrCycles  = 0;
                        uint64_t lastSWaitcntCycles = 0;

                        for(const auto& data : allLatencies[run])
                        {
                            if((write && data.instruction.find("ds_write") != std::string::npos)
                               || (!write && data.instruction.find("ds_read") != std::string::npos))
                            {
                                maxLdsInstrCycles = std::max(maxLdsInstrCycles, data.meanLatency());
                            }
                            else if(data.instruction.find("s_waitcnt") != std::string::npos)
                            {
                                lastSWaitcntCycles = data.meanLatency();
                            }
                        }

                        ldsInstrCyclesPerRun.push_back(maxLdsInstrCycles);
                        sWaitcntCyclesPerRun.push_back(lastSWaitcntCycles);

                        info << fmt::format("  Run {}: LDS Cycles: {}, s_waitcnt Cycles: {}\n",
                                            run + 1,
                                            maxLdsInstrCycles,
                                            lastSWaitcntCycles);
                    }

                    uint64_t actualMaxLdsInstrCycles  = MedianOfOddElements(ldsInstrCyclesPerRun);
                    uint64_t actualLastSWaitcntCycles = MedianOfOddElements(sWaitcntCyclesPerRun);
                    info << fmt::format("  Median s_waitcnt Cycles: {}\n",
                                        actualLastSWaitcntCycles);
                    info << fmt::format("  Median LDS Instruction Cycles: {}\n",
                                        actualMaxLdsInstrCycles);
                    info << fmt::format("  Model Predicted Cycles: {}\n", predictedCycles);
                    info << fmt::format("    Issue Cycles: {}\n", issueCycles);
                    info << fmt::format("    Data Cycles: {}\n", dataCycles);

                    INFO(info.str());
                    Log::debug(info.str());

                    CHECK_THAT(actualLastSWaitcntCycles,
                               Catch::Matchers::WithinAbs(predictedCycles, 1ul));
                    if(write && instrDwords == 4)
                        // ds_write_b128 requires queue info
                        CHECK_THAT(actualMaxLdsInstrCycles,
                                   Catch::Matchers::WithinAbs(predictedCycles, 12ul));
                    else
                        CHECK_THAT(actualMaxLdsInstrCycles,
                                   Catch::Matchers::WithinAbs(predictedCycles, 4ul));
                }
            }
        }
    }
}
