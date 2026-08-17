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
#include <rocRoller/Scheduling/Observers/WaitcntObserver.hpp>
#include <rocRoller/Scheduling/RoundRobinScheduler.hpp>
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>

#include <common/Scheduling.hpp>

#include "../catch/TestContext.hpp"
#include "../catch/TestKernels.hpp"
#include "Agent.hpp"
#include "LDSTestUtils.hpp"
#include "Utils.hpp"

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <functional>

using namespace rocRoller;

namespace KernelBodies
{
    Generator<Instruction> WeaveLDSAndWaitcnt0(ParameterizedLDSKernel* kernel)
    {
        int  counter = 0;
        auto context = kernel->getContext();
        for(int i = 1; i <= 4; ++i)
        {
            for(int k = 0; k < i; ++k)
            {
                co_yield kernel->scheduleLdsInstruction(counter);
            }
            co_yield Instruction::Wait(WaitCount::DSCnt(context->targetArchitecture(), 0));
            co_yield context->mem()->barrier({});
        }
    }

    Generator<Instruction> weaveNonzeroWaitcnt(ParameterizedLDSKernel* kernel)
    {
        int  counter = 0;
        auto context = kernel->getContext();
        for(int i = 0; i < 8; ++i)
        {
            for(int k = 0; k < 16; ++k)
            {
                co_yield kernel->scheduleLdsInstruction(counter);
            }
            co_yield Instruction::Wait(WaitCount::DSCnt(context->targetArchitecture(), i));
        }
    }

    Generator<Instruction> weaveLDSAndDecreasingWaitcnt(ParameterizedLDSKernel* kernel)
    {
        int  counter = 0;
        auto context = kernel->getContext();
        for(int i = 1; i <= 8; ++i)
        {
            for(int k = 0; k < i; ++k)
            {
                co_yield kernel->scheduleLdsInstruction(counter);
            }
            for(int w = 0; w < i; ++w)
            {
                co_yield Instruction::Wait(
                    WaitCount::DSCnt(context->targetArchitecture(), i - w - 1));
            }
        }
    }
}

class WaitcntTestValidator
{
public:
    static void WeaveLDSAndWaitcnt0(const KernelLatencyResults&  result,
                                    const LatencyAnalysisResult& analysis,
                                    int                          instrDwords,
                                    int                          strideMultiplier,
                                    bool                         write,
                                    uint32_t                     workgroupSize)
    {
        INFO(result.infoStr);

        if(workgroupSize > 64u)
        {
            CHECK(analysis.incorrectPredictionCount <= 28);
        }
        else
        {
            CHECK(analysis.incorrectPredictionCount <= 6);
        }
    }

    static void weaveNonzeroWaitcnt(const KernelLatencyResults&  result,
                                    const LatencyAnalysisResult& analysis,
                                    int                          instrDwords,
                                    int                          strideMultiplier,
                                    bool                         write,
                                    uint32_t /*workgroupSize*/)
    {
        INFO(result.infoStr);

        if(instrDwords == 4)
        {
            if(write)
            {
                CHECK(analysis.incorrectPredictionCount <= 64);
            }
            else
            {
                CHECK(std::abs(analysis.totalDelta) <= 80);
            }
        }
        else if(strideMultiplier >= 4)
        {
            CHECK(std::abs(analysis.totalDelta) <= 80);
        }
        else
        {
            CHECK((analysis.incorrectPredictionCount <= 8 || std::abs(analysis.totalDelta) <= 0));
        }
    }

    static void weaveLDSAndDecreasingWaitcnt(const KernelLatencyResults&  result,
                                             const LatencyAnalysisResult& analysis,
                                             int                          instrDwords,
                                             int                          strideMultiplier,
                                             bool                         write,
                                             uint32_t /*workgroupSize*/)
    {
        INFO(result.infoStr);

        if(write && instrDwords == 4)
        {
            CHECK(analysis.incorrectPredictionCount <= 28);
        }
        else
        {
            CHECK((analysis.incorrectPredictionCount <= 10 || std::abs(analysis.totalDelta) <= 0));
        }
    }
};

TEST_CASE("Weave LDS and waitcnt 0", "[rocprofiler][lds-model][gpu]")
{
    /*
    ds_read_b128 v[60:63], v1, model 16, profiler 16, delta 0
    s_waitcnt lgkmcnt(0), model 168, profiler 164, delta -4
    ds_read_b128 v[64:67], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[68:71], v1, model 4, profiler 4, delta 0
    s_waitcnt lgkmcnt(0), model 72, profiler 72, delta 0
    ds_read_b128 v[72:75], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[76:79], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[80:83], v1, model 4, profiler 4, delta 0
    s_waitcnt lgkmcnt(0), model 84, profiler 84, delta 0
    ds_read_b128 v[84:87], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[88:91], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[92:95], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[96:99], v1, model 4, profiler 4, delta 0
    s_waitcnt lgkmcnt(0), model 96, profiler 96, delta 0
    ds_read_b128 v[100:103], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[104:107], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[108:111], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[112:115], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[116:119], v1, model 4, profiler 4, delta 0
    s_waitcnt lgkmcnt(0), model 108, profiler 108, delta 0
    */
    runLDSTest(
        {true, KernelBodies::WeaveLDSAndWaitcnt0, WaitcntTestValidator::WeaveLDSAndWaitcnt0});
}

TEST_CASE("Weave LDS and non-zero waitcnt", "[rocprofiler][lds-model][gpu]")
{
    /*
    ...
    s_waitcnt lgkmcnt(2), model 136, profiler 132, delta -4
    ds_read_b128 v[124:127], v1, model 4, profiler 4, delta 0
    ...
    ds_read_b128 v[160:163], v1, model 4, profiler 4, delta 0
    s_waitcnt lgkmcnt(3), model 120, profiler 116, delta -4
    ds_read_b128 v[164:167], v1, model 4, profiler 4, delta 0
    ...
    ds_read_b128 v[200:203], v1, model 4, profiler 8, delta 4
    s_waitcnt lgkmcnt(4), model 104, profiler 100, delta -4
    ...
    */
    runLDSTest(
        {false, KernelBodies::weaveNonzeroWaitcnt, WaitcntTestValidator::weaveNonzeroWaitcnt});
}

TEST_CASE("Weave LDS and decreasing waitcnt", "[rocprofiler][lds-model][gpu]")
{
    /*
    ds_write_b32 v1, v2, model 8, profiler 8, delta 0
  * s_waitcnt lgkmcnt(0), model 60, profiler 52, delta -8
    ds_write_b32 v1, v3, model 8, profiler 8, delta 0
    ds_write_b32 v1, v4, model 8, profiler 8, delta 0
  * s_waitcnt lgkmcnt(1), model 52, profiler 44, delta -8
  * s_waitcnt lgkmcnt(0), model 16, profiler 8, delta -8
    ds_write_b32 v1, v5, model 8, profiler 8, delta 0
    ds_write_b32 v1, v6, model 8, profiler 8, delta 0
    ds_write_b32 v1, v7, model 8, profiler 8, delta 0
  * s_waitcnt lgkmcnt(2), model 44, profiler 36, delta -8
  * s_waitcnt lgkmcnt(1), model 16, profiler 8, delta -8
  * s_waitcnt lgkmcnt(0), model 16, profiler 8, delta -8
    ds_write_b32 v1, v8, model 8, profiler 8, delta 0
    ds_write_b32 v1, v9, model 8, profiler 8, delta 0
    ds_write_b32 v1, v10, model 8, profiler 8, delta 0
    ds_write_b32 v1, v11, model 8, profiler 8, delta 0
  * s_waitcnt lgkmcnt(3), model 36, profiler 28, delta -8
  * s_waitcnt lgkmcnt(2), model 16, profiler 8, delta -8
  * s_waitcnt lgkmcnt(1), model 16, profiler 8, delta -8
  * s_waitcnt lgkmcnt(0), model 16, profiler 8, delta -8
    */
    runLDSTest({false,
                KernelBodies::weaveLDSAndDecreasingWaitcnt,
                WaitcntTestValidator::weaveLDSAndDecreasingWaitcnt});
}
