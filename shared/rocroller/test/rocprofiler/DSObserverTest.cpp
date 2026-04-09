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
#include "LDSTestUtils.hpp"
#include "Utils.hpp"

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>

using namespace rocRoller;

namespace KernelBodies
{
    Generator<Instruction> ldsArithmeticWeave(ParameterizedLDSKernel* kernel)
    {
        auto context = kernel->getContext();
        auto s0
            = Register::Value::Placeholder(context, Register::Type::Scalar, DataType::UInt32, 1);
        auto s1
            = Register::Value::Placeholder(context, Register::Type::Scalar, DataType::UInt32, 1);
        co_yield s0->allocate();
        co_yield s1->allocate();

        int counter = 0;

        for(int i = 0; i < 14; ++i)
        {
            co_yield kernel->scheduleLdsInstruction(counter);
        }

        for(int i = 1; i < 8; ++i)
        {
            for(int k = 0; k < 4; ++k)
            {
                co_yield kernel->scheduleLdsInstruction(counter);
            }
            for(int j = 0; j < i; ++j)
            {
                co_yield generateOp<Expression::Add>(s0, s0, s1);
            }
        }
    }

    Generator<Instruction> steadyStateLds(ParameterizedLDSKernel* kernel)
    {
        int counter = 0;
        for(int i = 0; i < 32; ++i)
        {
            co_yield kernel->scheduleLdsInstruction(counter);
        }
    }
}

class DSObserverValidator
{
public:
    static void validateArithmeticWeave(const KernelLatencyResults&  result,
                                        const LatencyAnalysisResult& analysis,
                                        int                          instrDwords,
                                        int                          strideMultiplier,
                                        bool                         write,
                                        uint32_t /*workgroupSize*/)
    {
        INFO(result.infoStr);

        if(write && instrDwords == 4)
        {
            CHECK(analysis.incorrectPredictionCount <= 16);
        }
        else
        {
            CHECK((analysis.incorrectPredictionCount <= 4 || analysis.totalDelta == 0));
        }
    }

    static void validateSteadyState(const KernelLatencyResults&  result,
                                    const LatencyAnalysisResult& analysis,
                                    int                          instrDwords,
                                    int                          strideMultiplier,
                                    bool                         write,
                                    uint32_t                     workgroupSize)
    {
        INFO(result.infoStr);

        if(write && instrDwords == 4)
        {
            if(workgroupSize <= 128u)
            {
                CHECK(std::abs(analysis.totalDelta) <= 210);
            }
            else
            {
                CHECK(analysis.totalAbsoluteDelta <= 1600);
            }
        }
        else if(workgroupSize == 64u)
        {
            CHECK((analysis.incorrectPredictionCount <= 5 || analysis.totalDelta == 0));
        }
        else if(workgroupSize > 64u)
        {
            CHECK(std::abs(analysis.totalDelta) <= 300);
        }
        else
        {
            CHECK((analysis.incorrectPredictionCount <= 5 || analysis.totalDelta == 0));
        }
    }
};

TEST_CASE("Weave LDS and s_add", "[rocprofiler][lds-model][gpu]")
{
    runLDSTest(
        {false, KernelBodies::ldsArithmeticWeave, DSObserverValidator::validateArithmeticWeave});
}

TEST_CASE("Steady state LDS instructions", "[rocprofiler][lds-model][gpu]")
{
    /*
    ds_read_b128 v[4:7], v1, model 4, profiler 4, delta 0
    ds_read_b128 v[8:11], v1, model 4, profiler 4, delta 0
    ... 32 times
    */
    runLDSTest({true, KernelBodies::steadyStateLds, DSObserverValidator::validateSteadyState});
}
