// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "../../../library/src/amd_detail/rocblaslt/src/include/candidate_preparation.hpp"

#include <Tensile/AMDGPU.hpp>
#include <gtest/gtest.h>

namespace
{
    TEST(CandidatePreparation_pre_checkin, FallbackAndAutomaticMappingPackDifferentArguments)
    {
        TensileLite::AMDGPU hardware(TensileLite::AMDGPU::Processor::gfx950, 256, "test");
        TensileLite::ContractionSolution solution;
        solution.internalArgsSupport.version          = 2;
        solution.internalArgsSupport.wgm              = true;
        solution.sizeMapping.workGroupMappingXCC      = 8;
        solution.sizeMapping.workGroupMappingXCCGroup = 1;
        auto problem                                  = TensileLite::ContractionProblemGemm::GEMM(
            false, false, 64, 64, 64, 64, 64, 64, 0.0, false, 1);
        problem.setParams().setWGMXCC(4);

        auto packedXcc = [&] {
            uint32_t arg0 = 0, arg1 = 0;
            solution.calculateInternalArgs<false>(
                arg0, arg1, &hardware, problem.getParams(), 1, 8, 0, 0, 0, 0, 0, 1, {});
            return (arg1 >> 16) & 0x3f;
        };
        for(bool fallback : {true, false, true, false})
        {
            // Supply the hardware classification already cached by the solution.
            solution.isFallbackCUSol = fallback ? 1 : 0;
            problem.setParams().resetInternalArgs();
            EXPECT_EQ(rocblaslt::prepareCandidateXcc(problem, solution, hardware), fallback);
            EXPECT_EQ(packedXcc(), fallback ? 1u : 8u);
        }
    }
}
