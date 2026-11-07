// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "../../../library/src/amd_detail/rocblaslt/src/include/candidate_preparation.hpp"

#include <Tensile/AMDGPU.hpp>
#include <gtest/gtest.h>

namespace
{
    TEST(CandidatePreparation_pre_checkin, FallbackRequestsOneXccAndAutomaticMappingClearsIt)
    {
        TensileLite::AMDGPU hardware(TensileLite::AMDGPU::Processor::gfx950, 256, "test");
        TensileLite::ContractionSolution solution;
        auto                             problem = TensileLite::ContractionProblemGemm::GEMM(
            false, false, 64, 64, 64, 64, 64, 64, 0.0, false, 1);
        problem.setParams().setWGMXCC(4);

        // The kernel-argument packer uses a nonzero value as the mapping and
        // zero as the candidate's own, so the parameter carries the choice.
        auto requestedXcc = [&] { return problem.getParams().wgmxcc(); };
        for(bool fallback : {true, false, true, false})
        {
            // Supply the hardware classification already cached by the solution.
            solution.isFallbackCUSol = fallback ? 1 : 0;
            problem.setParams().resetInternalArgs();
            EXPECT_EQ(rocblaslt::prepareCandidateXcc(problem, solution, hardware), fallback);
            EXPECT_EQ(requestedXcc(), fallback ? 1 : 0);
        }
    }
}
