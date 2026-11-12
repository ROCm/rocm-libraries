// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "../../../library/src/amd_detail/rocblaslt/src/include/alpha_mode_logging.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ScaleAlphaVec.hpp>
#include <gtest/gtest.h>

TEST(HostUnitAlphaModeLogging, smoke_BenchAndProfileDistinguishScalarFromVector)
{
    struct Case
    {
        int         mode;
        bool        scalar;
        const char* flag;
        bool        vector;
    };
    const Case cases[] = {
        {0, false, "", false},
        {1, false, "--scaleAlpha_vector", true},
        {2, false, "--scaleAlpha_vector", true},
        {TensileLite::ScaleAlphaVecDeviceScalarBit, true, "--device_scalar_alpha", false},
    };
    for(const auto& test : cases)
    {
        SCOPED_TRACE(test.mode);
        auto problem = TensileLite::ContractionProblemGemm::GEMM(
            false, false, 4, 4, 4, 4, 4, 4, 1.0, false, 1);
        problem.setUseScaleAlphaVec(test.mode);
        problem.setParams().setDeviceScalarAlpha(test.scalar);
        EXPECT_STREQ(rocblaslt::benchAlphaFlag(problem), test.flag);
        EXPECT_EQ(rocblaslt::logVectorAlpha(problem), test.vector);
    }
}

TEST(HostUnitAlphaModeLogging, smoke_TraceDistinguishesScalarFromVector)
{
    EXPECT_STREQ(rocblaslt::traceAlphaLabel(rocblaslt_pointer_mode_host), "alpha");
    EXPECT_STREQ(rocblaslt::traceAlphaLabel(rocblaslt_pointer_mode_alpha_device_vector_beta_host),
                 "alphaVector");
    EXPECT_STREQ(
        rocblaslt::traceAlphaLabel(rocblaslt_pointer_mode_alpha_device_scalar_beta_host_ext),
        "alphaScalarDevice");
}
