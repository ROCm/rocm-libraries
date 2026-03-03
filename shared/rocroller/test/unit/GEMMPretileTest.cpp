// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include "GEMMF8F6F4.hpp"
#include "GEMMTestBase.hpp"

namespace GEMMTests
{
    using namespace rocRoller;
    namespace SolutionParams = rocRoller::Parameters::Solution;

    // ========================================================================
    // GEMMPretileTestSuite
    // ========================================================================

    // Params are: pretileScaleA, pretileScaleB, pretileB
    class GEMMPretileTestSuite : public BaseGEMMContextFixture<bool, bool, bool>
    {
    };

    TEST_P(GEMMPretileTestSuite, GPU_GEMM_Scale_Pretile_F4_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto [arch, pretileScaleA, pretileScaleB, pretileB] = GetParam();
        (void)arch;

        auto gemm           = GEMMProblemF8F6F4{32, 32, 64};
        gemm.transA         = "T";
        gemm.transB         = "N";
        gemm.macM           = 256;
        gemm.macN           = 256;
        gemm.macK           = 128;
        gemm.m              = 2 * gemm.macM;
        gemm.n              = 2 * gemm.macN;
        gemm.k              = 4 * gemm.macK;
        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadScalePathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadScalePathB = SolutionParams::LoadPath::BufferToVGPR;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;
        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;
        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        if(pretileScaleA)
            gemm.scalePretileA = {256, 4};
        if(pretileScaleB)
            gemm.scalePretileB = {4, 256};
        if(pretileB)
            gemm.pretileB = {64, 64};

        basicGEMM<FP4, FP4, float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMPretileTest,
                             GEMMPretileTestSuite,
                             ::testing::Combine(currentGPUISA(),
                                                ::testing::Bool(),
                                                ::testing::Bool(),
                                                ::testing::Bool()));

}
