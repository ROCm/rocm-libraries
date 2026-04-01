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

    // Params are: pretileScaleA, pretileScaleB, pretileA, pretileB
    class GEMMPretileTestSuite : public BaseGEMMContextFixture<bool, bool, bool, bool>
    {
    };

    TEST_P(GEMMPretileTestSuite, GPU_GEMM_Scale_Pretile_F4_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto [arch, pretileScaleA, pretileScaleB, pretileA, pretileB] = GetParam();

        auto gemm           = GEMMProblemF8F6F4{32, 32, 64};
        gemm.transA         = "T";
        gemm.transB         = "N";
        gemm.macM           = 256;
        gemm.macN           = 256;
        gemm.macK           = 128;
        gemm.m              = 4096;
        gemm.n              = 4096;
        gemm.k              = 32768;
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
        if(pretileA)
            gemm.pretileA = {64, 64};
        if(pretileB)
            gemm.pretileB = {64, 64};

        basicGEMM<FP4, FP4, float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMPretileTest,
                             GEMMPretileTestSuite,
                             ::testing::Combine(currentGPUISA(),
                                                ::testing::Bool(),
                                                ::testing::Bool(),
                                                ::testing::Bool(),
                                                ::testing::Bool()));

    // ========================================================================
    // GEMMPretileSwizzleScale32x8TestSuite
    // ========================================================================

    class GEMMPretileSwizzleScale32x8TestSuite : public BaseGEMMContextFixture<>
    {
    };

    TEST_P(GEMMPretileSwizzleScale32x8TestSuite, GPU_GEMM_PretileB_SwizzleScale_32x8_F4_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto gemm           = GEMMProblemF8F6F4{32, 32, 64};
        gemm.transA         = "T";
        gemm.transB         = "N";
        gemm.macM           = 256;
        gemm.macN           = 256;
        gemm.macK           = 256;
        gemm.m              = 2 * gemm.macM;
        gemm.n              = 2 * gemm.macN;
        gemm.k              = 4 * gemm.macK;
        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 2;

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

        gemm.swizzleScale = true;
        gemm.swizzleM     = 32;
        gemm.swizzleN     = 32;
        gemm.swizzleK     = 8;

        gemm.scalePretileA = {32, 8};
        gemm.scalePretileB = {8, 32};
        gemm.pretileB      = {128, 128};

        basicGEMM<FP4, FP4, float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMPretileTest,
                             GEMMPretileSwizzleScale32x8TestSuite,
                             currentGPUISA());

}
