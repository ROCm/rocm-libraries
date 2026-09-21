// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include "GEMMTestBase.hpp"
#include <rocRoller/GPUArchitecture/GPUCapability.hpp>

namespace GEMMTests
{
    using namespace rocRoller;
    namespace SolutionParams = rocRoller::Parameters::Solution;

    // ========================================================================
    // GEMMBasicTestSuite
    // ========================================================================

    /**
     * GEMMBasicTestGPU: Consolidated basic tests for smoke tests.
     * 
     * These tests exercise core GEMM features with minimal
     * parameterization (only GPU architecture).
     */
    class GEMMBasicTestSuite : public BaseGEMMContextFixture<>
    {
    };

    TEST_P(GEMMBasicTestSuite, GPU_BasicGEMM_FP32)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMBasicTestSuite, GPU_BasicGEMM_FP16)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.waveM = 32;
        gemm.waveN = 32;
        gemm.waveK = 8;
        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMBasicTestSuite, GPU_BasicGEMM_BF16)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x8_1k);
        GEMMProblem gemm;
        gemm.waveM = 32;
        gemm.waveN = 32;
        gemm.waveK = 8;
        basicGEMM<BFloat16, BFloat16, float>(gemm);
    }

    TEST_P(GEMMBasicTestSuite, GPU_BasicGEMM_FP8)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
        GEMMProblem gemm;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = 32;
        gemm.macM  = 256;
        gemm.macN  = 256;
        gemm.macK  = 128;
        basicGEMM<FP8, FP8, float>(gemm);
    }

    TEST_P(GEMMBasicTestSuite, GPU_BasicGEMM_StreamK)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.numWGs  = 4;
        gemm.streamK = StreamKMode::Standard;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMBasicTestSuite, GPU_BasicGEMM_DirectLDS)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        REQUIRE_ARCH_CAP(GPUCapability::HasDirectToLds);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDS;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMBasicTestSuite, GPU_BasicGEMM_UnrollK)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.unrollK   = 2;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMBasicTestSuite, GPU_BasicGEMM_Jammed)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.waveM          = 16;
        gemm.waveN          = 16;
        gemm.waveK          = 16;
        gemm.macM           = 64;
        gemm.macN           = 64;
        gemm.macK           = 16;
        gemm.workgroupSizeX = 256;
        gemm.workgroupSizeY = 1;
        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath      = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.m              = 2 * gemm.macM;
        gemm.n              = 2 * gemm.macN;
        gemm.k              = 2 * gemm.macK;
        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMBasicTestSuite, GPU_BasicGEMM_WMMA)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA);
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x16_f16);
        GEMMProblem gemm;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = 16;
        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        basicGEMM<Half, Half, float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMBasicTest, GEMMBasicTestSuite, currentGPUISA());

} // namespace GEMMTests
