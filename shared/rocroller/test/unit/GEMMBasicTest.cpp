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
