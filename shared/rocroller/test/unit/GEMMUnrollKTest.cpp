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

namespace GEMMTests
{
    using namespace rocRoller;

    class UnrollKTestGPU : public BaseGEMMContextFixture<>
    {
    };

    TEST_P(UnrollKTestGPU, GPU_BasicGEMM)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.fuseLoops = true;
        gemm.unrollK   = 2;

        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;

        gemm.macM = 128;
        gemm.macK = 4;

        basicGEMM<float>(gemm);
    }

    TEST_P(UnrollKTestGPU, GPU_BasicGEMMLDS)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.fuseLoops = false;
        gemm.unrollK   = 2;
        gemm.macK      = 4;
        basicGEMM<float>(gemm);
    }

    TEST_P(UnrollKTestGPU, GPU_BasicGEMMMoreLDS)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.macK      = 8;
        basicGEMM<float>(gemm);
    }

    TEST_P(UnrollKTestGPU, GPU_BasicGEMMMoreLDSA)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.macK      = 8;
        basicGEMM<float>(gemm);
    }

    TEST_P(UnrollKTestGPU, GPU_BasicGEMMMoreLDSB)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.macK      = 8;
        basicGEMM<float>(gemm);
    }

    TEST_P(UnrollKTestGPU, GPU_BasicGEMMFP16Prefetch3)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.m                 = 4096;
        gemm.n                 = 4096;
        gemm.k                 = 2048 * 3;
        gemm.loadPathA         = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB         = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath         = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.fuseLoops         = false;
        gemm.unrollK           = 3;
        gemm.macM              = 128;
        gemm.macN              = 16;
        gemm.macK              = 64;
        gemm.waveM             = 16;
        gemm.waveN             = 16;
        gemm.waveK             = 16;
        gemm.workgroupSizeX    = 256;
        gemm.workgroupSizeY    = 1;
        gemm.prefetch          = true;
        gemm.prefetchInFlight  = 3;
        gemm.prefetchLDSFactor = 2;
        gemm.prefetchMixMemOps = true;
        basicGEMM<Half>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMTest, UnrollKTestGPU, currentGPUISA());

    // Params are: K dimension size
    class GEMMUnrollKTailLoopTestGPU : public BaseGEMMContextFixture<std::tuple<int>>
    {
    };

    TEST_P(GEMMUnrollKTailLoopTestGPU, GPU_BasicGEMMUnrollKTailLoop)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        auto [k] = std::get<1>(GetParam());

        GEMMProblem gemm;
        gemm.m         = 64;
        gemm.n         = 128;
        gemm.k         = k;
        gemm.transA    = "T";
        gemm.transB    = "N";
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.fuseLoops = true;
        gemm.tailLoops = true;
        gemm.unrollK   = 4;
        gemm.macK      = 8;

        basicGEMM<float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMUnrollKTailLoopTest,
        GEMMUnrollKTailLoopTestGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Values(8, 16, 24, 32, 40, 48, 56, 64, 592))); // 592 = 18 * 4 * 8 + 8 * 2

    // Params are: prefetchInFlight, prefetchLDSFactor, prefetchMixMemOps
    class GEMMLDSPrefetchTestGPU : public BaseGEMMContextFixture<std::tuple<int, int, bool>>
    {
    };

    TEST_P(GEMMLDSPrefetchTestGPU, GPU_BasicGEMMLDSPrefetch)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        auto [inflight, ldsFactor, mixMemOps] = std::get<1>(GetParam());

        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer;
        gemm.fuseLoops = true;
        gemm.unrollK   = 2;
        gemm.macK      = 4;
        gemm.prefetch  = true;
        gemm.macM      = gemm.waveM * 4;
        gemm.macN      = gemm.waveN * 2;

        gemm.prefetchInFlight  = inflight;
        gemm.prefetchLDSFactor = ldsFactor;
        gemm.prefetchMixMemOps = mixMemOps;

        basicGEMM<float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMLDSPrefetchTest,
        GEMMLDSPrefetchTestGPU,
        ::testing::Combine(currentGPUISA(),
                           ::testing::Combine(::testing::Values(1, 2), // prefetchInFlight
                                              ::testing::Values(0, 1, 2), // prefetchLDSFactor
                                              ::testing::Values(false,
                                                                true)))); // prefetchMixMemOps

    // Params are: prefetchInFlight, prefetchLDSFactor, prefetchMixMemOps
    class GEMMFP16LDSPrefetchTestGPU : public BaseGEMMContextFixture<std::tuple<int, int, bool>>
    {
    };

    TEST_P(GEMMFP16LDSPrefetchTestGPU, GPU_BasicGEMMFP16LDSPrefetch)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        auto [inflight, ldsFactor, mixMemOps] = std::get<1>(GetParam());

        GEMMProblem gemm;
        gemm.k         = 64 * 16 * 2;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer;
        gemm.fuseLoops = true;
        gemm.unrollK   = 2;
        gemm.macM      = 128;
        gemm.macN      = 128;
        gemm.macK      = 16;
        gemm.prefetch  = true;
        gemm.waveK     = 8;

        gemm.transA = "N";
        gemm.transB = "N";

        gemm.prefetchInFlight  = inflight;
        gemm.prefetchLDSFactor = ldsFactor;
        gemm.prefetchMixMemOps = mixMemOps;

        basicGEMM<Half>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMFP16LDSPrefetchTest,
        GEMMFP16LDSPrefetchTestGPU,
        ::testing::Combine(currentGPUISA(),
                           ::testing::Combine(::testing::Values(1, 2), // prefetchInFlight
                                              ::testing::Values(0, 2), // prefetchLDSFactor
                                              ::testing::Values(false,
                                                                true)))); // prefetchMixMemOps

    // Params are: prefetchInFlight, prefetchLDSFactor, prefetchMixMemOps
    class GEMMLDSMultiPrefetchTestGPU : public BaseGEMMContextFixture<std::tuple<int, int, bool>>
    {
    };

    TEST_P(GEMMLDSMultiPrefetchTestGPU, GPU_BasicGEMMLDSMultiPrefetch)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        auto [inflight, ldsFactor, mixMemOps] = std::get<1>(GetParam());

        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 3;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.fuseLoops = false;
        gemm.unrollK   = 3;
        gemm.macK      = 4;
        gemm.prefetch  = true;

        gemm.prefetchInFlight  = inflight;
        gemm.prefetchLDSFactor = ldsFactor;
        gemm.prefetchMixMemOps = mixMemOps;

        basicGEMM<float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMLDSMultiPrefetchTest,
        GEMMLDSMultiPrefetchTestGPU,
        ::testing::Combine(currentGPUISA(),
                           ::testing::Combine(::testing::Values(1, 2, 3), // prefetchInFlight
                                              ::testing::Values(0, 2), // prefetchLDSFactor
                                              ::testing::Values(false,
                                                                true)))); // prefetchMixMemOps

} // namespace GEMMTests
