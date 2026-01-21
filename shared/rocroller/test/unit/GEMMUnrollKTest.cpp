
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

    // Params are: K dimension size
    class GEMMUnrollKTailLoopTestGPU : public BaseGEMMContextFixture<std::tuple<int>>
    {
    };

    TEST_P(UnrollKTestGPU, GPU_BasicGEMM)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.storeLDSD = false;
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
        gemm.storeLDSD = false;
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
        gemm.storeLDSD = false;
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
        gemm.storeLDSD = false;
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
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.macK      = 8;
        basicGEMM<float>(gemm);
    }

    TEST_P(UnrollKTestGPU, GPU_BasicGEMMLDSPrefetch)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storeLDSD = true;
        gemm.fuseLoops = true;
        gemm.unrollK   = 2;
        gemm.macK      = 4;
        gemm.prefetch  = true;
        gemm.macM      = gemm.waveM * 4;
        gemm.macN      = gemm.waveN * 2;

        for(auto inflight : {1, 2})
        {
            gemm.prefetchInFlight = inflight;
            for(auto ldsFactor : {0, 1, 2})
            {
                gemm.prefetchLDSFactor = ldsFactor;
                for(auto mixMemOps : {false, true})
                {
                    gemm.prefetchMixMemOps = mixMemOps;
                    basicGEMM<float>(gemm);
                }
            }
        }
    }

    TEST_P(UnrollKTestGPU, GPU_BasicGEMMFP16LDSPrefetch)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 16 * 2;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storeLDSD = true;
        gemm.fuseLoops = true;
        gemm.unrollK   = 2;
        gemm.macM      = 128;
        gemm.macN      = 128;
        gemm.macK      = 16;
        gemm.prefetch  = true;
        gemm.waveK     = 8;

        gemm.transA = "N";
        gemm.transB = "N";

        for(auto inflight : {1, 2})
        {
            gemm.prefetchInFlight = inflight;
            for(auto ldsFactor : {0, 2})
            {
                gemm.prefetchLDSFactor = ldsFactor;
                for(auto mixMemOps : {false, true})
                {
                    gemm.prefetchMixMemOps = mixMemOps;
                    basicGEMM<Half>(gemm);
                }
            }
        }
    }

    TEST_P(UnrollKTestGPU, GPU_BasicGEMMLDSMultiPrefetch)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 3;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 3;
        gemm.macK      = 4;
        gemm.prefetch  = true;

        for(auto inflight : {1, 2, 3})
        {
            gemm.prefetchInFlight = inflight;
            for(auto ldsFactor : {0, 2})
            {
                gemm.prefetchLDSFactor = ldsFactor;
                for(auto mixMemOps : {false, true})
                {
                    gemm.prefetchMixMemOps = mixMemOps;
                    basicGEMM<float>(gemm);
                }
            }
        }
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
        gemm.storeLDSD         = false;
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
        gemm.storeLDSD = false;
        gemm.fuseLoops = true;
        gemm.tailLoops = true;
        gemm.unrollK   = 4;
        gemm.macK      = 8;

        basicGEMM<float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMTest, UnrollKTestGPU, currentGPUISA());

    INSTANTIATE_TEST_SUITE_P(
        GEMMUnrollKTailLoopTest,
        GEMMUnrollKTailLoopTestGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Values(8, 16, 24, 32, 40, 48, 56, 64, 592))); // 592 = 18 * 4 * 8 + 8 * 2

} // namespace GEMMTests
