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

    // ========================================================================
    // GEMMUnrollKTestSuite
    // ========================================================================

    class GEMMUnrollKTestSuite : public BaseGEMMContextFixture<>
    {
    };

    TEST_P(GEMMUnrollKTestSuite, GPU_GEMM_UnrollK_Basic)
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

    TEST_P(GEMMUnrollKTestSuite, GPU_GEMM_UnrollK_LDS)
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

    TEST_P(GEMMUnrollKTestSuite, GPU_GEMM_UnrollK_MoreLDS)
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

    TEST_P(GEMMUnrollKTestSuite, GPU_GEMM_UnrollK_MoreLDS_A)
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

    TEST_P(GEMMUnrollKTestSuite, GPU_GEMM_UnrollK_MoreLDS_B)
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

    TEST_P(GEMMUnrollKTestSuite, GPU_GEMM_UnrollK_FP16_Prefetch3)
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

    INSTANTIATE_TEST_SUITE_P(GEMMUnrollKTest, GEMMUnrollKTestSuite, currentGPUISA());

    // ========================================================================
    // GEMMUnrollKLDSPrefetchTestSuite
    // ========================================================================

    // Params are: prefetchInFlight, prefetchLDSFactor, prefetchMixMemOps
    // Filter: develop had prefetchInFlight in {1, 2} only
    class GEMMUnrollKLDSPrefetchTestSuite
        : public BaseGEMMContextFixture<std::tuple<int, int, bool>>
    {
    };

    TEST_P(GEMMUnrollKLDSPrefetchTestSuite, GPU_GEMM_UnrollK_LDS_Prefetch)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        auto params    = std::get<1>(GetParam());
        int  inflight  = std::get<0>(params);
        int  ldsFactor = std::get<1>(params);
        bool mixMemOps = std::get<2>(params);

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

    using GEMMUnrollKLDSPrefetchParamGenerator
        = ::testing::internal::ParamGenerator<GEMMUnrollKLDSPrefetchTestSuite::ParamType>;
    static auto FilterLDSPrefetchParams(GEMMUnrollKLDSPrefetchParamGenerator&& inputParamGenerator)
    {
        std::vector<GEMMUnrollKLDSPrefetchTestSuite::ParamType> filtered;
        for(auto const& inputParam : inputParamGenerator)
        {
            int const inflight = std::get<0>(std::get<1>(inputParam));
            if(inflight == 3)
                continue;
            filtered.push_back(inputParam);
        }
        return ::testing::ValuesIn(filtered);
    }

    static auto UnrollKPrefetchParamCombine()
    {
        return ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(1, 2, 3), // prefetchInFlight
                               ::testing::Values(0, 1, 2), // prefetchLDSFactor
                               ::testing::Values(false, true)));
    }

    INSTANTIATE_TEST_SUITE_P(GEMMUnrollKTest,
                             GEMMUnrollKLDSPrefetchTestSuite,
                             FilterLDSPrefetchParams(UnrollKPrefetchParamCombine()));

    // ========================================================================
    // GEMMUnrollKFP16PrefetchTestSuite
    // ========================================================================

    // Params are: prefetchInFlight, prefetchLDSFactor, prefetchMixMemOps
    // Filter: develop had prefetchInFlight in {1, 2}, prefetchLDSFactor in {0, 2}
    class GEMMUnrollKFP16PrefetchTestSuite
        : public BaseGEMMContextFixture<std::tuple<int, int, bool>>
    {
    };

    TEST_P(GEMMUnrollKFP16PrefetchTestSuite, GPU_GEMM_UnrollK_FP16_LDS_Prefetch)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        auto params    = std::get<1>(GetParam());
        int  inflight  = std::get<0>(params);
        int  ldsFactor = std::get<1>(params);
        bool mixMemOps = std::get<2>(params);

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

    using GEMMUnrollKFP16PrefetchParamGenerator
        = ::testing::internal::ParamGenerator<GEMMUnrollKFP16PrefetchTestSuite::ParamType>;
    static auto
        FilterFP16PrefetchParams(GEMMUnrollKFP16PrefetchParamGenerator&& inputParamGenerator)
    {
        std::vector<GEMMUnrollKFP16PrefetchTestSuite::ParamType> filtered;
        for(auto const& inputParam : inputParamGenerator)
        {
            auto const& params    = std::get<1>(inputParam);
            int const   inflight  = std::get<0>(params);
            int const   ldsFactor = std::get<1>(params);
            if(inflight == 3 || ldsFactor == 1)
                continue;
            filtered.push_back(inputParam);
        }
        return ::testing::ValuesIn(filtered);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMUnrollKTest,
                             GEMMUnrollKFP16PrefetchTestSuite,
                             FilterFP16PrefetchParams(UnrollKPrefetchParamCombine()));

    // ========================================================================
    // GEMMUnrollKLDSMultiPrefetchTestSuite
    // ========================================================================

    // Params are: prefetchInFlight, prefetchLDSFactor, prefetchMixMemOps
    // Filter: develop had prefetchLDSFactor in {0, 2} only
    class GEMMUnrollKLDSMultiPrefetchTestSuite
        : public BaseGEMMContextFixture<std::tuple<int, int, bool>>
    {
    };

    TEST_P(GEMMUnrollKLDSMultiPrefetchTestSuite, GPU_GEMM_UnrollK_LDS_MultiPrefetch)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        auto params    = std::get<1>(GetParam());
        int  inflight  = std::get<0>(params);
        int  ldsFactor = std::get<1>(params);
        bool mixMemOps = std::get<2>(params);

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

    using GEMMUnrollKLDSMultiPrefetchParamGenerator
        = ::testing::internal::ParamGenerator<GEMMUnrollKLDSMultiPrefetchTestSuite::ParamType>;
    static auto FilterLDSMultiPrefetchParams(
        GEMMUnrollKLDSMultiPrefetchParamGenerator&& inputParamGenerator)
    {
        std::vector<GEMMUnrollKLDSMultiPrefetchTestSuite::ParamType> filtered;
        for(auto const& inputParam : inputParamGenerator)
        {
            int const ldsFactor = std::get<1>(std::get<1>(inputParam));
            if(ldsFactor == 1)
                continue;
            filtered.push_back(inputParam);
        }
        return ::testing::ValuesIn(filtered);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMUnrollKTest,
                             GEMMUnrollKLDSMultiPrefetchTestSuite,
                             FilterLDSMultiPrefetchParams(UnrollKPrefetchParamCombine()));

    // ========================================================================
    // GEMMUnrollKTailLoopTestSuite
    // ========================================================================

    // Params are: K dimension size
    class GEMMUnrollKTailLoopTestSuite : public BaseGEMMContextFixture<std::tuple<int>>
    {
    };

    TEST_P(GEMMUnrollKTailLoopTestSuite, GPU_GEMM_UnrollK_TailLoop)
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
        gemm.fuseLoops = true;
        gemm.tailLoops = true;
        gemm.unrollK   = 4;
        gemm.macK      = 8;

        basicGEMM<float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMUnrollKTest,
        GEMMUnrollKTailLoopTestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Values(8, 16, 24, 32, 40, 48, 56, 64, 592))); // 592 = 18 * 4 * 8 + 8 * 2

} // namespace GEMMTests
