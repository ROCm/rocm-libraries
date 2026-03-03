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
    // GEMMLargeMacroTileTestSuite
    // ========================================================================

    // Params are: load A path, load B path
    class GEMMLargeMacroTileTestSuite
        : public BaseGEMMContextFixture<
              std::tuple<SolutionParams::LoadPath, SolutionParams::LoadPath>>
    {
    };

    TEST_P(GEMMLargeMacroTileTestSuite, DISABLED_GPU_GEMM_LargeTile_Basic)
    {
        // NOTE: This test takes hours to finish
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem problem{.m = 1024, .n = 1024, .k = 256, .macM = 256, .macN = 256};
        std::tie(problem.loadPathA, problem.loadPathB) = std::get<1>(GetParam());
        basicGEMM<float>(problem);

        // To see runtime of kernel generation, compile code with timer enabled and
        // std::cout << TimerPool::summary() << std::endl;
    }

    TEST_P(GEMMLargeMacroTileTestSuite, GPU_GEMM_LargeTile_F8F6F4)
    {
        // NOTE: This test takes about 13 seconds (without enabling Unroll) to
        // finish when FuseLoops orders all pairs of memory nodes one by one.
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        auto gemm = GEMMProblemF8F6F4{32, 32, 64};
        gemm.m    = 512;
        gemm.n    = 256;
        gemm.k    = 512;

        gemm.macM = 256;
        gemm.macN = 256;
        gemm.macK = 128;

        std::tie(gemm.loadPathA, gemm.loadPathB) = std::get<1>(GetParam());

        // Use unrollK will significantly increase the kernel generation time.
        // To enable unrollK, maxVGPR has to be increased as well.
        //
        //gemm.unrollK = 2;
        //setKernelOptions({.maxVGPRs = 1024});

        basicGEMM<FP8, FP8, float>(gemm);

        // To see runtime of kernel generation, compile code with timer enabled and
        // std::cout << TimerPool::summary() << std::endl;
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMLargeMacroTileTest,
        GEMMLargeMacroTileTestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR),
                               ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR))));

} // namespace GEMMTests
