// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GEMMF8F6F4.hpp"
#include "GEMMTestBase.hpp"
#include <rocRoller/Parameters/Solution/StoreOption.hpp>

namespace GEMMTests
{
    using namespace rocRoller;
    namespace SolutionParams = rocRoller::Parameters::Solution;

    class GEMMStorePathTestGPU : public BaseGEMMContextFixture<SolutionParams::StorePath>
    {
    };

    TEST_P(GEMMStorePathTestGPU, GPU_BasicGEMM_IsLDSStore_StorePath)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        const auto storePath = std::get<1>(GetParam());

        GEMMProblem gemm;
        gemm.storePath = storePath;

        // For VGPRToGlobalMemoryWithBuffer and VGPRToGlobal, we should not use LDS
        if(storePath == SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer
           || storePath == SolutionParams::StorePath::VGPRToGlobalMemoryWithGlobal)
        {
            EXPECT_FALSE(SolutionParams::IsLDSStore(storePath));
        }
        else
        {
            EXPECT_TRUE(SolutionParams::IsLDSStore(storePath));
        }

        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMStorePathTestGPU, GPU_BasicGEMM_FP8_StorePath)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
        const auto storePath = std::get<1>(GetParam());

        auto gemm      = GEMMProblemF8NT{};
        gemm.storePath = storePath;

        basicGEMM<FP8, FP8, float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMStorePathTest,
        GEMMStorePathTestGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Values(SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer,
                              SolutionParams::StorePath::VGPRToGlobalMemoryWithGlobal,
                              SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer,
                              SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithGlobal)));

} // namespace GEMMTests
