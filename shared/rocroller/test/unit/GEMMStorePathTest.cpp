/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
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
