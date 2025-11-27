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

using namespace rocRoller;
namespace SolutionParams = rocRoller::Parameters::Solution;

namespace GEMMTests
{

    class GEMMTestStreamKGPU
        : public BaseGEMMContextFixture<std::tuple<StreamKMode,
                                                   SolutionParams::LoadPath, /* loadPathA */
                                                   SolutionParams::LoadPath, /* loadPathB */
                                                   bool /* storeLDSD */>>
    {
    };

    class GEMMTestStreamKWGMGPU
        : public BaseGEMMContextFixture<std::tuple<int, /* workgroupMapping dim */
                                                   int, /* workgroupMapping value */
                                                   bool, /* workgroupRemapXCC */
                                                   StreamKMode>>
    {
    };

    TEST_P(GEMMTestStreamKGPU, GPU_BasicGEMMFP16StreamK)
    {
        if(m_context->targetArchitecture().target().isCDNA1GPU())
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMMStreamK test";
        }

        GEMMProblem gemm;

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        gemm.numWGs = deviceProperties.multiProcessorCount;

        gemm.waveK = 8;
        gemm.macK  = 16;

        gemm.macM           = 128;
        gemm.macN           = 256;
        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 2;

        gemm.m = gemm.macM * 8;
        gemm.n = gemm.macN * gemm.numWGs / 2 + gemm.macN * 2;

        ASSERT_GE(gemm.m * gemm.n / gemm.macM / gemm.macN, gemm.numWGs);

        gemm.streamK = StreamKMode::Standard;
        gemm.k       = gemm.macK * 8;

        // TODO: Does not work with unrolling K
        //gemm.unrollK          = 2;
        //gemm.prefetch         = true;
        //gemm.prefetchInFlight = 2;

        std::tie(gemm.streamK, gemm.loadPathA, gemm.loadPathB, gemm.storeLDSD)
            = std::get<1>(GetParam());

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestStreamKGPU, GPU_BasicGEMMFP16StreamKSmall)
    {
        if(m_context->targetArchitecture().target().isCDNA1GPU())
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMMStreamK test";
        }

        GEMMProblem gemm;

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        gemm.numWGs = 3;

        gemm.waveK = 8;
        gemm.macK  = 16;

        gemm.macM           = 128;
        gemm.macN           = 128;
        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.m = 4 * gemm.macM;
        gemm.n = 4 * gemm.macN;

        ASSERT_GE(gemm.m * gemm.n / gemm.macM / gemm.macN, gemm.numWGs);

        gemm.k = gemm.macK * 8;

        std::tie(gemm.streamK, gemm.loadPathA, gemm.loadPathB, gemm.storeLDSD)
            = std::get<1>(GetParam());

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestStreamKGPU, GPU_BasicGEMMFP16StreamK_MultipleFixups)
    {
        if(m_context->targetArchitecture().target().isCDNA1GPU())
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMMStreamK test";
        }

        GEMMProblem gemm;

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));

        gemm.macM = 128;
        gemm.macN = 128;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 128;
        gemm.workgroupSizeY = 2;

        gemm.numWGs = 128;

        auto numTilesM = 1;
        auto numTilesN = 2;
        auto numTilesK = 249;

        gemm.m = numTilesM * gemm.macM;
        gemm.n = numTilesN * gemm.macN;
        gemm.k = numTilesK * gemm.macK;

        // assert that the number of output tiles is smaller than number of WGs
        // which means there is not enough data-parallel tiles, and has to split
        // K dimension into multiple tiles
        ASSERT_GE(gemm.numWGs, gemm.m * gemm.n / gemm.macM / gemm.macN);

        std::tie(gemm.streamK, gemm.loadPathA, gemm.loadPathB, gemm.storeLDSD)
            = std::get<1>(GetParam());

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestStreamKWGMGPU, GPU_BasicGEMMStreamKWorkgroupMapping)
    {
        if(m_context->targetArchitecture().target().isCDNA1GPU())
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMMStreamKWorkgroupMapping test";
        }

        GEMMProblem gemm;

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        gemm.numWGs = deviceProperties.multiProcessorCount;

        gemm.m = gemm.macM * 8;
        gemm.n = gemm.macN * gemm.numWGs / 2 + gemm.macN * 2;

        ASSERT_GE(gemm.m * gemm.n / gemm.macM / gemm.macN, gemm.numWGs);

        gemm.k = gemm.macK * 8;

        std::tie(gemm.workgroupMappingDim,
                 gemm.workgroupMappingValue,
                 gemm.workgroupRemapXCC,
                 gemm.streamK)
            = std::get<1>(GetParam());

        basicGEMM<float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMTestStreamKWGM,
        GEMMTestStreamKWGMGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(0, 1), /* workgroupMapping dim */
                               ::testing::Values(1, 2, 6), /* workgroupMapping value */
                               ::testing::Values(true, false), /* remapWorkgroupXCC */
                               ::testing::Values(StreamKMode::Standard,
                                                 StreamKMode::TwoTile,
                                                 StreamKMode::TwoTileDPFirst))));

    INSTANTIATE_TEST_SUITE_P(
        GEMMTestStreamK,
        GEMMTestStreamKGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(
                    StreamKMode::Standard, StreamKMode::TwoTile, StreamKMode::TwoTileDPFirst),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                  SolutionParams::LoadPath::BufferToVGPR), /* loadPathA */
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                  SolutionParams::LoadPath::BufferToVGPR), /* loadPathB */
                ::testing::Values(true, false) /* storeLDSD */
                )));

} // namespace GEMMTests
