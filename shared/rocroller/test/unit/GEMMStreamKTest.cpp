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
    namespace SolutionParams = rocRoller::Parameters::Solution;

    class StreamKMultipleFixupsTestGPU
        : public BaseGEMMContextFixture<std::tuple<StreamKMode,
                                                   SolutionParams::LoadPath, /* loadPathA */
                                                   SolutionParams::LoadPath, /* loadPathB */
                                                   bool /* storeLDSD */>>
    {
    };

    class StreamKWGMTestGPU
        : public BaseGEMMContextFixture<std::tuple<int, /* workgroupMapping dim */
                                                   int, /* workgroupMapping value */
                                                   bool, /* workgroupRemapXCC */
                                                   StreamKMode>>
    {
    };

    // Params are: A & B type, UnrollK, LoadPath A, LoadPath B, LDS D, StreamK two-tile, beta is zero
    class StreamKTestGPU : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                                    int,
                                                                    SolutionParams::LoadPath,
                                                                    SolutionParams::LoadPath,
                                                                    bool,
                                                                    rocRoller::StreamKMode,
                                                                    bool>>
    {
    };

    TEST_P(StreamKMultipleFixupsTestGPU, GPU_BasicGEMMFP16)
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

    TEST_P(StreamKWGMTestGPU, GPU_BasicGEMMStreamKWorkgroupMapping)
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

    TEST_P(StreamKTestGPU, GPU_BasicGEMM)
    {
        if(m_context->targetArchitecture().target().isCDNA1GPU())
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMMStreamK test";
        }

        auto [typeAB, unrollK, loadPathA, loadPathB, storeLDSD, mode, betaZero]
            = std::get<1>(GetParam());

        if((typeAB == DataType::Float)
           && (loadPathA == SolutionParams::LoadPath::BufferToLDSViaVGPR)
           && (loadPathB == SolutionParams::LoadPath::BufferToLDSViaVGPR) && (storeLDSD)
           && (mode != StreamKMode::Standard))
        {
            // We run out of LDS in this case
            GTEST_SKIP() << "Skipping GPU_BasicGEMMStreamK test";
        }

        GEMMProblem gemm;

        if(typeAB == DataType::Half)
        {
            gemm.waveK = 8;
            gemm.macK  = 16;
        }

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        gemm.numWGs = deviceProperties.multiProcessorCount;

        gemm.m = gemm.macM * 8;
        gemm.n = gemm.macN * gemm.numWGs / 2 + gemm.macN * 2;

        ASSERT_GE(gemm.m * gemm.n / gemm.macM / gemm.macN, gemm.numWGs);

        gemm.streamK = mode;
        gemm.k       = gemm.macK * 8;

        gemm.loadPathA = loadPathA;
        gemm.loadPathB = loadPathB;
        gemm.storeLDSD = storeLDSD;
        gemm.unrollK   = unrollK;

        gemm.transA = "T";
        gemm.transB = "N";

        if(betaZero)
            gemm.beta = 0;

        switch(typeAB)
        {
        case DataType::Half:
            basicGEMM<Half>(gemm);
            break;
        case DataType::Float:
            basicGEMM<float>(gemm);
            break;
        default:
            Throw<FatalError>(fmt::format("Unexpected data type: {}. ", toString(typeAB)));
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMTest,
        StreamKWGMTestGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(0, 1), /* workgroupMapping dim */
                               ::testing::Values(1, 2, 6), /* workgroupMapping value */
                               ::testing::Values(true, false), /* remapWorkgroupXCC */
                               ::testing::Values(StreamKMode::Standard,
                                                 StreamKMode::TwoTile,
                                                 StreamKMode::TwoTileDPFirst))));

    INSTANTIATE_TEST_SUITE_P(
        GEMMTest,
        StreamKMultipleFixupsTestGPU,
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

    INSTANTIATE_TEST_SUITE_P(
        GEMMTest,
        StreamKTestGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(rocRoller::DataType::Float,
                                  rocRoller::DataType::Half), // typeAB
                ::testing::Values(1, 2), // UnrollK
                ::testing::Values(SolutionParams::LoadPath::BufferToVGPR,
                                  SolutionParams::LoadPath::BufferToLDSViaVGPR), // LoadPath A
                ::testing::Values(SolutionParams::LoadPath::BufferToVGPR,
                                  SolutionParams::LoadPath::BufferToLDSViaVGPR), // LoadPath B
                ::testing::Values(true, false), // storeLDSD
                ::testing::Values(rocRoller::StreamKMode::Standard,
                                  rocRoller::StreamKMode::TwoTile,
                                  rocRoller::StreamKMode::TwoTileDPFirst), // StreamKMode
                ::testing::Values(true, false)))); // betaZero

} // namespace GEMMTests
