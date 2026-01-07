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

    // ProblemConfig: (dataTypeAB, macM, macN, macK, m, n, k, numWGs)
    using ProblemConfig = std::tuple<rocRoller::DataType, int, int, int, int, int, int, int>;

    class StreamKMultipleFixupsTestGPU
        : public BaseGEMMContextFixture<std::tuple<ProblemConfig,
                                                   StreamKMode,
                                                   SolutionParams::LoadPath, /* loadPathA */
                                                   SolutionParams::LoadPath, /* loadPathB */
                                                   bool /* storeLDSD */>>
    {
    };

    class StreamKWGMTestGPU
        : public BaseGEMMContextFixture<std::tuple<int, /* workgroupMapping dim */
                                                   int, /* workgroupMapping value */
                                                   bool, /* workgroupRemapXCC */
                                                   StreamKMode,
                                                   SolutionParams::LoadPath, /* loadPathA */
                                                   SolutionParams::LoadPath /* loadPathB */>>
    {
    };

    // PrefetchConfig: (prefetchInFlight, prefetchLDSFactor)
    using PrefetchConfig = std::tuple<int, int>;

    // Params: typeAB, unrollK, loadPathA, loadPathB, storeLDSD, mode, betaZero, prefetchConfig
    class StreamKTestGPU : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                                    int,
                                                                    SolutionParams::LoadPath,
                                                                    SolutionParams::LoadPath,
                                                                    bool,
                                                                    rocRoller::StreamKMode,
                                                                    bool,
                                                                    PrefetchConfig>>
    {
    };

    TEST_P(StreamKMultipleFixupsTestGPU, GPU_BasicGEMM)
    {
        if(m_context->targetArchitecture().target().isCDNA1GPU())
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMM test: CDNA1 not supported";
        }

        auto [problemConfig, mode, loadPathA, loadPathB, storeLDSD] = std::get<1>(GetParam());
        auto [dataTypeAB, macM, macN, macK, m, n, k, numWGs]        = problemConfig;

        GEMMProblem gemm;

        gemm.macM   = macM;
        gemm.macN   = macN;
        gemm.macK   = macK;
        gemm.m      = m;
        gemm.n      = n;
        gemm.k      = k;
        gemm.numWGs = numWGs;

        if(dataTypeAB == DataType::Half)
        {
            gemm.waveK = 8;
        }

        gemm.workgroupSizeX = 128;
        gemm.workgroupSizeY = 2;

        // assert that the number of output tiles is smaller than number of WGs
        // which means there is not enough data-parallel tiles, and has to split
        // K dimension into multiple tiles
        ASSERT_GE(gemm.numWGs, gemm.m * gemm.n / gemm.macM / gemm.macN);

        gemm.streamK   = mode;
        gemm.loadPathA = loadPathA;
        gemm.loadPathB = loadPathB;
        gemm.storeLDSD = storeLDSD;

        switch(dataTypeAB)
        {
        case DataType::Half:
            basicGEMM<Half>(gemm, false, false, 100);
            break;
        case DataType::Float:
            basicGEMM<float>(gemm, false, false, 100);
            break;
        default:
            Throw<FatalError>(fmt::format("Unexpected data type: {}. ", toString(dataTypeAB)));
        }
    }

    TEST_P(StreamKWGMTestGPU, GPU_BasicGEMMStreamKWorkgroupMapping)
    {
        if(m_context->targetArchitecture().target().isCDNA1GPU())
        {
            GTEST_SKIP()
                << "Skipping GPU_BasicGEMMStreamKWorkgroupMapping test: CDNA1 not supported";
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
                 gemm.streamK,
                 gemm.loadPathA,
                 gemm.loadPathB)
            = std::get<1>(GetParam());

        basicGEMM<float>(gemm);
    }

    TEST_P(StreamKTestGPU, GPU_BasicGEMM)
    {
        if(m_context->targetArchitecture().target().isCDNA1GPU())
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMM test: CDNA1 not supported";
        }

        auto [typeAB, unrollK, loadPathA, loadPathB, storeLDSD, mode, betaZero, prefetchConfig]
            = std::get<1>(GetParam());
        auto [prefetchInFlight, prefetchLDSFactor] = prefetchConfig;

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

        // Enable prefetch if prefetchInFlight > 0
        if(prefetchInFlight > 0)
        {
            gemm.prefetch          = true;
            gemm.prefetchInFlight  = prefetchInFlight;
            gemm.prefetchLDSFactor = prefetchLDSFactor;
        }

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
                                                 StreamKMode::TwoTileDPFirst),
                               ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR),
                               ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR))));

    INSTANTIATE_TEST_SUITE_P(
        GEMMTest,
        StreamKMultipleFixupsTestGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(
                    // ProblemConfig: (dataTypeAB, macM, macN, macK, m, n, k, numWGs)
                    ProblemConfig{rocRoller::DataType::Half, 128, 128, 16, 128, 256, 15936, 128},
                    ProblemConfig{rocRoller::DataType::Float,
                                  64,
                                  64,
                                  64,
                                  256,
                                  256,
                                  16384,
                                  256}), /* problemConfig */
                ::testing::Values(
                    StreamKMode::Standard, StreamKMode::TwoTile, StreamKMode::TwoTileDPFirst),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                  SolutionParams::LoadPath::BufferToVGPR,
                                  SolutionParams::LoadPath::GlobalToVGPR,
                                  SolutionParams::LoadPath::GlobalToLDSViaVGPR), /* loadPathA */
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                  SolutionParams::LoadPath::BufferToVGPR,
                                  SolutionParams::LoadPath::GlobalToVGPR,
                                  SolutionParams::LoadPath::GlobalToLDSViaVGPR), /* loadPathB */
                ::testing::Values(true, false) /* storeLDSD */
                )));

    using StreamKParamGenerator = ::testing::internal::ParamGenerator<StreamKTestGPU::ParamType>;
    static auto FilterValidStreamKParams(StreamKParamGenerator&& inputParamGenerator)
    {
        using LP = SolutionParams::LoadPath;
        using DT = rocRoller::DataType;
        using SM = rocRoller::StreamKMode;

        std::vector<StreamKTestGPU::ParamType> filtered;
        for(auto const& inputParam : inputParamGenerator)
        {
            auto const& params = std::get<1>(inputParam);

            auto const& typeAB                                = std::get<0>(params);
            auto const& unrollK                               = std::get<1>(params);
            auto const& loadPathA                             = std::get<2>(params);
            auto const& loadPathB                             = std::get<3>(params);
            auto const& storeLDSD                             = std::get<4>(params);
            auto const& mode                                  = std::get<5>(params);
            auto const& [prefetchInFlight, prefetchLDSFactor] = std::get<7>(params);

            // Prefetch requires LDS
            if((prefetchInFlight > 0) and (loadPathA == LP::BufferToVGPR)
               and (loadPathB == LP::BufferToVGPR))
            {
                continue;
            }

            // Prefetch requires unrollK > 1
            if(prefetchInFlight > 0 and unrollK < 2)
            {
                continue;
            }

            // Runs out of LDS
            if((typeAB == DT::Float) and (loadPathA == LP::BufferToLDSViaVGPR)
               and (loadPathB == LP::BufferToLDSViaVGPR) and (storeLDSD) and (mode != SM::Standard))
            {
                continue;
            }

            // Runs out of VGPRs: Float + both LDS paths + non-Standard StreamK + prefetch
            if((typeAB == DT::Float) and (loadPathA == LP::BufferToLDSViaVGPR)
               and (loadPathB == LP::BufferToLDSViaVGPR) and (mode != SM::Standard)
               and (prefetchInFlight > 0))
            {
                continue;
            }

            filtered.push_back(inputParam);
        }

        return ::testing::ValuesIn(filtered);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMTest,
        StreamKTestGPU,
        FilterValidStreamKParams(::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(rocRoller::DataType::Float,
                                                 rocRoller::DataType::Half),
                               ::testing::Values(1, 2), // unrollK
                               ::testing::Values(SolutionParams::LoadPath::BufferToVGPR,
                                                 SolutionParams::LoadPath::BufferToLDSViaVGPR),
                               ::testing::Values(SolutionParams::LoadPath::BufferToVGPR,
                                                 SolutionParams::LoadPath::BufferToLDSViaVGPR),
                               ::testing::Values(true, false), // storeLDSD
                               ::testing::Values(rocRoller::StreamKMode::Standard,
                                                 rocRoller::StreamKMode::TwoTile,
                                                 rocRoller::StreamKMode::TwoTileDPFirst),
                               ::testing::Values(true, false), // betaZero
                               ::testing::Values(PrefetchConfig{0, 0},
                                                 PrefetchConfig{1, 0},
                                                 PrefetchConfig{1, 2},
                                                 PrefetchConfig{2, 0},
                                                 PrefetchConfig{2, 2})))));

} // namespace GEMMTests
