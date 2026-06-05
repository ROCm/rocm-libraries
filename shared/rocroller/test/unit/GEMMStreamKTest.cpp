// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

    // ProblemConfig: (dataTypeAB, macM, macN, macK, m, n, k, numWGs)
    using ProblemConfig = std::tuple<rocRoller::DataType, int, int, int, int, int, int, int>;

    // ========================================================================
    // GEMMStreamKMultipleFixupsTestSuite
    // ========================================================================

    // Params: ProblemConfig, StreamKMode, loadPathA, loadPathB, storePath
    class GEMMStreamKMultipleFixupsTestSuite
        : public BaseGEMMContextFixture<std::tuple<ProblemConfig,
                                                   StreamKMode,
                                                   SolutionParams::LoadPath, /* loadPathA */
                                                   SolutionParams::LoadPath, /* loadPathB */
                                                   SolutionParams::StorePath /* storePath */>>
    {
    };

    TEST_P(GEMMStreamKMultipleFixupsTestSuite, GPU_GEMM_StreamK_MultipleFixups)
    {
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasWMMA))
        {
            GTEST_SKIP() << "Skipping GEMMStreamKMultipleFixupsTestSuite on architecture "
                         << m_context->targetArchitecture().target().toString();
        }

        auto [problemConfig, mode, loadPathA, loadPathB, storePath] = std::get<1>(GetParam());
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
        gemm.storePath = storePath;

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

    INSTANTIATE_TEST_SUITE_P(
        GEMMStreamKTest,
        GEMMStreamKMultipleFixupsTestSuite,
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
                ::testing::Values(
                    SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer,
                    SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer) /* storePath */
                )));

    // ========================================================================
    // GEMMStreamKWGMTestSuite
    // ========================================================================

    // Params: workgroupMapping dim, workgroupMapping value, workgroupRemapXCC, StreamKMode, loadPathA, loadPathB
    class GEMMStreamKWGMTestSuite
        : public BaseGEMMContextFixture<std::tuple<int, /* workgroupMapping dim */
                                                   int, /* workgroupMapping value */
                                                   bool, /* workgroupRemapXCC */
                                                   StreamKMode,
                                                   SolutionParams::LoadPath, /* loadPathA */
                                                   SolutionParams::LoadPath /* loadPathB */>>
    {
    };

    TEST_P(GEMMStreamKWGMTestSuite, GPU_GEMM_StreamK_WorkgroupMapping)
    {
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasWMMA))
        {
            GTEST_SKIP() << "Skipping StreamKWGMTestGPU on architecture "
                         << m_context->targetArchitecture().target().toString();
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

    INSTANTIATE_TEST_SUITE_P(
        GEMMStreamKTest,
        GEMMStreamKWGMTestSuite,
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

    // ========================================================================
    // GEMMStreamKTestSuite
    // ========================================================================

    // PrefetchConfig: (prefetchInFlight, prefetchLDSFactor)
    using PrefetchConfig = std::tuple<int, int>;

    // Params: typeAB, unrollK, loadPathA, loadPathB, storePath, mode, betaZero, prefetchConfig
    class GEMMStreamKTestSuite : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                                          int,
                                                                          SolutionParams::LoadPath,
                                                                          SolutionParams::LoadPath,
                                                                          SolutionParams::StorePath,
                                                                          rocRoller::StreamKMode,
                                                                          bool,
                                                                          PrefetchConfig>>
    {
    };

    TEST_P(GEMMStreamKTestSuite, GPU_GEMM_StreamK_Parameterized)
    {
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasWMMA))
        {
            GTEST_SKIP() << "Skipping StreamKTestGPU on architecture "
                         << m_context->targetArchitecture().target().toString();
        }

        auto [typeAB, unrollK, loadPathA, loadPathB, storePath, mode, betaZero, prefetchConfig]
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
        gemm.storePath = storePath;
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
        GEMMStreamKMultipleFixupsTestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(
                    // ProblemConfig: (dataTypeAB, macM, macN, macK, m, n, k, numWGs)
                    ProblemConfig{rocRoller::DataType::Half, 128, 128, 16, 128, 256, 15936, 128},
                    ProblemConfig{rocRoller::DataType::Float, 128, 128, 64, 128, 128, 4096, 128},
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
                ::testing::Values(
                    SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer,
                    SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer) /* storePath */
                )));

    using StreamKParamGenerator
        = ::testing::internal::ParamGenerator<GEMMStreamKTestSuite::ParamType>;
    static auto FilterValidStreamKParams(StreamKParamGenerator&& inputParamGenerator)
    {
        using LP = SolutionParams::LoadPath;
        using DT = rocRoller::DataType;
        using SM = rocRoller::StreamKMode;

        std::vector<GEMMStreamKTestSuite::ParamType> filtered;
        for(auto const& inputParam : inputParamGenerator)
        {
            auto const& params = std::get<1>(inputParam);

            auto const& typeAB                                = std::get<0>(params);
            auto const& unrollK                               = std::get<1>(params);
            auto const& loadPathA                             = std::get<2>(params);
            auto const& loadPathB                             = std::get<3>(params);
            auto const& storePath                             = std::get<4>(params);
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
               and (loadPathB == LP::BufferToLDSViaVGPR) and (SolutionParams::IsLDSStore(storePath))
               and (mode != SM::Standard))
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
        GEMMStreamKTest,
        GEMMStreamKTestSuite,
        FilterValidStreamKParams(::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(rocRoller::DataType::Float, rocRoller::DataType::Half),
                ::testing::Values(1, 2), // unrollK
                ::testing::Values(SolutionParams::LoadPath::BufferToVGPR,
                                  SolutionParams::LoadPath::BufferToLDSViaVGPR),
                ::testing::Values(SolutionParams::LoadPath::BufferToVGPR,
                                  SolutionParams::LoadPath::BufferToLDSViaVGPR),
                ::testing::Values(
                    SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer,
                    SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer), // storePath
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
