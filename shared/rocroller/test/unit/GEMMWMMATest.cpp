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

    // ========================================================================
    // GEMMWMMATestSuite
    // ========================================================================

    // Params are: A & B type, K tile size, (transA, transB), loadPathA, loadPathB
    class GEMMWMMATestSuite
        : public BaseGEMMContextFixture<std::tuple<std::pair<rocRoller::DataType, int>,
                                                   std::pair<std::string, std::string>,
                                                   SolutionParams::LoadPath,
                                                   SolutionParams::LoadPath>>
    {
    };

    TEST_P(GEMMWMMATestSuite, GPU_GEMM_WMMA_Basic)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA);
        auto [typeABAndWaveK, transOp, loadPathA, loadPathB] = std::get<1>(GetParam());
        auto [typeAB, waveK]                                 = typeABAndWaveK;

        switch(waveK)
        {
        case 16:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x16_f16);
            break;
        default:
            Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
        }

        GEMMProblem gemm;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = waveK;
        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        std::tie(gemm.transA, gemm.transB) = transOp;
        gemm.loadPathA                     = loadPathA;
        gemm.loadPathB                     = loadPathB;

        if(typeAB == DataType::Half)
        {
            basicGEMM<Half, Half, float>(gemm);
        }
        else if(typeAB == DataType::BFloat16)
        {
            basicGEMM<BFloat16, BFloat16, float>(gemm);
        }
        else
        {
            Throw<FatalError>("Invalid type.", ShowValue(typeAB));
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMWMMATest,
        GEMMWMMATestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(std::make_pair(rocRoller::DataType::Half, /*waveK*/ 16),
                                  std::make_pair(rocRoller::DataType::BFloat16, /*waveK*/ 16)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                  SolutionParams::LoadPath::GlobalToLDSViaVGPR),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                  SolutionParams::LoadPath::GlobalToLDSViaVGPR))));

    // ========================================================================
    // GEMMWMMAF16AccumTestSuite
    // ========================================================================

    // Params are: A & B type, K tile size, (transA, transB), loadPathA, loadPathB
    class GEMMWMMAF16AccumTestSuite
        : public BaseGEMMContextFixture<std::tuple<std::pair<rocRoller::DataType, int>,
                                                   std::pair<std::string, std::string>,
                                                   SolutionParams::LoadPath,
                                                   SolutionParams::LoadPath>>
    {
    };

    TEST_P(GEMMWMMAF16AccumTestSuite, GPU_GEMM_WMMA_F16Accum)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_F16_ACC);
        auto [dataTypeAndWaveK, transOp, loadPathA, loadPathB] = std::get<1>(GetParam());
        auto [dataType, waveK]                                 = dataTypeAndWaveK;

        switch(waveK)
        {
        case 16:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f16_16x16x16_f16);
            break;
        default:
            Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
        }

        GEMMProblem gemm;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = waveK;
        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        std::tie(gemm.transA, gemm.transB) = transOp;
        gemm.loadPathA                     = loadPathA;
        gemm.loadPathB                     = loadPathB;

        if(dataType == DataType::Half)
        {
            basicGEMM<Half, Half, Half, Half, Half>(gemm);
        }
        else if(dataType == DataType::BFloat16)
        {
            basicGEMM<BFloat16, BFloat16, BFloat16, BFloat16, BFloat16>(gemm);
        }
        else
        {
            Throw<FatalError>("Invalid type.", ShowValue(dataType));
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMWMMATest,
        GEMMWMMAF16AccumTestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(std::make_pair(rocRoller::DataType::Half, /*waveK*/ 16),
                                  std::make_pair(rocRoller::DataType::BFloat16, /*waveK*/ 16)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                  SolutionParams::LoadPath::GlobalToLDSViaVGPR),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                  SolutionParams::LoadPath::GlobalToLDSViaVGPR))));

    // ========================================================================
    // MixedGEMMWMMATestSuite
    // ========================================================================

    // Params are: A type, B type, K tile size, (transA, transB), loadPathA, loadPathB
    class MixedGEMMWMMATestSuite
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                   rocRoller::DataType,
                                                   int,
                                                   std::pair<std::string, std::string>,
                                                   SolutionParams::LoadPath,
                                                   SolutionParams::LoadPath>>
    {
    };

    TEST_P(MixedGEMMWMMATestSuite, GPU_GEMM_WMMA_Mixed)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA);
        auto [typeA, typeB, waveK, transOp, loadPathA, loadPathB] = std::get<1>(GetParam());

        switch(waveK)
        {
        case 16:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x16_f8);
            break;
        default:
            Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
        }

        GEMMProblem gemm;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = waveK;
        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        std::tie(gemm.transA, gemm.transB) = transOp;
        gemm.loadPathA                     = loadPathA;
        gemm.loadPathB                     = loadPathB;

        basicGEMMMixed(typeA, typeB, gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMWMMATest,
        MixedGEMMWMMATestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(16),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                  SolutionParams::LoadPath::GlobalToLDSViaVGPR),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                  SolutionParams::LoadPath::GlobalToLDSViaVGPR))));
} // namespace GEMMTests
