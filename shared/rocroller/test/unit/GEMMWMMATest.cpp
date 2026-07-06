// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include "GEMMF8F6F4.hpp"
#include "GEMMTestBase.hpp"

#include <rocRoller/KernelOptions_detail.hpp>

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
        case 4:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x4_f32);
            break;
        case 16:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x16_f16);
            break;
        case 32:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x32_f16);
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
        else if(typeAB == DataType::Float)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x4_f32);
            basicGEMM<float, float, float>(gemm);
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

    INSTANTIATE_TEST_SUITE_P(
        GEMMTestWMMA1250,
        GEMMWMMATestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(std::make_pair(rocRoller::DataType::Half, /*waveK*/ 32),
                                  std::make_pair(rocRoller::DataType::BFloat16, /*waveK*/ 32),
                                  std::make_pair(rocRoller::DataType::Float, /*waveK*/ 4)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR))));

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
        case 32:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f16_16x16x32_f16);
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

    INSTANTIATE_TEST_SUITE_P(
        GEMMTestWMMA1250,
        GEMMWMMAF16AccumTestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(std::make_pair(rocRoller::DataType::Half, /*waveK*/ 32),
                                  std::make_pair(rocRoller::DataType::BFloat16, /*waveK*/ 32)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR))));

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

        KernelOptions options{m_context->kernelOptions()};
        options->favourF8F6F4OverF8MatrixInstruction = false;
        setKernelOptions(options);

        GEMMProblem gemm;

        switch(waveK)
        {
        case 16:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x16_f8);
            break;
        case 64:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x64_f8);
            break;
        case 128:
            REQUIRE_ARCH_REVISION_ID(1);
            gemm.macK = waveK * 4;
            gemm.k    = gemm.macK * 4;
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x128_f8);
            break;
        default:
            Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
        }

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

    INSTANTIATE_TEST_SUITE_P(
        GEMMWMMA1250Test,
        MixedGEMMWMMATestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(/*waveK*/ 64, /*waveK*/ 128),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR))));

    // ========================================================================
    // MixedGEMMWMMAF8F6F4TestSuite
    // ========================================================================

    // Params are: A type, B type, K tile size, (transA, transB)
    class MixedGEMMWMMAF8F6F4TestSuite
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                   rocRoller::DataType,
                                                   int,
                                                   std::pair<std::string, std::string>>>
    {
    };

    TEST_P(MixedGEMMWMMAF8F6F4TestSuite, GPU_GEMM_WMMA_F8F6F4)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f8f6f4);
        auto [typeA, typeB, waveK, transOp] = std::get<1>(GetParam());
        AssertFatal(waveK == 128, "Invalid waveK value.", ShowValue(waveK));

        auto gemm = GEMMProblemF8F6F4(16, 16, waveK);
        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        // TODO: change setup_GEMMF8F6F4 to query wavefrontSize
        gemm.workgroupSizeX                = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY                = 2;
        std::tie(gemm.transA, gemm.transB) = transOp;

        basicGEMMMixed(typeA, typeB, gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMWMMA1250Test,
        MixedGEMMWMMAF8F6F4TestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(/*waveK*/ 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    // ========================================================================
    // ScaledMixedGEMMWMMAF8F6F4TestSuite
    // ========================================================================

    // Params are: A type, B type, scaleTypeA, scaleTypeB, K tile size, (scaleAMode, scaleBMode, scaleBlockSize), (transA, transB)
    class ScaledMixedGEMMWMMAF8F6F4TestSuite
        : public BaseGEMMContextFixture<std::tuple<
              rocRoller::DataType,
              rocRoller::DataType,
              rocRoller::DataType,
              rocRoller::DataType,
              int,
              std::tuple<rocRoller::Operations::ScaleMode, rocRoller::Operations::ScaleMode, int>,
              std::pair<std::string, std::string>>>
    {
    };

    TEST_P(ScaledMixedGEMMWMMAF8F6F4TestSuite, GPU_GEMM_WMMA_SCALED_F8F6F4)
    {
        REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasWMMA_scale_f8f6f4);
        auto [typeA, typeB, scaleTypeA, scaleTypeB, waveK, scaleModesAndSize, transOp]
            = std::get<1>(GetParam());

        AssertFatal(waveK == 128, "Invalid waveK value.", ShowValue(waveK));

        auto gemm = GEMMProblemF8F6F4(16, 16, waveK);

        std::tie(gemm.transA, gemm.transB) = transOp;

        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 2;
        gemm.scaleTypeA     = scaleTypeA;
        gemm.scaleTypeB     = scaleTypeB;

        std::tie(gemm.scaleAMode, gemm.scaleBMode, gemm.scaleBlockSize) = scaleModesAndSize;

        basicGEMMMixed(typeA, typeB, gemm);
    }

    TEST_P(ScaledMixedGEMMWMMAF8F6F4TestSuite, GPU_GEMM_WMMA_SCALED_F8F6F4_FLAT)
    {
        REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasWMMA_scale_f8f6f4);
        auto [typeA, typeB, scaleTypeA, scaleTypeB, waveK, scaleModesAndSize, transOp]
            = std::get<1>(GetParam());
        AssertFatal(waveK == 128, "Invalid waveK value.", ShowValue(waveK));

        auto gemm = GEMMProblemF8F6F4(16, 16, waveK);

        std::tie(gemm.transA, gemm.transB) = transOp;

        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        gemm.workgroupSizeX = gemm.wavefrontSize;
        gemm.workgroupSizeY = 1;
        gemm.scaleTypeA     = scaleTypeA;
        gemm.scaleTypeB     = scaleTypeB;

        std::tie(gemm.scaleAMode, gemm.scaleBMode, gemm.scaleBlockSize) = scaleModesAndSize;

        basicGEMMMixed(typeA, typeB, gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMWMMA1250Test,
        ScaledMixedGEMMWMMAF8F6F4TestSuite,
        filterValidDataTypeScaleTypeParams<ScaledMixedGEMMWMMAF8F6F4TestSuite::ParamType>(
            ::testing::Combine(
                currentGPUISA(),
                ::testing::Combine(
                    ::testing::Values(rocRoller::DataType::FP8,
                                      rocRoller::DataType::BF8,
                                      rocRoller::DataType::FP6,
                                      rocRoller::DataType::BF6,
                                      rocRoller::DataType::FP4),
                    ::testing::Values(rocRoller::DataType::FP8,
                                      rocRoller::DataType::BF8,
                                      rocRoller::DataType::FP6,
                                      rocRoller::DataType::BF6,
                                      rocRoller::DataType::FP4),
                    ::testing::Values(rocRoller::DataType::E8M0,
                                      rocRoller::DataType::E5M3,
                                      rocRoller::DataType::E4M3),
                    ::testing::Values(rocRoller::DataType::E8M0,
                                      rocRoller::DataType::E5M3,
                                      rocRoller::DataType::E4M3),
                    ::testing::Values(/*waveK*/ 128),
                    ::testing::Values(/*scaleAMode, scaleBMode, scaleBlockSize*/
                                      std::tuple<Operations::ScaleMode, Operations::ScaleMode, int>(
                                          Operations::ScaleMode::Separate,
                                          Operations::ScaleMode::Separate,
                                          32),
                                      std::tuple<Operations::ScaleMode, Operations::ScaleMode, int>(
                                          Operations::ScaleMode::Separate,
                                          Operations::ScaleMode::Separate,
                                          16),
                                      std::tuple<Operations::ScaleMode, Operations::ScaleMode, int>(
                                          Operations::ScaleMode::Separate,
                                          Operations::ScaleMode::SingleScale,
                                          32),
                                      std::tuple<Operations::ScaleMode, Operations::ScaleMode, int>(
                                          Operations::ScaleMode::SingleScale,
                                          Operations::ScaleMode::Separate,
                                          32),
                                      std::tuple<Operations::ScaleMode, Operations::ScaleMode, int>(
                                          Operations::ScaleMode::SingleScale,
                                          Operations::ScaleMode::SingleScale,
                                          32)),
                    ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                      std::pair<std::string, std::string>("N", "T"),
                                      std::pair<std::string, std::string>("T", "N"),
                                      std::pair<std::string, std::string>("T", "T"))))));

    // ========================================================================
    // GEMMWMMAClustersTestSuite
    // ========================================================================

    class GEMMWMMAClustersTestSuite : public BaseGEMMContextFixture<std::array<unsigned int, 3>>
    {
    };

    TEST_P(GEMMWMMAClustersTestSuite, GPU_GEMM_CLUSTERS)
    {
#ifndef ROCROLLER_HAS_HIP_WORKGROUP_CLUSTERS
        GTEST_SKIP() << "Workgroup cluster feature is disabled: the installed ROCm/HIP version "
                        "does not support hipLaunchAttributeClusterDimension.";
#endif
        REQUIRE_ARCH_CAP(GPUCapability::HasWorkgroupClusters);
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x32_f16);

        GEMMProblem gemm;
        gemm.m     = 2048;
        gemm.n     = 2048;
        gemm.k     = 128;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = 32;
        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;

        auto clusterSize    = std::get<1>(GetParam());
        auto numWorkgroupsX = gemm.n / static_cast<uint>(gemm.macN);
        auto numWorkgroupsY = gemm.m / static_cast<uint>(gemm.macM);

        AssertFatal(WorkgroupClustersDetail::IsValidWorkgroupClusterSize(
                        clusterSize, {numWorkgroupsX, numWorkgroupsY, 1}),
                    "Invalid workgroup cluster sizes",
                    ShowValue(clusterSize),
                    ShowValue(numWorkgroupsX),
                    ShowValue(numWorkgroupsY));

        gemm.workgroupClusterSizeX = clusterSize[0];
        gemm.workgroupClusterSizeY = clusterSize[1];
        gemm.workgroupClusterSizeZ = clusterSize[2];

        basicGEMM<Half, Half, float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMWMMA1250Test,
        GEMMWMMAClustersTestSuite,
        ::testing::Combine(currentGPUISA(),
                           ::testing::ValuesIn(WorkgroupClustersDetail::ValidWorkgroupClusterSizes(
                               {WorkgroupClustersDetail::MaxWorkgroupsPerCluster,
                                WorkgroupClustersDetail::MaxWorkgroupsPerCluster,
                                1}))));

    // ========================================================================
    // GEMMWMMAF4TestSuite
    // ========================================================================

    // Params are: (transA, transB)
    class GEMMWMMAF4TestSuite : public BaseGEMMContextFixture<std::pair<std::string, std::string>>
    {
    };

    TEST_P(GEMMWMMAF4TestSuite, GPU_GEMM_WMMA_F4)
    {
        REQUIRE_ARCH_REVISION_ID(1);
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_32x16x128_f4);
        auto transOp = std::get<1>(GetParam());

        auto gemm = GEMMProblemF8F6F4(32, 16, 128);
        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        gemm.workgroupSizeX                = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY                = 2;
        std::tie(gemm.transA, gemm.transB) = transOp;

        basicGEMM<FP4, FP4, float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMWMMA1250Rev1Test,
        GEMMWMMAF4TestSuite,
        ::testing::Combine(currentGPUISA(),
                           ::testing::Values(
                               // TODO: Adapt transpose from LDS coordinate
                               // transform to work with _f4 layout
                               // std::pair<std::string, std::string>("N", "N"),
                               // std::pair<std::string, std::string>("N", "T"),
                               std::pair<std::string, std::string>("T", "N")
                               /*std::pair<std::string, std::string>("T", "T")*/)));

    // ========================================================================
    // ScaledGEMMWMMAF4TestSuite
    // ========================================================================

    // Params are: A type, B type, scaleTypeA, scaleTypeB, (scaleAMode, scaleBMode, scaleBlockSize), (transA, transB)
    class ScaledGEMMWMMAF4TestSuite
        : public BaseGEMMContextFixture<std::tuple<
              rocRoller::DataType,
              rocRoller::DataType,
              rocRoller::DataType,
              rocRoller::DataType,
              std::tuple<rocRoller::Operations::ScaleMode, rocRoller::Operations::ScaleMode, int>,
              std::pair<std::string, std::string>>>
    {
    };

    TEST_P(ScaledGEMMWMMAF4TestSuite, GPU_GEMM_WMMA_SCALED_F4)
    {
        REQUIRE_ARCH_REVISION_ID(1);
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_scale_32x16x128_f4);
        auto [typeA, typeB, scaleTypeA, scaleTypeB, scaleModesAndSize, transOp]
            = std::get<1>(GetParam());

        auto gemm                          = GEMMProblemF8F6F4(32, 16, 128);
        std::tie(gemm.transA, gemm.transB) = transOp;

        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 2;
        gemm.scaleTypeA     = scaleTypeA;
        gemm.scaleTypeB     = scaleTypeB;

        std::tie(gemm.scaleAMode, gemm.scaleBMode, gemm.scaleBlockSize) = scaleModesAndSize;

        if(gemm.scaleBlockSize == 32)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);
        }
        else if(gemm.scaleBlockSize == 16)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling16);
        }
        else
        {
            Throw<FatalError>(fmt::format("Unsupported scaleBlockSize: {}. (Allowed 16, 32)",
                                          gemm.scaleBlockSize));
        }

        basicGEMMMixed(typeA, typeB, gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMWMMA1250Rev1Test,
        ScaledGEMMWMMAF4TestSuite,
        filterValidDataTypeScaleTypeParams<ScaledGEMMWMMAF4TestSuite::ParamType>(::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP4),
                               ::testing::Values(rocRoller::DataType::FP4),
                               ::testing::Values(rocRoller::DataType::E8M0,
                                                 rocRoller::DataType::E5M3,
                                                 rocRoller::DataType::E4M3),
                               ::testing::Values(rocRoller::DataType::E8M0,
                                                 rocRoller::DataType::E5M3,
                                                 rocRoller::DataType::E4M3),
                               ::testing::Values(/*scaleAMode, scaleBMode, scaleBlockSize*/
                                                 std::make_tuple(Operations::ScaleMode::Separate,
                                                                 Operations::ScaleMode::Separate,
                                                                 32),
                                                 std::make_tuple(Operations::ScaleMode::Separate,
                                                                 Operations::ScaleMode::Separate,
                                                                 16),
                                                 std::make_tuple(Operations::ScaleMode::Separate,
                                                                 Operations::ScaleMode::SingleScale,
                                                                 32),
                                                 std::make_tuple(Operations::ScaleMode::SingleScale,
                                                                 Operations::ScaleMode::Separate,
                                                                 32),
                                                 std::make_tuple(Operations::ScaleMode::SingleScale,
                                                                 Operations::ScaleMode::SingleScale,
                                                                 32)),
                               ::testing::Values(
                                   //std::pair<std::string, std::string>("N", "N"),
                                   //std::pair<std::string, std::string>("N", "T"),
                                   std::pair<std::string, std::string>("T", "N")
                                   //std::pair<std::string, std::string>("T", "T")
                                   )))));

    // ========================================================================
    // MixedGEMMWMMAF8F6F4TDMTestSuite
    // ========================================================================

    // Params are: A type, B type, K tile size, (transA, transB), (useTDMToLoadA, useTDMToLoadB)
    class MixedGEMMWMMAF8F6F4TDMTestSuite
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                   rocRoller::DataType,
                                                   int,
                                                   std::pair<std::string, std::string>,
                                                   std::pair<bool, bool>>>
    {
    };

    TEST_P(MixedGEMMWMMAF8F6F4TDMTestSuite, GPU_GEMM_WMMA_F8F6F4_TDM)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasTDM);
        auto [typeA, typeB, waveK, transOp, useTDMOp] = std::get<1>(GetParam());
        AssertFatal(waveK == 128, "Invalid waveK value.", ShowValue(waveK));

        auto gemm = GEMMProblemF8F6F4{16, 16, waveK};
        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        // TODO: change setup_GEMMF8F6F4 to query wavefrontSize
        gemm.workgroupSizeX                = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY                = 2;
        std::tie(gemm.transA, gemm.transB) = transOp;

        auto [useTDMToLoadA, useTDMToLoadB] = useTDMOp;

        gemm.loadPathA = useTDMToLoadA ? SolutionParams::LoadPath::TDMToLDS
                                       : SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = useTDMToLoadB ? SolutionParams::LoadPath::TDMToLDS
                                       : SolutionParams::LoadPath::BufferToLDSViaVGPR;

        basicGEMMMixed(typeA, typeB, gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMWMMA1250Test,
        MixedGEMMWMMAF8F6F4TDMTestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(/*waveK*/ 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")),
                               // useTDMToLoadA, useTDMToLoadB
                               ::testing::Values(std::pair<bool, bool>(true, false),
                                                 std::pair<bool, bool>(false, true),
                                                 std::pair<bool, bool>(true, true)))));
} // namespace GEMMTests
