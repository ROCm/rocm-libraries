// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Utils.hpp>

#include "GEMMTestBase.hpp"

namespace GEMMTests
{
    using namespace rocRoller;

    GEMMProblem SetupGEMMF16(uint waveM, uint waveN, uint waveK)
    {
        GEMMProblem gemm;

        // 1x4 jamming
        uint wavesPerWGX = 4;
        uint wavesPerWGY = 4;

        gemm.waveM = waveM;
        gemm.waveN = waveN;
        gemm.waveK = waveK;

        gemm.macM = wavesPerWGX * gemm.waveM;
        gemm.macN = wavesPerWGY * gemm.waveN;
        gemm.macK = 2 * gemm.waveK;

        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;

        gemm.workgroupSizeX = 256;
        gemm.workgroupSizeY = 1;

        gemm.m = 2 * gemm.macM;
        gemm.n = 3 * gemm.macN;
        gemm.k = 4 * gemm.macK;

        gemm.alpha = 2.1;
        gemm.beta  = 0.75;

        gemm.transA = "N";
        gemm.transB = "T";
        return gemm;
    }

    void CheckGEMMF16(rocRoller::ContextPtr m_context,
                      std::string           mfma,
                      uint                  numMFMAs,
                      uint                  numBufferAndGlobalLoads,
                      uint                  numDSWrites,
                      uint                  numDSReads,
                      uint                  numTrLoads)
    {
        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "v_mfma"), numMFMAs);
        EXPECT_EQ(countSubstring(generatedCode, mfma), numMFMAs);

        const auto allBufferLoads     = countSubstring(generatedCode, "buffer_load");
        const auto allGlobalLoads     = countSubstring(generatedCode, "global_load");
        const auto dwordX4BufferLoads = countSubstring(generatedCode, "buffer_load_dwordx4 ");
        const auto dwordX4GlobalLoads = countSubstring(generatedCode, "global_load_dwordx4 ");
        const auto dwordBufferLoads   = countSubstring(generatedCode, "buffer_load_dword ");
        const auto dwordGlobalLoads   = countSubstring(generatedCode, "global_load_dword ");

        EXPECT_EQ(allBufferLoads + allGlobalLoads, numBufferAndGlobalLoads);
        EXPECT_EQ(dwordX4BufferLoads + dwordX4GlobalLoads, numBufferAndGlobalLoads);
        EXPECT_EQ(dwordBufferLoads + dwordGlobalLoads, 0);

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b"), numDSWrites);
        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128 "), numDSWrites);

        EXPECT_EQ(countSubstring(generatedCode, "ds_read_b64_tr_b"), numTrLoads);

        EXPECT_EQ(countSubstring(generatedCode, "ds_read"), numDSReads + numTrLoads);
        EXPECT_EQ(countSubstring(generatedCode, "ds_read_b128 "), numDSReads);
    }

    // ========================================================================
    // GEMMF16TestSuite
    // ========================================================================

    // Params are: A & B type, K tile size, (transA, transB), loadPathA, and loadPathB
    class GEMMF16TestSuite
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                   int,
                                                   std::pair<std::string, std::string>,
                                                   SolutionParams::LoadPath,
                                                   SolutionParams::LoadPath>>
    {
    };

    TEST_P(GEMMF16TestSuite, GPU_GEMM_DataType_BF16_FP32_32x32x4)
    {
        GEMMProblem gemm;
        gemm.waveM = 32;
        gemm.waveN = 32;
        gemm.waveK = 4;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x4);
        basicGEMM<BFloat16, BFloat16, float>(gemm);
    }

    TEST_P(GEMMF16TestSuite, GPU_GEMM_DataType_BF16_BF16_32x32x4)
    {
        GEMMProblem gemm;
        gemm.waveM = 32;
        gemm.waveN = 32;
        gemm.waveK = 4;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x4);
        basicGEMM<BFloat16, BFloat16, BFloat16>(gemm);
    }

    TEST_P(GEMMF16TestSuite, GPU_GEMM_DataType_BF16_FP32_16x16x8)
    {
        GEMMProblem gemm;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = 8;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x8);
        basicGEMM<BFloat16, BFloat16, float>(gemm);
    }

    TEST_P(GEMMF16TestSuite, GPU_GEMM_DataType_BF16_FP32_16x16x16)
    {
        GEMMProblem gemm;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = 16;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x16_1k);
        basicGEMM<BFloat16, BFloat16, float>(gemm);
    }

    TEST_P(GEMMF16TestSuite, GPU_GEMM_DataType_BF16_BF16_16x16x16)
    {
        GEMMProblem gemm;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = 16;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x16_1k);
        basicGEMM<BFloat16, BFloat16, float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMF16Test,
        GEMMF16TestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(rocRoller::DataType::Half,
                                                 rocRoller::DataType::BFloat16),
                               ::testing::Values(16, 32),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")),
                               ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR),
                               ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR))));

    // ========================================================================
    // GEMMF16NTTestSuite
    // ========================================================================

    // Params are: loadPathA and loadPathB
    class GEMMF16NTTestSuite : public BaseGEMMContextFixture<
                                   std::tuple<SolutionParams::LoadPath, SolutionParams::LoadPath>>
    {
    };

    TEST_P(GEMMF16TestSuite, GPU_GEMM_DataType_FP16_Parameterized)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto [typeAB, MFMAK, transOp, loadPathA, loadPathB] = std::get<1>(GetParam());

        uint const waveM = (MFMAK == 32) ? 16 : 32;
        uint const waveN = (MFMAK == 32) ? 16 : 32;
        uint const waveK = MFMAK;

        auto problem = SetupGEMMF16(waveM, waveN, waveK);

        problem.loadPathA = loadPathA;
        problem.loadPathB = loadPathB;

        std::tie(problem.transA, problem.transB) = transOp;

        std::string typeStr{"f16"};

        switch(typeAB)
        {
        case DataType::Half:
            if(waveK == 32)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_16x16x32_f16);
            }
            else
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_32x32x16_f16);
            }
            basicGEMM<Half, Half, float>(problem);
            break;
        case DataType::BFloat16:
            if(waveK == 32)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_16x16x32_bf16);
            }
            else
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_32x32x16_bf16);
            }
            basicGEMM<BFloat16, BFloat16, float>(problem);
            typeStr = "bf16";
            break;
        default:
            Throw<FatalError>(fmt::format("Unexpected data type: {}. (Allowed: Half and Bfloat16)",
                                          toString(typeAB)));
        }

        uint const wfs           = problem.wavefrontSize;
        uint const wgX           = problem.workgroupSizeX;
        uint const wgY           = problem.workgroupSizeY;
        uint const numDWavetiles = problem.macM * problem.macN / (waveM * waveN);
        uint const numWaves      = wgX * wgY / wfs;

        uint const numDWavetilesPerWave = numDWavetiles / numWaves;
        uint const numMFMAsPerWave      = problem.macK / waveK;
        uint const numMFMAs             = numDWavetilesPerWave * numMFMAsPerWave;

        auto const& arch                = m_context->targetArchitecture();
        uint const  elementsPerWavetile = waveM * waveK / wfs;
        uint const  elementBits         = DataTypeInfo::Get(typeAB).elementBits;
        uint const  elementsPerTrLoad   = bitsPerTransposeLoad(arch, elementBits) / elementBits;

        uint const bitsPerWavetileLoad = elementsPerWavetile * elementBits;

        uint const trLoadsPerWave = elementsPerWavetile / elementsPerTrLoad;
        uint const dsLoadsPerWave = elementsPerWavetile / (bitsPerWavetileLoad / elementBits);

        // uint const bitsLoadedForAB
        //     = numDWavetilesPerWave * /*A & B*/ 2 * waveM * waveN * elementBits;
        uint const bitsLoadedForAB
            = (/*A*/ waveM * problem.macK + /*B*/ problem.macK * waveN) * elementBits;

        uint const elementBitsC   = DataTypeInfo::Get(DataType::Float).elementBits;
        uint const bitsLoadedForC = numDWavetilesPerWave * waveM * waveN * elementBitsC;

        // uint const numBufferLoads = (bitsLoadedForAB + bitsLoadedForC) / bitsPerWavetileLoad / wfs;
        // uint const numDSWrites    = bitsLoadedForAB / bitsPerWavetileLoad / wfs;

        uint const numBufferLoadsForC  = bitsLoadedForC / bitsPerWavetileLoad / wfs;
        uint const numDSWrites         = bitsLoadedForAB / bitsPerWavetileLoad / wfs;
        uint const numBufferLoadsForAB = numDSWrites;

        uint numTrLoads = 0;
        uint numDSReads = 0;
        { // 1x4 jamming = 4 tiles. Each tile of A gets multiplied by 4 tiles of B.
            if(problem.transA == "T")
                numDSReads += /*number of A tiles*/ 1 * numMFMAsPerWave * dsLoadsPerWave;
            if(problem.transB == "N")
                numDSReads += /*number of B tiles*/ 4 * numMFMAsPerWave * dsLoadsPerWave;

            if(problem.transA == "N")
                numTrLoads += /*number of A tiles*/ 1 * numMFMAsPerWave * trLoadsPerWave;
            if(problem.transB == "T")
                numTrLoads += /*number of B tiles*/ 4 * numMFMAsPerWave * trLoadsPerWave;
        }

        auto const mfma{fmt::format("v_mfma_f32_{}x{}x{}_{}", waveM, waveN, waveK, typeStr)};

        CheckGEMMF16(m_context,
                     mfma,
                     numMFMAs,
                     numBufferLoadsForC + numBufferLoadsForAB,
                     numDSWrites,
                     numDSReads,
                     numTrLoads);
    }

    TEST_P(GEMMF16NTTestSuite, GPU_GEMM_DataType_FP16_32x32x8)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.waveM = 32;
        gemm.waveN = 32;
        gemm.waveK = 8;

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMF16NTTestSuite, GPU_GEMM_DataType_BF16_BF16_32x32x8)
    {
        GEMMProblem gemm;
        gemm.waveM = 32;
        gemm.waveN = 32;
        gemm.waveK = 8;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x8_1k);
        basicGEMM<BFloat16, BFloat16, BFloat16>(gemm);
    }

    TEST_P(GEMMF16NTTestSuite, GPU_GEMM_DataType_BF16_FP32_32x32x8)
    {
        GEMMProblem gemm;
        gemm.waveM = 32;
        gemm.waveN = 32;
        gemm.waveK = 8;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x8_1k);
        basicGEMM<BFloat16, BFloat16, float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMF16Test,
        GEMMF16NTTestSuite,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR),
                               ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR))));

} // namespace GEMMTests
