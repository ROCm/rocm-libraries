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

#include "GEMMF8F6F4.hpp"
#include "GEMMTestBase.hpp"

namespace GEMMTests
{
    using namespace rocRoller;
    namespace SolutionParams = rocRoller::Parameters::Solution;

    class GEMMDirectLDSTestBasicGPU : public BaseGEMMContextFixture<>
    {
    };

    // Params are: A & B type, M tile size, (transA, transB), load A Path, load B Path
    class GEMMDirectLDSTestGPU
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                   int,
                                                   std::pair<std::string, std::string>,
                                                   SolutionParams::LoadPath,
                                                   SolutionParams::LoadPath>>
    {
    };

    // Params are: A & B type, K tile size, (transA, transB), DirectLDS A & B
    class GEMMDirectLDSF8F6F4TestGPU
        : public BaseGEMMContextFixture<
              std::tuple<rocRoller::DataType, int, std::pair<std::string, std::string>, bool, bool>>
    {
    };

    static void checkNumDwordx4(std::string generatedCode,
                                const int   numBitsPerElementAB,
                                const int   macM,
                                const int   macN,
                                const int   macK,
                                const int   workgroupSizeTotal)
    {
        auto const numBitsPerDwordx4   = 4 * 4 * 8;
        auto const numBitsPerElementC  = 32;
        auto const numBufferLoadsForAB = ((macM * macK + macN * macK) * numBitsPerElementAB)
                                         / workgroupSizeTotal / numBitsPerDwordx4;
        auto const numBufferLoadsForC
            = ((macM * macN) * numBitsPerElementC) / workgroupSizeTotal / numBitsPerDwordx4;

        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx4"),
                  numBufferLoadsForAB + numBufferLoadsForC);
    }

    TEST_P(GEMMDirectLDSF8F6F4TestGPU, GPU_BasicGEMM)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDirectToLds);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        auto [typeAB, MFMAK, transOp, directLDSA, directLDSB] = std::get<1>(GetParam());

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        auto problem = GEMMProblemF8F6F4{waveM, waveN, waveK};

        std::tie(problem.transA, problem.transB) = transOp;

        problem.loadPathA = directLDSA ? SolutionParams::LoadPath::BufferToLDS
                                       : SolutionParams::LoadPath::BufferToLDSViaVGPR;
        problem.loadPathB = directLDSB ? SolutionParams::LoadPath::BufferToLDS
                                       : SolutionParams::LoadPath::BufferToLDSViaVGPR;
        problem.storeLDSD = false;

        std::string modifiers{"cbsz:0b000 blgp:0b000"};

        auto const numBitsPerElementAB = DataTypeInfo::Get(typeAB).elementBits;

        switch(typeAB)
        {
        case DataType::FP8:
            basicGEMM<FP8, FP8, float>(problem);
            break;
        case DataType::BF8:
            basicGEMM<BF8, BF8, float>(problem);
            modifiers = "cbsz:0b001 blgp:0b001";
            break;
        case DataType::FP6:
            GTEST_SKIP() << "Test not yet supported for FP6" << std::endl;
            break;
        case DataType::BF6:
            GTEST_SKIP() << "Test not yet supported for BF6" << std::endl;
            break;
        case DataType::FP4:
            basicGEMM<FP4, FP4, float>(problem);
            modifiers = "cbsz:0b100 blgp:0b100";
            break;
        default:
            Throw<FatalError>(
                fmt::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }
        auto generatedCode = m_context->instructions()->toString();
        if(directLDSA && directLDSB)
        {
            EXPECT_EQ(generatedCode.find("ds_write"), std::string::npos);
        }
        checkNumDwordx4(generatedCode,
                        numBitsPerElementAB,
                        problem.macM,
                        problem.macN,
                        problem.macK,
                        problem.workgroupSizeX * problem.workgroupSizeY);
    }

    TEST_P(GEMMDirectLDSF8F6F4TestGPU, GPU_ScaledBasicGEMM)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDirectToLds);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto [typeAB, MFMAK, transOp, directLDSA, directLDSB] = std::get<1>(GetParam());

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        auto problem = GEMMProblemF8F6F4{waveM, waveN, waveK};

        std::tie(problem.transA, problem.transB) = transOp;

        problem.scaleAMode = Operations::ScaleMode::Separate;
        problem.scaleBMode = Operations::ScaleMode::Separate;

        problem.scaleTypeA = DataType::E8M0;
        problem.scaleTypeB = DataType::E8M0;

        problem.loadPathA = directLDSA ? SolutionParams::LoadPath::BufferToLDS
                                       : SolutionParams::LoadPath::BufferToLDSViaVGPR;
        problem.loadPathB = directLDSB ? SolutionParams::LoadPath::BufferToLDS
                                       : SolutionParams::LoadPath::BufferToLDSViaVGPR;
        problem.storeLDSD = false;

        problem.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        std::string modifiers{"cbsz:0b000 blgp:0b000"};

        auto const numBitsPerElementAB = DataTypeInfo::Get(typeAB).elementBits;

        switch(typeAB)
        {
        case DataType::FP8:
            basicGEMM<FP8, FP8, float>(problem);
            break;
        case DataType::BF8:
            basicGEMM<BF8, BF8, float>(problem);
            modifiers = "cbsz:0b001 blgp:0b001";
            break;
        case DataType::FP6:
            GTEST_SKIP() << "Test not yet supported for FP6" << std::endl;
            break;
        case DataType::BF6:
            GTEST_SKIP() << "Test not yet supported for BF6" << std::endl;
            break;
        case DataType::FP4:
            basicGEMM<FP4, FP4, float>(problem);
            modifiers = "cbsz:0b100 blgp:0b100";
            break;
        default:
            Throw<FatalError>(
                fmt::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }

        std::string generatedCode = m_context->instructions()->toString();
        if(directLDSA && directLDSB)
        {
            EXPECT_EQ(generatedCode.find("ds_write"), std::string::npos);
        }
        checkNumDwordx4(generatedCode,
                        numBitsPerElementAB,
                        problem.macM,
                        problem.macN,
                        problem.macK,
                        problem.workgroupSizeX * problem.workgroupSizeY);
    }

    TEST_P(GEMMDirectLDSTestBasicGPU, GPU_BasicGEMMFP32D2L)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDS;
        gemm.transA    = "T";
        gemm.transB    = "N";
        gemm.m         = 3072;
        gemm.n         = 4096;
        gemm.k         = 4096;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMDirectLDSTestBasicGPU, GPU_BasicGEMMFP32D2LPadded)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDirectToLds);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDS;
        gemm.storeLDSD = false;
        gemm.transA    = "T";
        gemm.transB    = "N";
        gemm.m         = 3072;
        gemm.n         = 4096;
        gemm.k         = 4096;

        // This kernel uses 32bit buffer_load instructions; and
        // therefore each workgroup loads 1024 bytes per instruction
        gemm.padA = {1024, 64};
        gemm.padB = {1024, 96};
        basicGEMM<float>(gemm);

        auto instructions    = m_context->instructions()->toString();
        auto ldsWriteStrides = direct2LDSWriteStrides(instructions);

        std::set<int> expectedLDSWriteStrides;
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasWiderDirectToLds))
        {
            expectedLDSWriteStrides = {4 * (1024 + 64), 4 * (1024 + 96)};
        }
        else
        {
            expectedLDSWriteStrides = {1024 + 64, 1024 + 96};
        }
        EXPECT_EQ(ldsWriteStrides, expectedLDSWriteStrides);
    }

    TEST_P(GEMMDirectLDSTestGPU, GPU_BasicGEMMFP32)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDirectToLds);

        auto [typeAB, tileSizeM, transOp, loadPathA, loadPathB] = std::get<1>(GetParam());
        AssertFatal(typeAB == DataType::Float,
                    fmt::format("Expected A & B type to be Float but got {}.", toString(typeAB)));

        GEMMProblem gemm;
        gemm.macM      = tileSizeM;
        gemm.transA    = transOp.first;
        gemm.transB    = transOp.second;
        gemm.loadPathA = loadPathA;
        gemm.loadPathB = loadPathB;
        gemm.storeLDSD = false;

        basicGEMM<float>(gemm);

        if(SolutionParams::IsBufferToLDS(loadPathA) && SolutionParams::IsBufferToLDS(loadPathB))
        {
            auto generatedCode = m_context->instructions()->toString();
            EXPECT_EQ(generatedCode.find("ds_write"), std::string::npos);
        }
    }

    TEST_P(GEMMDirectLDSF8F6F4TestGPU, GPU_ScaledBasicGEMM_Prefetch2)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDirectToLds);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto [typeAB, MFMAK, transOp, directLDSA, directLDSB] = std::get<1>(GetParam());

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        auto problem = GEMMProblemF8F6F4{waveM, waveN, waveK};

        std::tie(problem.transA, problem.transB) = transOp;

        problem.scaleAMode = Operations::ScaleMode::Separate;
        problem.scaleBMode = Operations::ScaleMode::Separate;

        problem.scaleTypeA = DataType::E8M0;
        problem.scaleTypeB = DataType::E8M0;

        problem.loadPathA = directLDSA ? SolutionParams::LoadPath::BufferToLDS
                                       : SolutionParams::LoadPath::BufferToLDSViaVGPR;
        problem.loadPathB = directLDSB ? SolutionParams::LoadPath::BufferToLDS
                                       : SolutionParams::LoadPath::BufferToLDSViaVGPR;
        problem.storeLDSD = false;

        problem.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        problem.prefetch         = true;
        problem.prefetchInFlight = 2;
        problem.unrollK          = 2;

        std::string modifiers{"cbsz:0b000 blgp:0b000"};

        auto const numBitsPerElementAB = DataTypeInfo::Get(typeAB).elementBits;

        switch(typeAB)
        {
        case DataType::FP8:
            basicGEMM<FP8, FP8, float>(problem);
            break;
        case DataType::BF8:
            basicGEMM<BF8, BF8, float>(problem);
            modifiers = "cbsz:0b001 blgp:0b001";
            break;
        case DataType::FP6:
            GTEST_SKIP() << "Test not yet supported for FP6" << std::endl;
            break;
        case DataType::BF6:
            GTEST_SKIP() << "Test not yet supported for BF6" << std::endl;
            break;
        case DataType::FP4:
            basicGEMM<FP4, FP4, float>(problem);
            modifiers = "cbsz:0b100 blgp:0b100";
            break;
        default:
            Throw<FatalError>(
                fmt::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }

        if(directLDSA && directLDSB)
        {
            auto generatedCode = m_context->instructions()->toString();
            EXPECT_EQ(generatedCode.find("ds_write"), std::string::npos);
        }
    }

    TEST_P(GEMMDirectLDSF8F6F4TestGPU, GPU_BasicGEMM_Prefetch2)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDirectToLds);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        auto [typeAB, MFMAK, transOp, directLDSA, directLDSB] = std::get<1>(GetParam());

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        auto problem = GEMMProblemF8F6F4{waveM, waveN, waveK};

        std::tie(problem.transA, problem.transB) = transOp;

        problem.loadPathA = directLDSA ? SolutionParams::LoadPath::BufferToLDS
                                       : SolutionParams::LoadPath::BufferToLDSViaVGPR;
        problem.loadPathB = directLDSB ? SolutionParams::LoadPath::BufferToLDS
                                       : SolutionParams::LoadPath::BufferToLDSViaVGPR;
        problem.storeLDSD = false;

        problem.prefetch         = true;
        problem.prefetchInFlight = 2;
        problem.unrollK          = 2;

        std::string modifiers{"cbsz:0b000 blgp:0b000"};

        auto const numBitsPerElementAB = DataTypeInfo::Get(typeAB).elementBits;

        switch(typeAB)
        {
        case DataType::FP8:
            basicGEMM<FP8, FP8, float>(problem);
            break;
        case DataType::BF8:
            basicGEMM<BF8, BF8, float>(problem);
            modifiers = "cbsz:0b001 blgp:0b001";
            break;
        case DataType::FP6:
            GTEST_SKIP() << "Test not yet supported for FP6" << std::endl;
            break;
        case DataType::BF6:
            GTEST_SKIP() << "Test not yet supported for BF6" << std::endl;
            break;
        case DataType::FP4:
            basicGEMM<FP4, FP4, float>(problem);
            modifiers = "cbsz:0b100 blgp:0b100";
            break;
        default:
            Throw<FatalError>(
                fmt::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }

        if(directLDSA && directLDSB)
        {
            auto generatedCode = m_context->instructions()->toString();
            EXPECT_EQ(generatedCode.find("ds_write"), std::string::npos);
        }
    }

    template <typename DirectLDSTestType>
    static auto FilterOutNonDirectLDSParamValues(
        ::testing::internal::ParamGenerator<DirectLDSTestType>&& inputParamGenerator)
    {
        using LP = SolutionParams::LoadPath;
        using SM = rocRoller::Operations::ScaleMode;

        std::vector<DirectLDSTestType> filtered;
        for(auto const& inputParam : inputParamGenerator)
        {
            auto const& params = std::get<1>(inputParam);

            auto const& loadPathA = std::get<3>(params);
            auto const& loadPathB = std::get<4>(params);

            if(SolutionParams::IsBufferToLDS(loadPathA) || SolutionParams::IsBufferToLDS(loadPathB))
            {
                filtered.push_back(inputParam);
            }
        }

        return ::testing::ValuesIn(filtered);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMDirectLDSTestBasic, GEMMDirectLDSTestBasicGPU, currentGPUISA());

    INSTANTIATE_TEST_SUITE_P(
        GEMMDirectLDSTest,
        GEMMDirectLDSTestGPU,
        FilterOutNonDirectLDSParamValues<GEMMDirectLDSTestGPU::ParamType>(::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(rocRoller::DataType::Float),
                               ::testing::Values(64, 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")),
                               ::testing::Values(SolutionParams::LoadPath::BufferToLDS,
                                                 SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR),
                               ::testing::Values(SolutionParams::LoadPath::BufferToLDS,
                                                 SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                 SolutionParams::LoadPath::GlobalToVGPR,
                                                 SolutionParams::LoadPath::GlobalToLDSViaVGPR)))));

    INSTANTIATE_TEST_SUITE_P(
        GEMMDirectLDSTest,
        GEMMDirectLDSF8F6F4TestGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(64, 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")),
                               ::testing::Values(true, false),
                               ::testing::Values(true, false))));
} // namespace GEMMTests
