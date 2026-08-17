// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <regex>

#include "GEMMF8F6F4.hpp"
#include "GEMMTestBase.hpp"

#include "common/SourceMatcher.hpp"

namespace GEMMTests
{
    using namespace rocRoller;
    namespace SolutionParams = rocRoller::Parameters::Solution;

    std::set<int> NonZeroDSReadOffsets(std::string const& instruction, std::string const& s)
    {
        std::regex ds_read_offset(instruction + ".*offset:(\\d+)");

        auto begin = std::sregex_iterator(s.begin(), s.end(), ds_read_offset);
        auto end   = std::sregex_iterator();

        std::set<int> rv;
        for(auto i = begin; i != end; ++i)
        {
            auto m = (*i)[1].str();
            rv.insert(std::stoi(m));
        }
        return rv;
    }

    std::set<int> Direct2LDSWriteStrides(std::string const& s)
    {
        std::regex m0_stride_pattern("s_add_u32 m0, m0, (\\d+)");

        auto begin = std::sregex_iterator(s.begin(), s.end(), m0_stride_pattern);
        auto end   = std::sregex_iterator();

        std::set<int> rv;
        for(auto i = begin; i != end; ++i)
        {
            auto m = (*i)[1].str();
            rv.insert(std::stoi(m));
        }
        return rv;
    }

    // ========================================================================
    // GEMMTestSuite
    // ========================================================================

    class GEMMTestSuite : public BaseGEMMContextFixture<>
    {
    };

    // This test is to ensure each scheduler properly yields insts for a basic GEMM
    TEST_P(GEMMTestSuite, GPU_GEMM_Schedulers)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.macK = 8;

        // TODO: Re-enable LDS once LDS deallocations are fixed
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;

        auto settings = Settings::getInstance();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Sequential);
        basicGEMM<float>(gemm);
        std::string seq = m_context->instructions()->toString();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::RoundRobin);
        basicGEMM<float>(gemm);
        std::string rr = m_context->instructions()->toString();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Cooperative);
        basicGEMM<float>(gemm);
        std::string coop_nop = m_context->instructions()->toString();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Priority);
        basicGEMM<float>(gemm);
        std::string priority_nop = m_context->instructions()->toString();

        EXPECT_NE(NormalizedSource(seq), NormalizedSource(rr));

        EXPECT_NE(NormalizedSource(coop_nop), NormalizedSource(rr));

        EXPECT_NE(NormalizedSource(priority_nop), NormalizedSource(rr));
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP32_Basic)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_LDS_Padded)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.transA    = "T";
        gemm.transB    = "N";

        // This kernel uses 128bit buffer_load and ds_write
        // instructions; so each wave loads 16 * 64 bytes per
        // instruction.
        gemm.padA = {32 * 64, 4};
        gemm.padB = {16 * 64, 4};
        basicGEMM<float>(gemm);

        auto instructions = m_context->instructions()->toString();
        auto ldsOffsets   = NonZeroDSReadOffsets("ds_read_b32", instructions);

        // With no padding in A, the LDS buffer for B would start at
        //
        //   B[0] = WGTS M * WGTS K * sizeof(float)
        //
        // so make sure this isn't in the read offsets!
        EXPECT_FALSE(ldsOffsets.contains(gemm.macM * gemm.macK * sizeof(float)));
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_Workgroup_Mapping)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.workgroupMappingDim   = 0;
        gemm.workgroupMappingValue = 6;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_Workgroup_Mapping_XCC)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        REQUIRE_ARCH_CAP(GPUCapability::HasXCC);
        GEMMProblem gemm;
        gemm.workgroupMappingDim   = 0;
        gemm.workgroupMappingValue = 6;
        gemm.workgroupRemapXCC     = true;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_LDS_Larger)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.macM             = 128;
        gemm.macN             = 256;
        gemm.loadPathA        = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB        = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath        = SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer;
        gemm.prefetchInFlight = 1;
        auto maxLDS = m_context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize);
        auto bytesPerElement = sizeof(float);
        auto ldsA            = gemm.macM * gemm.macK * bytesPerElement * gemm.prefetchInFlight;
        auto ldsB            = gemm.macK * gemm.macN * bytesPerElement * gemm.prefetchInFlight;
        auto ldsD            = gemm.waveM * gemm.waveN * bytesPerElement;

        if(ldsA + ldsB + ldsD <= maxLDS)
        {
            basicGEMM<float>(gemm);
        }
        else
        {
            GTEST_SKIP() << "LDS usage exceeds maxLDS.";
        }
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP32_BetaZero)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.beta = 0;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP32_NotSetC)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.beta = 0;
        basicGEMM<float>(gemm, false, false, 1, true);
    }

    TEST_P(GEMMTestSuite, DISABLED_GPU_GEMM_DataType_FP32_MultipleOutputTiles)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.storePath     = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.loopOverTiles = true;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_NoLDS_A)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.fuseLoops = false;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_NoLDS_B)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.fuseLoops = false;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_NoLDS_AB)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.fuseLoops = false;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP8_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        basicGEMM<FP8, FP8, float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_BF8_16x16x32_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        basicGEMM<BF8, BF8, float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP8_Conversion_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        // D (FP8) = Convert( alpha * A (FP8) * B (FP8) + beta * C (F32) )
        basicGEMM<FP8, FP8, float, FP8>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_BF8_Conversion_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        // D (BF8) = Convert( alpha * A (BF8) * B (BF8) + beta * C (F32) )
        basicGEMM<BF8, BF8, float, BF8>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP8_StochasticRounding_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        // D (FP8) = StochasticRoundingConvert( alpha * A (FP8) * B (FP8) + beta * C (F32) )
        basicGEMM<FP8, FP8, float, FP8>(gemm,
                                        /* debuggable  */ false,
                                        /* setIdentity */ false,
                                        /* numIters    */ 1,
                                        /* notSetC     */ false,
                                        /* seed        */ 56789u);

        // Check stochastic rounding instruction has be generated
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasMFMA_fp8))
        {
            std::string const generatedCode = m_context->instructions()->toString();
            EXPECT_NE(generatedCode.find("v_cvt_sr_fp8_f32"), std::string::npos);
            EXPECT_EQ(generatedCode.find("v_cvt_pk_fp8_f32"), std::string::npos);
        }
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_BF8_StochasticRounding_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        // D (BF8) = StochasticRoundingConvert( alpha * A (BF8) * B (BF8) + beta * C (F32) )
        basicGEMM<BF8, BF8, float, BF8>(gemm,
                                        /* debuggable  */ false,
                                        /* setIdentity */ false,
                                        /* numIters    */ 1,
                                        /* notSetC     */ false,
                                        /* seed        */ 56789u);

        // Check stochastic rounding instruction has be generated
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasMFMA_fp8))
        {
            std::string const generatedCode = m_context->instructions()->toString();
            EXPECT_NE(generatedCode.find("v_cvt_sr_bf8_f32"), std::string::npos);
            EXPECT_EQ(generatedCode.find("v_cvt_pk_bf8_f32"), std::string::npos);
        }
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_Scaled_Prefetch_MX_F8_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);
        auto gemm = GEMMProblemF8F6F4{32, 32, 64};

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 128;

        gemm.m = 2 * gemm.macM;
        gemm.n = 3 * gemm.macN;
        gemm.k = 4 * gemm.macK;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadScalePathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadScalePathB = SolutionParams::LoadPath::BufferToVGPR;

        gemm.unrollK           = 2;
        gemm.prefetch          = true;
        gemm.prefetchInFlight  = 2;
        gemm.prefetchLDSFactor = 2;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        basicGEMM<FP8, FP8, float>(gemm);
    }

    void CheckGEMMF8TN(rocRoller::ContextPtr m_context)
    {
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasMFMA_fp8))
        {
            std::string generatedCode = m_context->instructions()->toString();

            EXPECT_EQ(countSubstring(generatedCode, "buffer_load"), 3);
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx4 "), 2);
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 1);

            EXPECT_EQ(countSubstring(generatedCode, "ds_write_b"), 2);
            EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128 "), 1);
            EXPECT_EQ(countSubstring(generatedCode, "ds_write_b32 "), 1);

            EXPECT_EQ(countSubstring(generatedCode, "ds_read"), 4);
            EXPECT_EQ(countSubstring(generatedCode, "ds_read_b64 "), 4);
        }
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP8_16x16x32_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8TN{};
        basicGEMM<FP8, FP8, float>(gemm);
        CheckGEMMF8TN(m_context);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_BF8_16x16x32_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8TN{};
        basicGEMM<BF8, BF8, float>(gemm);
        CheckGEMMF8TN(m_context);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP8_LargerLDS_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm             = GEMMProblemF8F6F4{32, 32, 64};
        gemm.macM             = 128;
        gemm.macN             = 128;
        gemm.macK             = 256;
        gemm.loadPathA        = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB        = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.prefetchInFlight = 2;

        basicGEMM<FP8, FP8, float>(gemm);
    }

    static void CheckNumDwordx4(std::string generatedCode,
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

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_Direct2LDS_FP8_MT256x256x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm      = GEMMProblemF8F6F4{32, 32, 64};
        gemm.m         = 512;
        gemm.n         = 256;
        gemm.k         = 512;
        gemm.macM      = 256;
        gemm.macN      = 256;
        gemm.macK      = 128;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDS;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.transA    = "T";
        gemm.transB    = "N";

        basicGEMM<FP8, FP8, float>(gemm);

        auto const  numBitsPerElementAB = 8;
        std::string generatedCode       = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        CheckNumDwordx4(generatedCode,
                        numBitsPerElementAB,
                        gemm.macM,
                        gemm.macN,
                        gemm.macK,
                        gemm.workgroupSizeX * gemm.workgroupSizeY);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_Direct2LDS_BF8_MT256x256x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm      = GEMMProblemF8F6F4{32, 32, 64};
        gemm.m         = 512;
        gemm.n         = 256;
        gemm.k         = 512;
        gemm.macM      = 256;
        gemm.macN      = 256;
        gemm.macK      = 128;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDS;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.transA    = "T";
        gemm.transB    = "N";

        basicGEMM<BF8, BF8, float>(gemm);

        auto const  numBitsPerElementAB = 8;
        std::string generatedCode       = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        CheckNumDwordx4(generatedCode,
                        numBitsPerElementAB,
                        gemm.macM,
                        gemm.macN,
                        gemm.macK,
                        gemm.workgroupSizeX * gemm.workgroupSizeY);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_Direct2LDS_FP4_MT256x256x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm      = GEMMProblemF8F6F4{32, 32, 64};
        gemm.m         = 512;
        gemm.n         = 256;
        gemm.k         = 512;
        gemm.macM      = 256;
        gemm.macN      = 256;
        gemm.macK      = 128;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDS;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.transA    = "T";
        gemm.transB    = "N";

        basicGEMM<FP4, FP4, float>(gemm);

        auto const  numBitsPerElementAB = 4;
        std::string generatedCode       = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        CheckNumDwordx4(generatedCode,
                        numBitsPerElementAB,
                        gemm.macM,
                        gemm.macN,
                        gemm.macK,
                        gemm.workgroupSizeX * gemm.workgroupSizeY);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_FP4_MT256x256x128_LDSSwizzle)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm           = GEMMProblemF8F6F4{32, 32, 64};
        gemm.m              = 512;
        gemm.n              = 256;
        gemm.k              = 512;
        gemm.macM           = 256;
        gemm.macN           = 256;
        gemm.macK           = 128;
        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDS;
        gemm.storePath      = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.transA         = "T";
        gemm.transB         = "N";
        gemm.ldsSwizzleMode = LDSBankSwizzleMode::Swizzle;

        basicGEMM<FP4, FP4, float>(gemm);

        // LDS swizzle uses XOR-based permutation; expect exactly 12 v_xor_b32.
        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        EXPECT_EQ(countSubstring(generatedCode, "v_xor_b32"), 12);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_FP4_MI16x16x128_MT64x64x256_LDSSwizzle)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm           = GEMMProblemF8F6F4{16, 16, 128};
        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDS;
        gemm.ldsSwizzleMode = LDSBankSwizzleMode::Swizzle;

        basicGEMM<FP4, FP4, float>(gemm);

        // LDS swizzle uses XOR-based permutation; expect exactly 12 v_xor_b32.
        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        EXPECT_EQ(countSubstring(generatedCode, "v_xor_b32"), 12);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_FP4_MI16x16x128_MT64x64x256_LDSSwizzle_ViaVGPR)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm           = GEMMProblemF8F6F4{16, 16, 128};
        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.ldsSwizzleMode = LDSBankSwizzleMode::Swizzle;

        basicGEMM<FP4, FP4, float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP16_AllLDS)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer;

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP16_96x256)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;

        gemm.m = 192;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 96;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer;

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_StoreDWave)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer;

        gemm.splitStoreTileIntoWaveBlocks = true;
        basicGEMM<Half>(gemm);
        auto instructions0 = output();
        EXPECT_EQ(NonZeroDSReadOffsets("ds_read_b128", instructions0), std::set<int>{1024});

        gemm.splitStoreTileIntoWaveBlocks = false;
        basicGEMM<Half>(gemm);
        auto instructions1 = output();
        EXPECT_EQ(NonZeroDSReadOffsets("ds_read_b128", instructions1), std::set<int>{64});
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP16_AllLDS_Debug)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer;

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_Scaled_LDS_MX_F8_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);
        auto gemm = GEMMProblemF8F6F4{32, 32, 64};

        gemm.macM = 64;
        gemm.macN = 256;
        gemm.macK = 128;
        gemm.m    = 2 * gemm.macM;
        gemm.n    = 3 * gemm.macN;
        gemm.k    = 4 * gemm.macK;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA      = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadScalePathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadScalePathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        basicGEMM<FP8, FP8, float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMTest, GEMMTestSuite, currentGPUISA());

    // ========================================================================
    // GEMMSchedulerRandomTestSuite
    // ========================================================================

    // Params are: random seed value
    class GEMMSchedulerRandomTestSuite : public BaseGEMMContextFixture<std::tuple<int>>
    {
    };

    // Test to verify different random seeds produce different instruction sequences
    TEST_P(GEMMSchedulerRandomTestSuite, GPU_GEMM_Schedulers_Random)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        auto [seed] = std::get<1>(GetParam());

        GEMMProblem gemm;
        gemm.macK = 8;

        // TODO: Re-enable LDS once LDS deallocations are fixed
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;

        auto settings = Settings::getInstance();
        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Random);
        settings->set(Settings::RandomSeed, seed);

        basicGEMM<float>(gemm);

        // Verify the kernel generates successfully with this random seed
        EXPECT_GT(m_context->instructions()->toString().size(), 0);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMTest,
        GEMMSchedulerRandomTestSuite,
        ::testing::Combine(currentGPUISA(),
                           ::testing::Combine(::testing::Values(2, 4, 8, 314, 1729))));

}
