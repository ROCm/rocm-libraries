// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

// #include <regex>

#include "GEMMF8F6F4.hpp"
#include "GEMMTestBase.hpp"

#include "common/SourceMatcher.hpp"

namespace GEMMTests
{
    using namespace rocRoller;
    namespace SolutionParams = rocRoller::Parameters::Solution;

    // ========================================================================
    // GEMMSwizzleScaledPrefetchTestSuite
    // ========================================================================

    // Params are: waveK
    class GEMMSwizzleScaledPrefetchTestSuite : public BaseGEMMContextFixture<std::tuple<int>>
    {
    };

    TEST_P(GEMMSwizzleScaledPrefetchTestSuite, GPU_GEMM_Scaled_Prefetch_MX_F4_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto [waveK] = std::get<1>(GetParam());

        int waveM = (waveK == 128) ? 16 : 32;
        int waveN = (waveK == 128) ? 16 : 32;

        auto gemm = GEMMProblemF8F6F4{waveM, waveN, waveK};

        gemm.macM = 256;
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

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.swizzleScale  = true;
        gemm.swizzleM      = 64;
        gemm.swizzleN      = 64;
        gemm.swizzleK      = 8;
        gemm.prefetchScale = true;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        basicGEMM<FP4, FP4, float>(gemm);

        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_ubyte "), 0);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 0);
        // 1x4 wave config: NumAScaleLoadTiles = 256/64 = 4 and NumBScaleLoadTiles = 256/4/64 = 1
        // prefetched : 2 * 5 = 10
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx2 "), 15);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMSwizzleScaledTest,
                             GEMMSwizzleScaledPrefetchTestSuite,
                             ::testing::Combine(currentGPUISA(),
                                                ::testing::Combine(::testing::Values(64, 128))));

    // ========================================================================
    // GEMMSwizzleScaledTestSuite
    // ========================================================================

    class GEMMSwizzleScaledTestSuite : public BaseGEMMContextFixture<>
    {
    };

    TEST_P(GEMMSwizzleScaledTestSuite, GPU_GEMM_Scaled_Prefetch_LDS_MX_F4_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto gemm = GEMMProblemF8F6F4{32, 32, 64};

        gemm.macM = 256;
        gemm.macN = 256;
        gemm.macK = 128;

        gemm.m = 2 * gemm.macM;
        gemm.n = 3 * gemm.macN;
        gemm.k = 4 * gemm.macK;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadScalePathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadScalePathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;

        gemm.unrollK           = 2;
        gemm.prefetch          = true;
        gemm.prefetchInFlight  = 2;
        gemm.prefetchLDSFactor = 2;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.swizzleScale  = true;
        gemm.prefetchScale = true;

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        basicGEMM<FP4, FP4, float>(gemm);

        std::string generatedCode = m_context->instructions()->toString();
        // Swizzle applied for scales loaded via LDS
        EXPECT_GT(countSubstring(generatedCode, "ds_read_b32 "), 0);
        EXPECT_GT(countSubstring(generatedCode, "v_permlane32_swap_b32 "), 0);
    }

    TEST_P(GEMMSwizzleScaledTestSuite, GPU_GEMM_Scaled_Prefetch_D2L_MX_F4_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto gemm = GEMMProblemF8F6F4{32, 32, 64};

        gemm.macM = 256;
        gemm.macN = 256;
        gemm.macK = 128;

        gemm.m = 2 * gemm.macM;
        gemm.n = 3 * gemm.macN;
        gemm.k = 4 * gemm.macK;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadScalePathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadScalePathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDS;

        gemm.unrollK           = 2;
        gemm.prefetch          = true;
        gemm.prefetchInFlight  = 2;
        gemm.prefetchLDSFactor = 1;
        gemm.prefetchMixMemOps = true;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.swizzleScale  = true;
        gemm.swizzleM      = 64;
        gemm.swizzleN      = 64;
        gemm.swizzleK      = 8;
        gemm.prefetchScale = true;

        gemm.workgroupMappingDim   = 0;
        gemm.workgroupMappingValue = 2;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        basicGEMM<FP4, FP4, float>(gemm);

        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_ubyte "), 0);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 0);
        // 1x4 wave config: NumAScaleLoadTiles = 256/64 = 4 and NumBScaleLoadTiles = 256/4/64 = 1
        // prefetched scale: 2 * 5 = 10
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx2 "), 15);
    }

    TEST_P(GEMMSwizzleScaledTestSuite, GPU_GEMM_Scaled_Prefetch_D2L_MX_F4_TN_192x256)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto gemm = GEMMProblemF8F6F4{32, 32, 64};

        gemm.macM = 192;
        gemm.macN = 256;
        gemm.macK = 128;

        gemm.m = 2 * gemm.macM;
        gemm.n = 3 * gemm.macN;
        gemm.k = 4 * gemm.macK;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadScalePathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadScalePathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDS;

        gemm.unrollK           = 2;
        gemm.prefetch          = true;
        gemm.prefetchInFlight  = 2;
        gemm.prefetchLDSFactor = 1;
        gemm.prefetchMixMemOps = true;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.swizzleScale  = true;
        gemm.swizzleM      = 64;
        gemm.swizzleN      = 64;
        gemm.swizzleK      = 8;
        gemm.prefetchScale = true;

        gemm.workgroupMappingDim   = 0;
        gemm.workgroupMappingValue = 2;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        basicGEMM<FP4, FP4, float>(gemm);

        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_ubyte "), 0);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 0);
        // 1x4 wave config: NumAScaleLoadTiles = 192/64 = 3 and NumBScaleLoadTiles = 256/4/64 = 1
        // prefetched scale: 2 * 4 = 8
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx2 "), 12);
    }

    TEST_P(GEMMSwizzleScaledTestSuite, GPU_GEMM_Scaled_StoreHazard_MX_F8_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);
        auto gemm = GEMMProblemF8F6F4{16, 16, 128};

        gemm.macM = 64;
        gemm.macN = 64;
        gemm.macK = 128;
        gemm.m    = 2 * gemm.macM;
        gemm.n    = 3 * gemm.macN;
        gemm.k    = 4 * gemm.macK;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA      = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadScalePathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadScalePathB = SolutionParams::LoadPath::BufferToVGPR;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        basicGEMM<FP8, FP8, float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMSwizzleScaledTest, GEMMSwizzleScaledTestSuite, currentGPUISA());

    // ========================================================================
    // GEMMMXFP4TNSwizzleScaledUnrollTestSuite
    // ========================================================================

    // Params are: mi K tile size, unroll factor
    class GEMMMXFP4TNSwizzleScaledUnrollTestSuite
        : public BaseGEMMContextFixture<std::tuple<int, int>>
    {
    };

    TEST_P(GEMMMXFP4TNSwizzleScaledUnrollTestSuite, GPU_GEMM_Scaled_MX_FP4_TN_Swizzle_64x4_Unroll)
    {

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto [waveK, unrollK] = std::get<1>(GetParam());

        int waveM = (waveK == 128) ? 16 : 32;
        int waveN = (waveK == 128) ? 16 : 32;

        auto gemm = GEMMProblemF8F6F4{waveM, waveN, waveK};

        gemm.macM = 256;
        gemm.macN = 256;
        gemm.macK = 128;

        gemm.m = 2 * gemm.macM;
        gemm.n = 3 * gemm.macN;
        gemm.k = 4 * gemm.macK;

        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 2;

        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadScalePathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadScalePathB = SolutionParams::LoadPath::BufferToVGPR;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.swizzleScale = true;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        // #FIXME: Support for unrollK = 4 and waveK = 128
        if(unrollK == 4 && waveK == 128)
            GTEST_SKIP() << "Skipping GPU_GEMMMXFP4TNSwizzleScaled64x4Unroll test";
        gemm.unrollK = unrollK;
        if(unrollK > 1)
            gemm.swizzleK = 4 * unrollK;
        basicGEMM<FP4, FP4, float>(gemm);

        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_ubyte "), 0);
        if(unrollK == 0)
        {
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 4);
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx2 "), 0);
        }
        else if(unrollK == 2)
        {
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 0);
            // 2x2 wave config: NumAScaleLoadTiles = 256/2/64 = 2 (+2 for Tail Loop) and NumBScaleLoadTiles = 256/2/64 = 2 (+2 for Tail Loop)
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx2 "), 8);
        }
        else if(unrollK == 4)
        {
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 0);
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx2 "), 0);
        }
    }

    TEST_P(GEMMMXFP4TNSwizzleScaledUnrollTestSuite, GPU_GEMM_Scaled_MX_FP4_TN_Swizzle_32x8_Unroll)
    {

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto [waveK, unrollK] = std::get<1>(GetParam());

        int waveM = (waveK == 128) ? 16 : 32;
        int waveN = (waveK == 128) ? 16 : 32;

        auto gemm = GEMMProblemF8F6F4{waveM, waveN, waveK};

        gemm.macM = 128;
        gemm.macN = 128;
        gemm.macK = 256;

        gemm.m = 2 * gemm.macM;
        gemm.n = 3 * gemm.macN;
        gemm.k = 4 * gemm.macK;

        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 2;

        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadScalePathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadScalePathB = SolutionParams::LoadPath::BufferToVGPR;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.swizzleScale = true;
        gemm.swizzleM     = 32;
        gemm.swizzleN     = 32;
        gemm.swizzleK     = 8;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        gemm.unrollK = unrollK;
        basicGEMM<FP4, FP4, float>(gemm);

        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_ubyte "), 0);
        if(unrollK == 0)
        {
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 4);
        }
        else if(unrollK == 2)
        {
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 12);
        }
        else if(unrollK == 4)
        {
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 20);
        }
    }

    INSTANTIATE_TEST_SUITE_P(GEMMSwizzleScaledTest,
                             GEMMMXFP4TNSwizzleScaledUnrollTestSuite,
                             ::testing::Combine(currentGPUISA(),
                                                ::testing::Combine(::testing::Values(64, 128),
                                                                   ::testing::Values(0, 2, 4))));

    // ========================================================================
    // GEMMSwizzleScaledF4TNTestSuite
    // ========================================================================

    // Params are: waveK, loadLDSScaleA, loadLDSScaleB, unrollK, loadPathAB, padA, padB
    class GEMMSwizzleScaledF4TNTestSuite : public BaseGEMMContextFixture<int,
                                                                         SolutionParams::LoadPath,
                                                                         SolutionParams::LoadPath,
                                                                         int,
                                                                         SolutionParams::LoadPath,
                                                                         int,
                                                                         int>
    {
    };

    TEST_P(GEMMSwizzleScaledF4TNTestSuite, GPU_GEMM_Scaled_Swizzle_F4_TN)
    {
        auto const& [arch, waveK, loadScaleA, loadScaleB, unrollK, loadPathAB, padA, padB]
            = GetParam();

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        int waveM = (waveK == 128) ? 16 : 32;
        int waveN = (waveK == 128) ? 16 : 32;

        auto gemm = GEMMProblemF8F6F4(waveM, waveN, waveK);

        gemm.macM = 128;
        gemm.macN = 128;
        gemm.macK = 256;
        gemm.m    = 2 * gemm.macM;
        gemm.n    = 3 * gemm.macN;
        gemm.k    = 4 * gemm.macK;

        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 2;

        gemm.loadPathA = loadPathAB;
        gemm.loadPathB = loadPathAB;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.swizzleScale = true;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        gemm.loadScalePathA = loadScaleA;
        gemm.loadScalePathB = loadScaleB;
        gemm.unrollK        = unrollK;
        if(padA != 0)
            gemm.padA = {padA, 128};
        if(padB != 0)
            gemm.padB = {padB, 128};

        basicGEMM<FP4, FP4, float>(gemm);

        std::string generatedCode = m_context->instructions()->toString();
        if(loadPathAB == SolutionParams::LoadPath::BufferToLDS
           && loadScaleA == SolutionParams::LoadPath::BufferToLDS
           && loadScaleB == SolutionParams::LoadPath::BufferToLDS)
        {
            EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        }
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_ubyte "), 0);

        if(loadPathAB == SolutionParams::LoadPath::BufferToLDS
           || loadPathAB == SolutionParams::LoadPath::BufferToLDSViaVGPR)
        {
            auto offsets = NonZeroDSReadOffsets("ds_read_b128", generatedCode);
            if(padA == 0 && padB == 0)
            {
                EXPECT_EQ(offsets.contains(4 * 1024), true);
                EXPECT_EQ(offsets.contains(4 * 1024 + 2 * 128), false);
            }
            if(padA == 2048 && padB == 2048)
            {
                EXPECT_EQ(offsets.contains(4 * 1024), false);
                EXPECT_EQ(offsets.contains(4 * 1024 + 2 * 128), true);
            }
        }

        if(loadPathAB == SolutionParams::LoadPath::BufferToLDS)
        {
            auto strides = Direct2LDSWriteStrides(generatedCode);
            if(padA == 0 && padB == 0)
            {
                EXPECT_EQ(strides.contains(4 * 1024), true);
                EXPECT_EQ(strides.contains(4 * 1024 + 2 * 128), false);
            }
            if(padA == 2048 && padB == 2048)
            {
                EXPECT_EQ(strides.contains(4 * 1024), false);
                EXPECT_EQ(strides.contains(4 * 1024 + 2 * 128), true);
            }
        }
    }

    using SwizzleScaledF4TNParamGenerator
        = ::testing::internal::ParamGenerator<GEMMSwizzleScaledF4TNTestSuite::ParamType>;
    static auto
        FilterValidSwizzleScaledF4TNParams(SwizzleScaledF4TNParamGenerator&& inputParamGenerator)
    {
        using LP = SolutionParams::LoadPath;

        std::vector<GEMMSwizzleScaledF4TNTestSuite::ParamType> filtered;
        for(auto const& inputParam : inputParamGenerator)
        {
            auto const& [arch, waveK, loadScaleA, loadScaleB, unrollK, loadAB, padA, padB]
                = inputParam;

            if(unrollK == 4 && (loadAB == LP::BufferToLDSViaVGPR || waveK == 128))
                continue;

            filtered.push_back(inputParam);
        }

        return ::testing::ValuesIn(filtered);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMSwizzleScaledTest,
                             GEMMSwizzleScaledF4TNTestSuite,
                             FilterValidSwizzleScaledF4TNParams(::testing::Combine(
                                 currentGPUISA(),
                                 ::testing::Values(64, 128),
                                 ::testing::Values(SolutionParams::LoadPath::BufferToVGPR,
                                                   SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                   SolutionParams::LoadPath::BufferToLDS),
                                 ::testing::Values(SolutionParams::LoadPath::BufferToVGPR,
                                                   SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                   SolutionParams::LoadPath::BufferToLDS),
                                 ::testing::Values(0, 2, 4),
                                 ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                                   SolutionParams::LoadPath::BufferToVGPR,
                                                   SolutionParams::LoadPath::BufferToLDS),
                                 ::testing::Values(0, 2048),
                                 ::testing::Values(0, 2048))));

    // ========================================================================
    // GEMMSwizzleScaledStreamKTestSuite
    // ========================================================================

    // Params: StreamKMode
    class GEMMSwizzleScaledStreamKTestSuite : public BaseGEMMContextFixture<StreamKMode>
    {
    };

    TEST_P(GEMMSwizzleScaledStreamKTestSuite, GPU_GEMM_Scaled_StreamK_Prefetch_MX_F4_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        auto streamKMode = std::get<1>(GetParam());

        auto gemm = GEMMProblemF8F6F4{32, 32, 64};

        gemm.macM = 256;
        gemm.macN = 256;
        gemm.macK = 128;

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

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.swizzleScale  = true;
        gemm.swizzleM      = 64;
        gemm.swizzleN      = 64;
        gemm.swizzleK      = 8;
        gemm.prefetchScale = true;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        gemm.numWGs = deviceProperties.multiProcessorCount;

        gemm.m = gemm.macM * 8;
        gemm.n = gemm.macN * gemm.numWGs / 2 + gemm.macN * 2;
        gemm.k = gemm.macK * 8;

        gemm.streamK = streamKMode;

        basicGEMM<FP4, FP4, float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMSwizzleScaledTest,
                             GEMMSwizzleScaledStreamKTestSuite,
                             ::testing::Combine(currentGPUISA(),
                                                ::testing::Values(StreamKMode::Standard,
                                                                  StreamKMode::TwoTile,
                                                                  StreamKMode::TwoTileDPFirst)));

} // namespace GEMMTests
