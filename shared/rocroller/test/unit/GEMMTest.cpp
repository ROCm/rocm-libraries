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

#include <regex>

#include "GEMMF8F6F4.hpp"
#include "GEMMTestBase.hpp"

#include "common/SourceMatcher.hpp"

using namespace rocRoller;
namespace SolutionParams = rocRoller::Parameters::Solution;

namespace GEMMTests
{
    class GEMMTestGPU : public BaseGEMMContextFixture<>
    {
    };

    // Params are: mi K tile size, unroll factor
    class GEMMMXFP4TNSwizzleScaledUnrollTestGPU
        : public BaseGEMMContextFixture<std::tuple<int, int>>
    {
    };

    struct GEMMF4SwizzleScaledTNGPU
        : public BaseGEMMContextFixture<int, bool, bool, int, SolutionParams::LoadPath>
    {
    };

    // This test is to ensure each scheduler properly yields insts for a basic GEMM
    TEST_P(GEMMTestGPU, GPU_BasicGEMM_Schedulers)
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

        std::set<std::string> insts;
        std::vector<int>      seeds = {2, 4, 8, 314, 1729};
        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Random);
        for(auto seed : seeds)
        {
            settings->set(Settings::RandomSeed, seed);
            basicGEMM<float>(gemm);
            std::string rand     = m_context->instructions()->toString();
            bool        not_seen = insts.insert(rand).second;
            EXPECT_EQ(not_seen, true);
        }
        // Can not compare random insts to others because non-zero chance seed generates such insts
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMM)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMWorkgroupMapping)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.workgroupMappingDim   = 0;
        gemm.workgroupMappingValue = 6;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMWorkgroupMappingXCC)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        REQUIRE_ARCH_CAP(GPUCapability::HasXCC);
        GEMMProblem gemm;
        gemm.workgroupMappingDim   = 0;
        gemm.workgroupMappingValue = 6;
        gemm.workgroupRemapXCC     = true;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMLargerLDS)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.macM             = 128;
        gemm.macN             = 256;
        gemm.loadPathA        = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB        = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storeLDSD        = true;
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

    TEST_P(GEMMTestGPU, GPU_BasicGEMMBetaIsZero)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.beta = 0;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMNotSetC)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.beta = 0;
        basicGEMM<float>(gemm, false, false, 1, true);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMBetaIsZeroStreamK)
    {
        if(m_context->targetArchitecture().target().isCDNA1GPU())
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMMBeta0StreamK test";
        }

        GEMMProblem gemm;

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        gemm.numWGs = deviceProperties.multiProcessorCount;

        gemm.m = gemm.macM * 8;
        gemm.n = gemm.macN * gemm.numWGs / 2 + gemm.macN * 2;

        ASSERT_GE(gemm.m * gemm.n / gemm.macM / gemm.macN, gemm.numWGs);

        gemm.streamK = StreamKMode::Standard;
        gemm.k       = gemm.macK * 8;

        // TODO: Does not work with unrolling K
        //gemm.unrollK          = 2;
        //gemm.prefetch         = true;
        //gemm.prefetchInFlight = 2;

        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storeLDSD = true;

        gemm.beta = 0;

        for(auto twoTile : {true, false})
        {
            gemm.streamK = twoTile ? StreamKMode::TwoTile : StreamKMode::Standard;
            basicGEMM<float>(gemm, false, false, 1, true);
        }
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMStreamK)
    {
        if(m_context->targetArchitecture().target().isCDNA1GPU())
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMMStreamK test";
        }

        GEMMProblem gemm;

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        gemm.numWGs = deviceProperties.multiProcessorCount;

        gemm.m = gemm.macM * 8;
        gemm.n = gemm.macN * gemm.numWGs / 2 + gemm.macN * 2;

        ASSERT_GE(gemm.m * gemm.n / gemm.macM / gemm.macN, gemm.numWGs);

        gemm.streamK = StreamKMode::Standard;
        gemm.k       = gemm.macK * 8;

        // TODO: Does not work with unrolling K
        //gemm.unrollK          = 2;
        //gemm.prefetch         = true;
        //gemm.prefetchInFlight = 2;

        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storeLDSD = true;

        for(auto twoTile : {true, false})
        {
            gemm.streamK = twoTile ? StreamKMode::TwoTile : StreamKMode::Standard;
            basicGEMM<float>(gemm);
        }
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMStreamKTwoTileDPFirst)
    {
        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));

        GEMMProblem gemm;
        gemm.numWGs = deviceProperties.multiProcessorCount;
        gemm.m      = gemm.macM * 8;
        gemm.n      = gemm.macN * gemm.numWGs / 2 + gemm.macN * 2;
        ASSERT_GE(gemm.m * gemm.n / gemm.macM / gemm.macN, gemm.numWGs);
        gemm.k = gemm.macK * 8;

        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storeLDSD = true;
        gemm.streamK   = StreamKMode::TwoTileDPFirst;

        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, DISABLED_GPU_BasicGEMMMultipleOutputTiles)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.storeLDSD     = false;
        gemm.loopOverTiles = true;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMNoLDSA)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.fuseLoops = false;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMNoLDSB)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.fuseLoops = false;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMNoLDSAB)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.fuseLoops = false;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollK)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.storeLDSD = false;
        gemm.fuseLoops = true;
        gemm.unrollK   = 2;

        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;

        gemm.macM = 128;
        gemm.macK = 4;

        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollKTailLoop)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.m         = 64;
        gemm.n         = 128;
        gemm.transA    = "T";
        gemm.transB    = "N";
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.storeLDSD = false;
        gemm.fuseLoops = true;
        gemm.tailLoops = true;
        gemm.unrollK   = 4;
        gemm.macK      = 8;
        for(auto k : {8, 16, 24, 32, 40, 48, 56, 64})
        {
            gemm.k = k;
            basicGEMM<float>(gemm);
        }
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollKLDS)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 2;
        gemm.macK      = 4;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollKMoreLDS)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.macK      = 8;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollKMoreLDSA)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.macK      = 8;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollKMoreLDSB)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.macK      = 8;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollKLDSPrefetch)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storeLDSD = true;
        gemm.fuseLoops = true;
        gemm.unrollK   = 2;
        gemm.macK      = 4;
        gemm.prefetch  = true;
        gemm.macM      = gemm.waveM * 4;
        gemm.macN      = gemm.waveN * 2;

        for(auto inflight : {1, 2})
        {
            gemm.prefetchInFlight = inflight;
            for(auto ldsFactor : {0, 1, 2})
            {
                gemm.prefetchLDSFactor = ldsFactor;
                for(auto mixMemOps : {false, true})
                {
                    gemm.prefetchMixMemOps = mixMemOps;
                    basicGEMM<float>(gemm);
                }
            }
        }
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP16UnrollKLDSPrefetch)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 16 * 2;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storeLDSD = true;
        gemm.fuseLoops = true;
        gemm.unrollK   = 2;
        gemm.macM      = 128;
        gemm.macN      = 128;
        gemm.macK      = 16;
        gemm.prefetch  = true;
        gemm.waveK     = 8;

        gemm.transA = "N";
        gemm.transB = "N";

        for(auto inflight : {1, 2})
        {
            gemm.prefetchInFlight = inflight;
            for(auto ldsFactor : {0, 2})
            {
                gemm.prefetchLDSFactor = ldsFactor;
                for(auto mixMemOps : {false, true})
                {
                    gemm.prefetchMixMemOps = mixMemOps;
                    basicGEMM<Half>(gemm);
                }
            }
        }
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollKLDSMultiPrefetch)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 3;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 3;
        gemm.macK      = 4;
        gemm.prefetch  = true;

        for(auto inflight : {1, 2, 3})
        {
            gemm.prefetchInFlight = inflight;
            for(auto ldsFactor : {0, 2})
            {
                gemm.prefetchLDSFactor = ldsFactor;
                for(auto mixMemOps : {false, true})
                {
                    gemm.prefetchMixMemOps = mixMemOps;
                    basicGEMM<float>(gemm);
                }
            }
        }
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP16Prefetch3)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.m                 = 4096;
        gemm.n                 = 4096;
        gemm.k                 = 2048 * 3;
        gemm.loadPathA         = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB         = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storeLDSD         = false;
        gemm.fuseLoops         = false;
        gemm.unrollK           = 3;
        gemm.macM              = 128;
        gemm.macN              = 16;
        gemm.macK              = 64;
        gemm.waveM             = 16;
        gemm.waveN             = 16;
        gemm.waveK             = 16;
        gemm.workgroupSizeX    = 256;
        gemm.workgroupSizeY    = 1;
        gemm.prefetch          = true;
        gemm.prefetchInFlight  = 3;
        gemm.prefetchLDSFactor = 2;
        gemm.prefetchMixMemOps = true;
        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP8_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        basicGEMM<FP8, FP8, float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMBF8_16x16x32_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        basicGEMM<BF8, BF8, float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMConversionFP8_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        // D (FP8) = Convert( alpha * A (FP8) * B (FP8) + beta * C (F32) )
        basicGEMM<FP8, FP8, float, FP8>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMConversionBF8_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        // D (BF8) = Convert( alpha * A (BF8) * B (BF8) + beta * C (F32) )
        basicGEMM<BF8, BF8, float, BF8>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMSRConversionFP8_NT)
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

    TEST_P(GEMMTestGPU, GPU_BasicGEMMSRConversionBF8_NT)
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

    TEST_P(GEMMTestGPU, GPU_ScaledPrefetchGEMMMXF8TN)
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

        gemm.loadPathA     = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB     = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadLDSScaleA = false;
        gemm.loadLDSScaleB = false;

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

    void check_GEMMF8_TN(rocRoller::ContextPtr m_context)
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

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP8_16x16x32_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8TN{};
        basicGEMM<FP8, FP8, float>(gemm);
        check_GEMMF8_TN(m_context);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMBF8_16x16x32_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8TN{};
        basicGEMM<BF8, BF8, float>(gemm);
        check_GEMMF8_TN(m_context);
    }

    TEST_P(GEMMF4SwizzleScaledTNGPU, GEMMF4SwizzleScaledTNGPU)
    {
        auto const& [arch, waveK, loadLDSScaleA, loadLDSScaleB, unrollK, loadPathAB] = GetParam();

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        if(unrollK == 4
           && (loadPathAB == SolutionParams::LoadPath::BufferToLDSViaVGPR || waveK == 128))
        {
            GTEST_SKIP() << "FIXME: Re-visit when ClusterParallelChains is implemented";
        }

        int waveM = (waveK == 128) ? 16 : 32;
        int waveN = (waveK == 128) ? 16 : 32;

        auto gemm = GEMMProblemF8F6F4{waveM, waveN, waveK};

        gemm.macM = 256;
        gemm.macN = 256;
        gemm.macK = 128;
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

        gemm.loadLDSScaleA = loadLDSScaleA;
        gemm.loadLDSScaleB = loadLDSScaleB;
        gemm.unrollK       = unrollK;

        basicGEMM<FP4, FP4, float>(gemm);

        std::string generatedCode = m_context->instructions()->toString();
        // when both the scales are loaded directly from buffer into VGPRs
        if(!loadLDSScaleA && !loadLDSScaleB)
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_ubyte "), 0);
        // when either scale is loaded via LDS -- no swizzle applied
        if(loadLDSScaleA || loadLDSScaleB)
            EXPECT_GT(countSubstring(generatedCode, "ds_read_u8 "), 0);

        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_ubyte "), 0);
    }

    TEST_P(GEMMMXFP4TNSwizzleScaledUnrollTestGPU, GPU_GEMMMXFP4TNSwizzleScaled64x4Unroll)
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

        gemm.loadPathA     = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB     = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadLDSScaleA = false;
        gemm.loadLDSScaleB = false;

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
            // 2x2 wave config: NumAScaleLoadTiles = 256/2/64 = 2 and NumBScaleLoadTiles = 256/2/64 = 2
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx2 "), 4);
        }
        else if(unrollK == 4)
        {
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 0);
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx2 "), 0);
        }
    }

    TEST_P(GEMMMXFP4TNSwizzleScaledUnrollTestGPU, GPU_GEMMMXFP4TNSwizzleScaled32x8Unroll)
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

        gemm.loadPathA     = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB     = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadLDSScaleA = false;
        gemm.loadLDSScaleB = false;

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
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 8);
        }
        else if(unrollK == 4)
        {
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 16);
        }
    }

    TEST_P(GEMMTestGPU, GPU_SwizzleScaledPrefetchGEMMMXF4TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);

        for(auto waveK : {64, 128})
        {
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

            gemm.loadPathA     = SolutionParams::LoadPath::BufferToLDSViaVGPR;
            gemm.loadPathB     = SolutionParams::LoadPath::BufferToLDSViaVGPR;
            gemm.loadLDSScaleA = false;
            gemm.loadLDSScaleB = false;

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

            gemm.scaleBlockSize = m_context->targetArchitecture().GetCapability(
                GPUCapability::DefaultScaleBlockSize);

            basicGEMM<FP4, FP4, float>(gemm);

            std::string generatedCode = m_context->instructions()->toString();
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_ubyte "), 0);
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 0);
            // 1x4 wave config: NumAScaleLoadTiles = 256/64 = 4 and NumBScaleLoadTiles = 256/4/64 = 1
            // prefetched : 2 * 5 = 10
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx2 "), 10);
        }
    }

    TEST_P(GEMMTestGPU, GPU_SwizzleScaledPrefetchLDSGEMMMXF4TN)
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

        gemm.loadPathA     = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB     = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadLDSScaleA = true;
        gemm.loadLDSScaleB = true;

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
        // no swizzle applied for scales loaded via LDS
        EXPECT_GT(countSubstring(generatedCode, "ds_read_u8 "), 0);
    }

    TEST_P(GEMMTestGPU, GPU_SwizzleScaledPrefetchD2LGEMMMXF4TN)
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

        gemm.loadLDSScaleA = false;
        gemm.loadLDSScaleB = false;
        gemm.loadPathA     = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB     = SolutionParams::LoadPath::BufferToLDS;

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
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx2 "), 10);
    }

    TEST_P(GEMMTestGPU, GPU_SwizzleScaledPrefetchD2LGEMMMXF4TN_192x256)
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

        gemm.loadLDSScaleA = false;
        gemm.loadLDSScaleB = false;
        gemm.loadPathA     = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB     = SolutionParams::LoadPath::BufferToLDS;

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
        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx2 "), 8);
    }

    TEST_P(GEMMTestGPU, GPU_StoreHazardScaledGEMMMXF8TN)
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

        gemm.loadPathA     = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB     = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadLDSScaleA = false;
        gemm.loadLDSScaleB = false;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        basicGEMM<FP8, FP8, float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_LargerLDSGEMMFP8_32x32x64_TN)
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

    TEST_P(GEMMTestGPU, GPU_GEMM_FP8_Direct2LDS_MT256x256x128_MI32x32x64_TN)
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
        gemm.storeLDSD = false;
        gemm.transA    = "T";
        gemm.transB    = "N";

        basicGEMM<FP8, FP8, float>(gemm);

        auto const  numBitsPerElementAB = 8;
        std::string generatedCode       = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        checkNumDwordx4(generatedCode,
                        numBitsPerElementAB,
                        gemm.macM,
                        gemm.macN,
                        gemm.macK,
                        gemm.workgroupSizeX * gemm.workgroupSizeY);
    }

    TEST_P(GEMMTestGPU, GPU_GEMM_BF8_Direct2LDS_MT256x256x128_MI32x32x64_TN)
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
        gemm.storeLDSD = false;
        gemm.transA    = "T";
        gemm.transB    = "N";

        basicGEMM<BF8, BF8, float>(gemm);

        auto const  numBitsPerElementAB = 8;
        std::string generatedCode       = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        checkNumDwordx4(generatedCode,
                        numBitsPerElementAB,
                        gemm.macM,
                        gemm.macN,
                        gemm.macK,
                        gemm.workgroupSizeX * gemm.workgroupSizeY);
    }

    TEST_P(GEMMTestGPU, GPU_GEMM_FP4_Direct2LDS_MT256x256x128_MI32x32x64_TN)
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
        gemm.storeLDSD = false;
        gemm.transA    = "T";
        gemm.transB    = "N";

        basicGEMM<FP4, FP4, float>(gemm);

        auto const  numBitsPerElementAB = 4;
        std::string generatedCode       = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        checkNumDwordx4(generatedCode,
                        numBitsPerElementAB,
                        gemm.macM,
                        gemm.macN,
                        gemm.macK,
                        gemm.workgroupSizeX * gemm.workgroupSizeY);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP16AllLDS)
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
        gemm.storeLDSD = true;

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP16_96x256)
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
        gemm.storeLDSD = true;

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMStoreDWave)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;

        auto nonZeroDSReadOffsets = [](auto s) {
            std::regex ds_read_offset("ds_read_b128.*offset:(\\d+)");

            auto begin = std::sregex_iterator(s.begin(), s.end(), ds_read_offset);
            auto end   = std::sregex_iterator();

            std::set<int> rv;
            for(auto i = begin; i != end; ++i)
            {
                auto m = (*i)[1].str();
                rv.insert(std::stoi(m));
            }
            return rv;
        };

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
        gemm.storeLDSD = true;

        gemm.splitStoreTileIntoWaveBlocks = true;
        basicGEMM<Half>(gemm);
        auto instructions0 = output();
        EXPECT_EQ(nonZeroDSReadOffsets(instructions0), std::set<int>{1024});

        gemm.splitStoreTileIntoWaveBlocks = false;
        basicGEMM<Half>(gemm);
        auto instructions1 = output();
        EXPECT_EQ(nonZeroDSReadOffsets(instructions1), std::set<int>{64});
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP16AllLDSDebug)
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
        gemm.storeLDSD = true;

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_ScaledLDSGEMMMXF8TN)
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

        gemm.loadPathA     = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB     = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadLDSScaleA = true;
        gemm.loadLDSScaleB = true;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        basicGEMM<FP8, FP8, float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMTest, GEMMTestGPU, currentGPUISA());

    INSTANTIATE_TEST_SUITE_P(GEMMMXFP4TNSwizzleScaledUnrollTest,
                             GEMMMXFP4TNSwizzleScaledUnrollTestGPU,
                             ::testing::Combine(currentGPUISA(),
                                                ::testing::Combine(::testing::Values(64, 128),
                                                                   ::testing::Values(0, 2, 4))));
    INSTANTIATE_TEST_SUITE_P(
        GEMMF4SwizzleScaledTNGPU,
        GEMMF4SwizzleScaledTNGPU,
        ::testing::Combine(currentGPUISA(),
                           ::testing::Values(64, 128),
                           ::testing::Values(false, true),
                           ::testing::Values(false, true),
                           ::testing::Values(0, 2, 4),
                           ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                                             SolutionParams::LoadPath::BufferToVGPR)));
}
