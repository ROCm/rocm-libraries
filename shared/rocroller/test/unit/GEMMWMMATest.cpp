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

    // Params are: A & B type, K tile size, (transA, transB)
    class GEMMTestWMMAGPU
        : public BaseGEMMContextFixture<
              std::tuple<std::pair<rocRoller::DataType, int>, std::pair<std::string, std::string>>>
    {
    };

    // Params are: A & B type, K tile size, (transA, transB)
    class GEMMTestWMMAF16AccumGPU
        : public BaseGEMMContextFixture<
              std::tuple<std::pair<rocRoller::DataType, int>, std::pair<std::string, std::string>>>
    {
    };

    // Params are: A type, B type, K tile size, (transA, transB)
    class MixedGEMMTestWMMAGPU
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                   rocRoller::DataType,
                                                   int,
                                                   std::pair<std::string, std::string>>>
    {
    };

    TEST_P(GEMMTestWMMAGPU, GPU_BasicGEMM)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA);
        auto [typeABAndWaveK, transOp] = std::get<1>(GetParam());
        auto [typeAB, waveK]           = typeABAndWaveK;

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

    TEST_P(GEMMTestWMMAF16AccumGPU, GPU_BasicGEMM)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_F16_ACC);
        auto [dataTypeAndWaveK, transOp] = std::get<1>(GetParam());
        auto [dataType, waveK]           = dataTypeAndWaveK;

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

    TEST_P(MixedGEMMTestWMMAGPU, GPU_BasicGEMM)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA);
        auto [typeA, typeB, waveK, transOp] = std::get<1>(GetParam());

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

        basicGEMMMixed(typeA, typeB, gemm);
    }

    INSTANTIATE_TEST_SUITE_P(
        GEMMTestWMMA,
        GEMMTestWMMAGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(std::make_pair(rocRoller::DataType::Half, /*waveK*/ 16),
                                  std::make_pair(rocRoller::DataType::BFloat16, /*waveK*/ 16)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        GEMMTestWMMA,
        GEMMTestWMMAF16AccumGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(std::make_pair(rocRoller::DataType::Half, /*waveK*/ 16),
                                  std::make_pair(rocRoller::DataType::BFloat16, /*waveK*/ 16)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        MixedGEMMTestWMMA,
        MixedGEMMTestWMMAGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(16),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")))));
} // namespace GEMMTests
