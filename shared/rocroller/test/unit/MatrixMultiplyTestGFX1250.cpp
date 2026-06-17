/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <rocRoller/KernelOptions_detail.hpp>

#include "MatrixMultiplyTestBase.hpp"

#include "Utilities.hpp"

using namespace rocRoller;

namespace MatrixMultiplyTest
{
    namespace SolutionParams = rocRoller::Parameters::Solution;
    // Params are: (AB type, waveK), (transA, transB)
    class WMMATestGFX1250
        : public BaseMatrixMultiplyContextFixture<
              std::tuple<std::pair<rocRoller::DataType, int>, std::pair<std::string, std::string>>>
    {
    };

    // Params are: (AB type, waveK), (transA, transB)
    class F16AccWMMATestGFX1250
        : public BaseMatrixMultiplyContextFixture<
              std::tuple<std::pair<rocRoller::DataType, int>, std::pair<std::string, std::string>>>
    {
    };

    // Params are: A type, B type, waveK, (transA, transB)
    class MixedWMMATestGFX1250
        : public BaseMatrixMultiplyContextFixture<std::tuple<rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             int,
                                                             std::pair<std::string, std::string>>>
    {
    };

    // Params are: A type, B type, waveK, (transA, transB)
    class MixedWMMAF8F6F4TestGFX1250
        : public BaseMatrixMultiplyContextFixture<std::tuple<rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             int,
                                                             std::pair<std::string, std::string>>>
    {
    };

    // Params are: A type, B type, A scale type, B scale type, waveK, scaleBlockSize, (transA, transB)
    class MixedWMMAF8F6F4ScaledTestGFX1250
        : public BaseMatrixMultiplyContextFixture<std::tuple<rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             int,
                                                             uint,
                                                             std::pair<std::string, std::string>>>
    {
    };

    // Params are: (transA, transB)
    class WMMAF4TestGFX1250
        : public BaseMatrixMultiplyContextFixture<std::pair<std::string, std::string>>
    {
    };

    // Params are: A typeA, B type, A scale type, B scale type, scaleBlockSize, (transA, transB)
    class WMMAF4ScaledTestGFX1250
        : public BaseMatrixMultiplyContextFixture<std::tuple<rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             uint,
                                                             std::pair<std::string, std::string>>>
    {
    };

    // Params: waveK
    class ABCWMMATestGFX1250 : public BaseMatrixMultiplyContextFixture<int>
    {
    };

    TEST_P(WMMATestGFX1250, GPU_MatrixMultiplyMacroTileWMMA)
    {
        const auto [typeAndWaveK, transOp] = std::get<1>(GetParam());
        const auto [typeAB, waveK]         = typeAndWaveK;
        const auto [transA, transB]        = transOp;
        const auto loadPathB               = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        auto       typeStr{"f16"};
        switch(typeAB)
        {
        case DataType::Half:
            matrixMultiplyMacroTile<Half, Half, float>(16, 16, waveK, 1, loadPathB, transA, transB);
            break;
        case DataType::BFloat16:
            matrixMultiplyMacroTile<BFloat16, BFloat16, float>(
                16, 16, waveK, 1, loadPathB, transA, transB);
            typeStr = "bf16";
            break;
        case DataType::Float:
            matrixMultiplyMacroTile<float, float, float>(
                16, 16, waveK, 1, loadPathB, transA, transB);
            typeStr = "f32";
            break;
        default:
            Throw<FatalError>(fmt::format(
                "Unexpected data type: {}. (Allowed: Half, Bfloat16, or Float)", toString(typeAB)));
        };

        const auto        numWMMAs = 4; // F32 & F16: mac_k = 4 * wave_k
        const std::string wmmaMnemonic{fmt::format("v_wmma_f32_16x16x{}_{}", waveK, typeStr)};
        std::string       generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, wmmaMnemonic), numWMMAs);
    }

    TEST_P(F16AccWMMATestGFX1250, GPU_MatrixMultiplyMacroTileWMMA)
    {
        const auto [typeAndWaveK, transOp] = std::get<1>(GetParam());
        const auto [dataType, waveK]       = typeAndWaveK;
        const auto [transA, transB]        = transOp;
        const auto loadPathB               = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        auto       typeStr{"f16"};
        switch(dataType)
        {
        case DataType::Half:
            if(waveK == 32)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f16_16x16x32_f16);
            }
            else
            {
                Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
            }
            matrixMultiplyMacroTile<Half, Half, Half, Half>(
                16, 16, waveK, 1, loadPathB, transA, transB);
            break;
        case DataType::BFloat16:
            if(waveK == 32)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_bf16_16x16x32_bf16);
            }
            else
            {
                Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
            }
            matrixMultiplyMacroTile<BFloat16, BFloat16, BFloat16, BFloat16>(
                16, 16, waveK, 1, loadPathB, transA, transB);
            typeStr = "bf16";
            break;
        default:
            Throw<FatalError>(fmt::format("Unexpected data type: {}. (Allowed: Half and Bfloat16)",
                                          toString(dataType)));
        };

        const auto        numWMMAs = 4; // F16 mac_k = 4 * wave_k
        const std::string wmmaMnemonic{
            fmt::format("v_wmma_{}_16x16x{}_{}", typeStr, waveK, typeStr)};
        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, wmmaMnemonic), numWMMAs);
    }

    TEST_P(WMMATestGFX1250, GPU_MatrixMultiplyABWMMA)
    {
        const auto [typeAndWaveK, transOp] = std::get<1>(GetParam());
        const auto [typeAB, waveK]         = typeAndWaveK;
        const auto [transA, transB]        = transOp;
        const auto loadPathAB              = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        auto       typeStr{"f16"};
        switch(typeAB)
        {
        case DataType::Half:
            matrixMultiplyAB<Half, Half, float>(
                16, 16, waveK, 1, loadPathAB, transA == "T", transB == "T");
            break;
        case DataType::BFloat16:
            matrixMultiplyAB<BFloat16, BFloat16, float>(
                16, 16, waveK, 1, loadPathAB, transA == "T", transB == "T");
            typeStr = "bf16";
            break;
        case DataType::Float:
            matrixMultiplyAB<float, float, float>(
                16, 16, waveK, 1, loadPathAB, transA == "T", transB == "T");
            typeStr = "f32";
            break;
        default:
            Throw<FatalError>(fmt::format(
                "Unexpected data type: {}. (Allowed: Half, Bfloat16, or Float)", toString(typeAB)));
        };

        const auto        numWMMAs = 2; // mac_k = 2 * wave_k
        const std::string wmmaMnemonic{fmt::format("v_wmma_f32_16x16x{}_{}", waveK, typeStr)};
        std::string       generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, wmmaMnemonic), numWMMAs);
    }

    TEST_P(F16AccWMMATestGFX1250, GPU_MatrixMultiplyABWMMA)
    {
        const auto [typeAndWaveK, transOp] = std::get<1>(GetParam());
        const auto [typeAB, waveK]         = typeAndWaveK;
        const auto [transA, transB]        = transOp;
        const auto loadPathAB              = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        auto       typeStr{"f16"};
        switch(typeAB)
        {
        case DataType::Half:
            if(waveK == 32)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f16_16x16x32_f16);
            }
            else
            {
                Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
            }
            matrixMultiplyAB<Half, Half, Half, Half>(
                16, 16, waveK, 1, loadPathAB, transA == "T", transB == "T");
            break;
        case DataType::BFloat16:
            if(waveK == 32)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_bf16_16x16x32_bf16);
            }
            else
            {
                Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
            }
            matrixMultiplyAB<BFloat16, BFloat16, BFloat16, BFloat16>(
                16, 16, waveK, 1, loadPathAB, transA == "T", transB == "T");
            typeStr = "bf16";
            break;
        default:
            Throw<FatalError>(fmt::format("Unexpected data type: {}. (Allowed: Half and Bfloat16)",
                                          toString(typeAB)));
        };

        const auto        numWMMAs = 2; // mac_k = 2 * wave_k
        const std::string wmmaMnemonic{
            fmt::format("v_wmma_{}_16x16x{}_{}", typeStr, waveK, typeStr)};
        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, wmmaMnemonic), numWMMAs);
    }

    TEST_P(MixedWMMATestGFX1250, GPU_MatrixMultiplyMacroTileMixedWMMA)
    {
        const auto [typeA, typeB, waveK, transOp] = std::get<1>(GetParam());
        const auto [transA, transB]               = transOp;
        const auto  loadPathB                     = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        std::string typeStr;

        KernelOptions options{};
        options->favourF8F6F4OverF8MatrixInstruction = false;
        setKernelOptions(options);

        if(typeA == typeB)
        {
            switch(typeA)
            {
            case DataType::FP8:
                matrixMultiplyMacroTile<FP8, FP8, float>(
                    16, 16, waveK, 1, loadPathB, transA, transB);
                typeStr = "fp8_fp8";
                break;
            case DataType::BF8:
                matrixMultiplyMacroTile<BF8, BF8, float>(
                    16, 16, waveK, 1, loadPathB, transA, transB);
                typeStr = "bf8_bf8";
                break;
            default:
                Throw<FatalError>(fmt::format("Unexpected data type: {}. (Allowed: FP8 and BF8)",
                                              toString(typeA)));
            };
        }
        else if(typeA == DataType::FP8)
        {
            AssertFatal(typeB == DataType::BF8,
                        "Unexpected data type: " + ShowValue(typeB) + "(Allowed: BF8)");
            matrixMultiplyMacroTile<FP8, BF8, float>(16, 16, waveK, 1, loadPathB, transA, transB);
            typeStr = "fp8_bf8";
        }
        else
        {
            AssertFatal(typeA == DataType::BF8,
                        "Unexpected data type: " + ShowValue(typeA) + "(Allowed: BF8)");
            AssertFatal(typeB == DataType::FP8,
                        "Unexpected data type: " + ShowValue(typeB) + "(Allowed: FP8)");
            matrixMultiplyMacroTile<BF8, FP8, float>(16, 16, waveK, 1, loadPathB, transA, transB);
            typeStr = "bf8_fp8";
        }

        const auto        numWMMAs = 2; // F8 mac_k = 2 * wave_k
        const std::string wmmaMnemonic{fmt::format("v_wmma_f32_16x16x{}_{}", waveK, typeStr)};
        std::string       generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, wmmaMnemonic), numWMMAs);
    }

    TEST_P(MixedWMMATestGFX1250, GPU_MatrixMultiplyMacroTileMixedF16AccWMMA)
    {
        REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasWMMA_f16_16x16x64_f8,
                                GPUCapability::HasWMMA_f16_16x16x128_f8);

        const auto [typeA, typeB, waveK, transOp] = std::get<1>(GetParam());
        const auto [transA, transB]               = transOp;
        const auto  loadPathB                     = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        std::string typeStr;

        KernelOptions options{};
        options->favourF8F6F4OverF8MatrixInstruction = false;
        setKernelOptions(options);

        if(typeA == typeB)
        {
            switch(typeA)
            {
            case DataType::FP8:
                matrixMultiplyMacroTile<FP8, FP8, Half, Half>(
                    16, 16, waveK, 1, loadPathB, transA, transB);
                typeStr = "fp8_fp8";
                break;
            case DataType::BF8:
                matrixMultiplyMacroTile<BF8, BF8, Half, Half>(
                    16, 16, waveK, 1, loadPathB, transA, transB);
                typeStr = "bf8_bf8";
                break;
            default:
                Throw<FatalError>(fmt::format("Unexpected data type: {}. (Allowed: FP8 and BF8)",
                                              toString(typeA)));
            };
        }
        else if(typeA == DataType::FP8)
        {
            AssertFatal(typeB == DataType::BF8,
                        "Unexpected data type: " + ShowValue(typeB) + "(Allowed: BF8)");
            matrixMultiplyMacroTile<FP8, BF8, Half, Half>(
                16, 16, waveK, 1, loadPathB, transA, transB);
            typeStr = "fp8_bf8";
        }
        else
        {
            AssertFatal(typeA == DataType::BF8,
                        "Unexpected data type: " + ShowValue(typeA) + "(Allowed: BF8)");
            AssertFatal(typeB == DataType::FP8,
                        "Unexpected data type: " + ShowValue(typeB) + "(Allowed: FP8)");
            matrixMultiplyMacroTile<BF8, FP8, Half, Half>(
                16, 16, waveK, 1, loadPathB, transA, transB);
            typeStr = "bf8_fp8";
        }

        const auto        numWMMAs = 2; // F8 mac_k = 2 * wave_k
        const std::string wmmaMnemonic{fmt::format("v_wmma_f16_16x16x{}_{}", waveK, typeStr)};
        std::string       generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, wmmaMnemonic), numWMMAs);
    }

    TEST_P(MixedWMMAF8F6F4TestGFX1250, GPU_MatrixMultiplyMacroTileMixedWMMA)
    {
        const auto [typeA, typeB, waveK, transOp] = std::get<1>(GetParam());
        const auto [transA, transB]               = transOp;
        const auto loadPathB                      = SolutionParams::LoadPath::BufferToLDSViaVGPR;

        matrixMultiplyMacroTileMixed(typeA, typeB, 16, 16, waveK, 1, loadPathB, transA, transB);

        const auto        numWMMAs = 2; // F8, F6, and F4 mac_k = 2 * wave_k
        const std::string wmmaMnemonic{fmt::format("v_wmma_f32_16x16x{}_f8f6f4", waveK)};
        std::string       generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, wmmaMnemonic), numWMMAs);
    }

    TEST_P(WMMAF4TestGFX1250, GPU_MatrixMultiplyMacroTileWMMA)
    {
        const auto [transA, transB] = std::get<1>(GetParam());
        const auto loadPathB        = SolutionParams::LoadPath::BufferToLDSViaVGPR;

        matrixMultiplyMacroTile<FP4, FP4, float>(32, 16, 128, 1, loadPathB, transA, transB);

        const auto        numWMMAs = 2; // F4 mac_k = 2 * wave_k
        const std::string wmmaMnemonic{"v_wmma_f32_32x16x128_f4"};
        std::string       generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, wmmaMnemonic), numWMMAs);
    }

    TEST_P(WMMAF4ScaledTestGFX1250, GPU_MatrixMultiplyMacroTileWMMA)
    {
        const auto [typeA, typeB, scaleTypeA, scaleTypeB, scaleBlockSize, transOp]
            = std::get<1>(GetParam());
        const auto [transA, transB] = transOp;
        const ScaleParams scaleParams{scaleTypeA, scaleTypeB, scaleBlockSize};
        const auto        loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;

        matrixMultiplyMacroTileMixed(
            typeA, typeB, 32, 16, 128, 1, loadPathB, transA, transB, scaleParams);

        const auto  numWMMAs = 2; // F4 mac_k = 2 * wave_k
        std::string wmmaMnemonic;
        if(scaleBlockSize == 32)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);
            wmmaMnemonic = "v_wmma_scale_f32_32x16x128_f4";
        }
        else if(scaleBlockSize == 16)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling16);
            wmmaMnemonic = "v_wmma_scale16_f32_32x16x128_f4";
        }
        else
        {
            Throw<FatalError>(
                fmt::format("Unsupported scaleBlockSize: {}. (Allowed 16, 32)", scaleBlockSize));
        }
        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, wmmaMnemonic), numWMMAs);
    }

    TEST_P(MixedWMMAF8F6F4ScaledTestGFX1250, GPU_ScaledMatrixMultiplyMacroTileF8F6F4)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_scale_f8f6f4);

        const auto [typeA, typeB, scaleTypeA, scaleTypeB, waveK, scaleBlockSize, transOp]
            = std::get<1>(GetParam());
        const auto [transA, transB]   = transOp;
        const ScaleParams scaleParams = {
            .scaleTypeA = scaleTypeA, .scaleTypeB = scaleTypeB, .scaleBlockSize = scaleBlockSize};
        const auto loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;

        matrixMultiplyMacroTileMixed(
            typeA, typeB, 16, 16, waveK, 1, loadPathB, transA, transB, scaleParams);

        const auto  numWMMAs = 2; // F8, F6, and F4 mac_k = 2 * wave_k
        std::string wmmaMnemonic;
        if(scaleBlockSize == 32)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);
            wmmaMnemonic = fmt::format("v_wmma_scale_f32_16x16x{}_f8f6f4", waveK);
        }
        else if(scaleBlockSize == 16)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling16);
            wmmaMnemonic = fmt::format("v_wmma_scale16_f32_16x16x{}_f8f6f4", waveK);
        }
        else
        {
            Throw<FatalError>(
                fmt::format("Unsupported scaleBlockSize: {}. (Allowed 16, 32)", scaleBlockSize));
        }
        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, wmmaMnemonic), numWMMAs);
    }

    TEST_P(MixedWMMATestGFX1250, GPU_MatrixMultiplyABMixedWMMA)
    {
        const auto [typeA, typeB, waveK, transOp] = std::get<1>(GetParam());
        const auto [transA, transB]               = transOp;
        const auto  loadPathAB                    = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        std::string typeStr;

        KernelOptions options{};
        options->favourF8F6F4OverF8MatrixInstruction = false;
        setKernelOptions(options);

        if(typeA == typeB)
        {
            switch(typeA)
            {
            case DataType::FP8:
                matrixMultiplyAB<FP8, FP8, float>(
                    16, 16, waveK, 1, loadPathAB, transA == "T", transB == "T");
                typeStr = "fp8_fp8";
                break;
            case DataType::BF8:
                matrixMultiplyAB<BF8, BF8, float>(
                    16, 16, waveK, 1, loadPathAB, transA == "T", transB == "T");
                typeStr = "bf8_bf8";
                break;
            default:
                Throw<FatalError>(fmt::format("Unexpected data type: {}. (Allowed: FP8 and BF8)",
                                              toString(typeA)));
            };
        }
        else if(typeA == DataType::FP8)
        {
            AssertFatal(typeB == DataType::BF8,
                        "Unexpected data type: " + ShowValue(typeB) + "(Allowed: BF8)");
            matrixMultiplyAB<FP8, BF8, float>(
                16, 16, waveK, 1, loadPathAB, transA == "T", transB == "T");
            typeStr = "fp8_bf8";
        }
        else
        {
            AssertFatal(typeA == DataType::BF8,
                        "Unexpected data type: " + ShowValue(typeA) + "(Allowed: BF8)");
            AssertFatal(typeB == DataType::FP8,
                        "Unexpected data type: " + ShowValue(typeB) + "(Allowed: FP8)");
            matrixMultiplyAB<BF8, FP8, float>(
                16, 16, waveK, 1, loadPathAB, transA == "T", transB == "T");
            typeStr = "bf8_fp8";
        }

        const auto        numWMMAs = 2; // F8 mac_k = 2 * wave_k
        const std::string wmmaMnemonic{fmt::format("v_wmma_f32_16x16x{}_{}", waveK, typeStr)};
        std::string       generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, wmmaMnemonic), numWMMAs);
    }

    TEST_P(MixedWMMATestGFX1250, GPU_MatrixMultiplyABMixedF16AccWMMA)
    {
        REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasWMMA_f16_16x16x64_f8,
                                GPUCapability::HasWMMA_f16_16x16x128_f8);

        const auto [typeA, typeB, waveK, transOp] = std::get<1>(GetParam());
        const auto [transA, transB]               = transOp;
        const auto  loadPathAB                    = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        std::string typeStr;

        KernelOptions options{};
        options->favourF8F6F4OverF8MatrixInstruction = false;
        setKernelOptions(options);

        if(typeA == typeB)
        {
            switch(typeA)
            {
            case DataType::FP8:
                matrixMultiplyAB<FP8, FP8, Half, Half>(
                    16, 16, waveK, 1, loadPathAB, transA == "T", transB == "T");
                typeStr = "fp8_fp8";
                break;
            case DataType::BF8:
                matrixMultiplyAB<BF8, BF8, Half, Half>(
                    16, 16, waveK, 1, loadPathAB, transA == "T", transB == "T");
                typeStr = "bf8_bf8";
                break;
            default:
                Throw<FatalError>(fmt::format("Unexpected data type: {}. (Allowed: FP8 and BF8)",
                                              toString(typeA)));
            };
        }
        else if(typeA == DataType::FP8)
        {
            AssertFatal(typeB == DataType::BF8,
                        "Unexpected data type: " + ShowValue(typeB) + "(Allowed: BF8)");
            matrixMultiplyAB<FP8, BF8, Half, Half>(
                16, 16, waveK, 1, loadPathAB, transA == "T", transB == "T");
            typeStr = "fp8_bf8";
        }
        else
        {
            AssertFatal(typeA == DataType::BF8,
                        "Unexpected data type: " + ShowValue(typeA) + "(Allowed: BF8)");
            AssertFatal(typeB == DataType::FP8,
                        "Unexpected data type: " + ShowValue(typeB) + "(Allowed: FP8)");
            matrixMultiplyAB<BF8, FP8, Half, Half>(
                16, 16, waveK, 1, loadPathAB, transA == "T", transB == "T");
            typeStr = "bf8_fp8";
        }

        const auto        numWMMAs = 2; // F8 mac_k = 2 * wave_k
        const std::string wmmaMnemonic{fmt::format("v_wmma_f16_16x16x{}_{}", waveK, typeStr)};
        std::string       generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, wmmaMnemonic), numWMMAs);
    }

    TEST_P(ABCWMMATestGFX1250, GPU_MatrixMultiplyABCF16AccWMMAFP16)
    {
        const auto waveK      = std::get<1>(GetParam());
        const auto loadPathAB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_F16_ACC);

        if(waveK == 32)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f16_16x16x32_f16);
        }
        else
        {
            Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
        }
        matrixMultiplyABC<Half, Half>(16, 16, waveK, 1, loadPathAB);

        const auto        numWMMAs = 2; // mac_k = 2 * wave_k
        const std::string wmmaMnemonic{fmt::format("v_wmma_f16_16x16x{}_f16", waveK)};
        std::string       generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, wmmaMnemonic), numWMMAs);
    }

    TEST_P(ABCWMMATestGFX1250, GPU_MatrixMultiplyABCF16AccWMMABFloat16)
    {
        const auto waveK      = std::get<1>(GetParam());
        const auto loadPathAB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_F16_ACC);

        if(waveK == 32)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_bf16_16x16x32_bf16);
        }
        else
        {
            Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
        }
        matrixMultiplyABC<BFloat16, BFloat16>(16, 16, waveK, 1, loadPathAB);

        const auto        numWMMAs = 2; // mac_k = 2 * wave_k
        const std::string wmmaMnemonic{fmt::format("v_wmma_bf16_16x16x{}_bf16", waveK)};
        std::string       generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, wmmaMnemonic), numWMMAs);
    }

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiply1250,
        WMMATestGFX1250,
        ::testing::Combine(
            ::testing::Values(GPUArchTargetGFX1250Rev0, GPUArchTargetGFX1250Rev1),
            ::testing::Combine(
                ::testing::Values(std::make_pair(rocRoller::DataType::Half, /*waveK*/ 32),
                                  std::make_pair(rocRoller::DataType::BFloat16, /*waveK*/ 32),
                                  std::make_pair(rocRoller::DataType::Float, /*waveK*/ 4)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiply1250,
        F16AccWMMATestGFX1250,
        ::testing::Combine(
            ::testing::Values(GPUArchTargetGFX1250Rev0, GPUArchTargetGFX1250Rev1),
            ::testing::Combine(
                ::testing::Values(std::make_pair(rocRoller::DataType::Half, /*waveK*/ 32),
                                  std::make_pair(rocRoller::DataType::BFloat16, /*waveK*/ 32)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiply1250Rev1,
        MixedWMMATestGFX1250,
        ::testing::Combine(
            ::testing::Values(GPUArchTargetGFX1250Rev1),
            ::testing::Combine(
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(/*waveK*/ 128),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiply1250,
        MixedWMMATestGFX1250,
        ::testing::Combine(
            ::testing::Values(GPUArchTargetGFX1250Rev0, GPUArchTargetGFX1250Rev1),
            ::testing::Combine(
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(/*waveK*/ 64),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiply1250,
        MixedWMMAF8F6F4TestGFX1250,
        ::testing::Combine(
            ::testing::Values(GPUArchTargetGFX1250Rev0, GPUArchTargetGFX1250Rev1),
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

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiply1250Rev1,
        WMMAF4TestGFX1250,
        ::testing::Combine(
            ::testing::Values(GPUArchTargetGFX1250Rev1),
            // TODO: implement transpose support for v_wmma_f32_32x16x128_f4, then re-enable other transposition cases.
            ::testing::Values(std::pair<std::string, std::string>("T", "N"))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiply1250Rev1,
        WMMAF4ScaledTestGFX1250,
        filterValidDataTypeScaleTypeParams<WMMAF4ScaledTestGFX1250::ParamType>(::testing::Combine(
            ::testing::Values(GPUArchTargetGFX1250Rev1),
            ::testing::Combine(
                ::testing::Values(rocRoller::DataType::FP4),
                ::testing::Values(rocRoller::DataType::FP4),
                ::testing::Values(rocRoller::DataType::E8M0,
                                  rocRoller::DataType::E5M3,
                                  rocRoller::DataType::E4M3),
                ::testing::Values(rocRoller::DataType::E8M0,
                                  rocRoller::DataType::E5M3,
                                  rocRoller::DataType::E4M3),
                ::testing::Values(/*scaleBlockSize*/ 16, 32),
                // TODO: mxDataGenerator does not work when fast-moving dim is not multiple
                // of scale-block size (the case for non-TN cases).
                ::testing::Values(std::pair<std::string, std::string>("T", "N"))))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiply1250,
        MixedWMMAF8F6F4ScaledTestGFX1250,
        filterValidDataTypeScaleTypeParams<MixedWMMAF8F6F4ScaledTestGFX1250::ParamType>(
            ::testing::Combine(
                ::testing::Values(GPUArchTargetGFX1250Rev0, GPUArchTargetGFX1250Rev1),
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
                    ::testing::Values(/*scaleBlockSize*/ 16, 32),
                    // TODO: mxDataGenerator does not work when fast-moving dim is not multiple
                    // of scale-block size (the case for non-TN cases).
                    ::testing::Values(std::pair<std::string, std::string>("T", "N"))))));

    INSTANTIATE_TEST_SUITE_P(MatrixMultiplyABCWMMA1250,
                             ABCWMMATestGFX1250,
                             ::testing::Combine(::testing::Values(GPUArchTargetGFX1250Rev0,
                                                                  GPUArchTargetGFX1250Rev1),
                                                ::testing::Values(/*waveK*/ 32)));
} // namespace MatrixMultiplyTest
