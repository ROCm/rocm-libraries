// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "MatrixMultiplyTestBase.hpp"

#include <rocRoller/KernelOptions_detail.hpp>

namespace MatrixMultiplyTest
{
    using namespace rocRoller;
    namespace SolutionParams = rocRoller::Parameters::Solution;

    // Params are: (AB type, waveK), (transA, transB), loadPathAB
    class WMMATestGFX120X
        : public BaseMatrixMultiplyContextFixture<std::tuple<std::pair<rocRoller::DataType, int>,
                                                             std::pair<std::string, std::string>,
                                                             SolutionParams::LoadPath>>
    {
    };

    // Params are: (AB type, waveK), (transA, transB), loadPathAB
    class F16AccWMMATestGFX120X
        : public BaseMatrixMultiplyContextFixture<std::tuple<std::pair<rocRoller::DataType, int>,
                                                             std::pair<std::string, std::string>,
                                                             SolutionParams::LoadPath>>
    {
    };

    // Params are: A type, B type, waveK, (transA, transB), loadPathAB
    class MixedWMMATestGFX120X
        : public BaseMatrixMultiplyContextFixture<std::tuple<rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             int,
                                                             std::pair<std::string, std::string>,
                                                             SolutionParams::LoadPath>>
    {
    };

    // Params: waveK, laodPathAB
    class ABCWMMATestGFX120X
        : public BaseMatrixMultiplyContextFixture<std::tuple<int, SolutionParams::LoadPath>>
    {
    };

    TEST_P(WMMATestGFX120X, GPU_MatrixMultiplyMacroTileWMMA)
    {
        const auto [typeAndWaveK, transOp, loadPathB] = std::get<1>(GetParam());
        const auto [typeAB, waveK]                    = typeAndWaveK;
        const auto [transA, transB]                   = transOp;
        auto typeStr{"f16"};
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

    TEST_P(F16AccWMMATestGFX120X, GPU_MatrixMultiplyMacroTileWMMA)
    {
        const auto [typeAndWaveK, transOp, loadPathB] = std::get<1>(GetParam());
        const auto [dataType, waveK]                  = typeAndWaveK;
        const auto [transA, transB]                   = transOp;
        auto typeStr{"f16"};
        switch(dataType)
        {
        case DataType::Half:
            if(waveK == 16)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f16_16x16x16_f16);
            }
            else
            {
                Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
            }
            matrixMultiplyMacroTile<Half, Half, Half, Half>(
                16, 16, waveK, 1, loadPathB, transA, transB);
            break;
        case DataType::BFloat16:
            if(waveK == 16)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_bf16_16x16x16_bf16);
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

    TEST_P(WMMATestGFX120X, GPU_MatrixMultiplyABWMMA)
    {
        const auto [typeAndWaveK, transOp, loadPathAB] = std::get<1>(GetParam());
        const auto [typeAB, waveK]                     = typeAndWaveK;
        const auto [transA, transB]                    = transOp;
        auto typeStr{"f16"};
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

    TEST_P(F16AccWMMATestGFX120X, GPU_MatrixMultiplyABWMMA)
    {
        const auto [typeAndWaveK, transOp, loadPathAB] = std::get<1>(GetParam());
        const auto [typeAB, waveK]                     = typeAndWaveK;
        const auto [transA, transB]                    = transOp;
        auto typeStr{"f16"};
        switch(typeAB)
        {
        case DataType::Half:
            if(waveK == 16)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f16_16x16x16_f16);
            }
            else
            {
                Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
            }
            matrixMultiplyAB<Half, Half, Half, Half>(
                16, 16, waveK, 1, loadPathAB, transA == "T", transB == "T");
            break;
        case DataType::BFloat16:
            if(waveK == 16)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_bf16_16x16x16_bf16);
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

    TEST_P(MixedWMMATestGFX120X, GPU_MatrixMultiplyMacroTileMixedWMMA)
    {
        const auto [typeA, typeB, waveK, transOp, loadPathB] = std::get<1>(GetParam());
        const auto [transA, transB]                          = transOp;
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

    TEST_P(MixedWMMATestGFX120X, GPU_MatrixMultiplyABMixedWMMA)
    {
        const auto [typeA, typeB, waveK, transOp, loadPathAB] = std::get<1>(GetParam());
        const auto [transA, transB]                           = transOp;
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

    TEST_P(ABCWMMATestGFX120X, GPU_MatrixMultiplyABCF16AccWMMAFP16)
    {
        const auto [waveK, loadPathAB] = std::get<1>(GetParam());
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_F16_ACC);

        if(waveK == 16)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f16_16x16x16_f16);
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

    TEST_P(ABCWMMATestGFX120X, GPU_MatrixMultiplyABCF16AccWMMABFloat16)
    {
        const auto [waveK, loadPathAB] = std::get<1>(GetParam());
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_F16_ACC);

        if(waveK == 16)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_bf16_16x16x16_bf16);
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
        MatrixMultiply120X,
        WMMATestGFX120X,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX1200},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX1201}),
            ::testing::Combine(
                ::testing::Values(std::make_pair(rocRoller::DataType::Half, /*waveK*/ 16),
                                  std::make_pair(rocRoller::DataType::BFloat16, /*waveK*/ 16)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiply120X,
        F16AccWMMATestGFX120X,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX1200},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX1201}),
            ::testing::Combine(
                ::testing::Values(std::make_pair(rocRoller::DataType::Half, /*waveK*/ 16),
                                  std::make_pair(rocRoller::DataType::BFloat16, /*waveK*/ 16)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiply120X,
        MixedWMMATestGFX120X,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX1200},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX1201}),
            ::testing::Combine(
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(/*waveK*/ 16),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")),
                ::testing::Values(SolutionParams::LoadPath::BufferToLDSViaVGPR))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiplyABC120X,
        ABCWMMATestGFX120X,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX1200},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX1201}),
            ::testing::Combine(::testing::Values(/*waveK*/ 16),
                               ::testing::Values(SolutionParams::LoadPath::BufferToVGPR))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiplyABCWMMA120X,
        ABCWMMATestGFX120X,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX1200},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX1201}),
            ::testing::Combine(::testing::Values(/*waveK*/ 16),
                               ::testing::Values(SolutionParams::LoadPath::BufferToVGPR))));
} // namespace MatrixMultiplyTest