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

#include <rocRoller/CodeGen/Utils.hpp>

#include "MatrixMultiplyTestBase.hpp"
#include <common/SourceMatcher.hpp>

using namespace rocRoller;

namespace MatrixMultiplyTest
{
    class MatrixMultiplyTestGPU : public BaseMatrixMultiplyContextFixture<>
    {
    };

    // Params are: AB type, K tile size, (transA, transB)
    class MatrixMultiplyTestGPUF16
        : public BaseMatrixMultiplyContextFixture<
              std::tuple<rocRoller::DataType, int, std::pair<std::string, std::string>>>
    {
    };

    class MatrixMultiplyTestGPUF8 : public BaseMatrixMultiplyContextFixture<rocRoller::DataType>
    {
    };

    // Params are: AB type, K tile size, (transA, transB)
    class MatrixMultiplyF8F6F4TestGPU
        : public BaseMatrixMultiplyContextFixture<
              std::tuple<rocRoller::DataType, int, std::pair<std::string, std::string>>>
    {
    };

    // Params are: A type, B type, K tile size, (transA, transB)
    class MatrixMultiplyMixedTestGPU
        : public BaseMatrixMultiplyContextFixture<std::tuple<rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             int,
                                                             std::pair<std::string, std::string>>>
    {
    };

    class MatrixMultiplyTestGPUBFloat16
        : public BaseMatrixMultiplyContextFixture<std::tuple<int, int, int>>
    {
    };

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyMacroTile)
    {
        matrixMultiplyMacroTile<float, float, float>(32, 32, 2, 1);
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyMacroTileFP16)
    {
        matrixMultiplyMacroTile<Half, Half, Half>(32, 32, 8, 1, false);

        if(!commandKernel)
            return;

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int expectedLocalWriteOffset = 0;
        int numLocalRead             = 0;
        int expectedLocalReadOffset  = 0;
        for(auto const& instruction : instructions)
        {
            // Count the number of ds_write_b128 instructions and make sure they have
            // the expected offset values
            if(instruction.starts_with("ds_write_b128"))
            {
                if(expectedLocalWriteOffset > 0)
                    EXPECT_TRUE(instruction.ends_with("offset:"
                                                      + std::to_string(expectedLocalWriteOffset)));
                expectedLocalWriteOffset += 64;
            }

            if(instruction.starts_with("ds_read_u16"))
            {
                numLocalRead++;

                if(expectedLocalReadOffset > 0)
                    EXPECT_TRUE(
                        instruction.ends_with("offset:" + std::to_string(expectedLocalReadOffset)));

                if(numLocalRead % 4 == 0)
                {
                    expectedLocalReadOffset = numLocalRead / 4 * 512;
                }
                else
                {
                    expectedLocalReadOffset += 64;
                }
            }
        }

        EXPECT_EQ(expectedLocalWriteOffset, 128);
        EXPECT_EQ(numLocalRead, 16);
    }

    TEST_P(MatrixMultiplyTestGPUBFloat16, GPU_MatrixMultiplyMacroTile_FP32_BF16)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x4);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x8_1k);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x8);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x16_1k);

        auto [mfma_m, mfma_n, mfma_k] = std::get<std::tuple<int, int, int>>(GetParam());

        matrixMultiplyMacroTile<BFloat16, BFloat16, float>(mfma_m, mfma_n, mfma_k, 1, false);
    }

    TEST_P(MatrixMultiplyTestGPUBFloat16, GPU_MatrixMultiplyMacroTile_BF16_BF16)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x4);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x8_1k);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x8);
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x16_1k);

        auto [mfma_m, mfma_n, mfma_k] = std::get<std::tuple<int, int, int>>(GetParam());

        matrixMultiplyMacroTile<BFloat16, BFloat16, BFloat16>(mfma_m, mfma_n, mfma_k, 1, false);
    }

    TEST_P(MatrixMultiplyTestGPUF16, GPU_MatrixMultiplyMacroTileF16)
    {
        auto [typeAB, MFMAK, transOp] = std::get<1>(GetParam());

        uint const waveM = (MFMAK == 32) ? 16 : 32;
        uint const waveN = (MFMAK == 32) ? 16 : 32;
        uint const waveK = MFMAK;

        auto const transA = transOp.first;
        auto const transB = transOp.second;

        auto typeStr = "f16";
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
            matrixMultiplyMacroTile<Half, Half, float>(
                waveM, waveN, waveK, 1, true, transA, transB);
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
            matrixMultiplyMacroTile<BFloat16, BFloat16, float>(
                waveM, waveN, waveK, 1, true, transA, transB);
            typeStr = "bf16";
            break;
        default:
            Throw<FatalError>(fmt::format("Unexpected data type: {}. (Allowed: Half and Bfloat16)",
                                          toString(typeAB)));
        }

        std::string generatedCode = m_context->instructions()->toString();

        uint const elementBits = DataTypeInfo::Get(typeAB).elementBits;

        auto const& arch = m_context->targetArchitecture();

        std::string const mfmaMnemonic{
            fmt::format("v_mfma_f32_{}x{}x{}_{}", waveM, waveN, waveK, typeStr)};
        std::string const trLoadMnemonic{transposeLoadMnemonic(arch, elementBits)};

        uint const numMFMAs            = 4;
        uint const elementsPerWavetile = waveM * waveK / 64;
        uint const elementsPerTrLoad   = bitsPerTransposeLoad(arch, elementBits) / elementBits;
        uint const trLoadsPerMFMA      = elementsPerWavetile / elementsPerTrLoad;
        uint       expectedTrLoads     = 0;
        if(transA == "N")
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;
        if(transB == "T")
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;

        EXPECT_EQ(countSubstring(generatedCode, "v_mfma"), numMFMAs);
        EXPECT_EQ(countSubstring(generatedCode, mfmaMnemonic), numMFMAs);
        EXPECT_EQ(countSubstring(generatedCode, trLoadMnemonic), expectedTrLoads);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyMacroTileF8_16x16x32_NN)
    {
        bool const isFP8 = std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8;
        if(isFP8)
            matrixMultiplyMacroTile<FP8, FP8, float>(16, 16, 32, 1, false, "N", "N");
        else
            matrixMultiplyMacroTile<BF8, BF8, float>(16, 16, 32, 1, false, "N", "N");

        if(!commandKernel)
            return;

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int               expectedLocalWriteOffset = 0;
        int               numLocalRead             = 0;
        int               expectedLocalReadOffset  = 0;
        int               numMFMA                  = 0;
        std::string const mfma_pattern
            = isFP8 ? "v_mfma_f32_16x16x32_fp8_fp8" : "v_mfma_f32_16x16x32_bf8_bf8";
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with(mfma_pattern))
                numMFMA++;

            // Count the number of ds_write_b128 instructions and make sure they have
            // the expected offset values
            if(instruction.starts_with("ds_write_b128"))
            {
                if(expectedLocalWriteOffset > 0)
                    EXPECT_TRUE(instruction.ends_with("offset:"
                                                      + std::to_string(expectedLocalWriteOffset)));
                expectedLocalWriteOffset += 1024;
            }

            if(instruction.starts_with("ds_read_u8"))
            {
                numLocalRead++;

                if(expectedLocalReadOffset > 0)
                    EXPECT_TRUE(
                        instruction.ends_with("offset:" + std::to_string(expectedLocalReadOffset)));

                if(numLocalRead % 8 == 0)
                {
                    expectedLocalReadOffset = numLocalRead / 8 * 512;
                }
                else
                {
                    expectedLocalReadOffset += 16;
                }
            }
        }

        EXPECT_EQ(expectedLocalWriteOffset, 1024);
        EXPECT_EQ(numLocalRead, 16);
        EXPECT_EQ(numMFMA, 2);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyMacroTileF8_32x32x16_NN)
    {
        bool const isFP8 = std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8;
        if(isFP8)
            matrixMultiplyMacroTile<FP8, FP8, float>(32, 32, 16, 1, false, "N", "N");
        else
            matrixMultiplyMacroTile<BF8, BF8, float>(32, 32, 16, 1, false, "N", "N");

        if(!commandKernel)
            return;

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int               expectedLocalWriteOffset = 0;
        int               numLocalRead             = 0;
        int               expectedLocalReadOffset  = 0;
        int               numMFMA                  = 0;
        std::string const mfma_pattern
            = isFP8 ? "v_mfma_f32_32x32x16_fp8_fp8" : "v_mfma_f32_32x32x16_bf8_bf8";
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with(mfma_pattern))
                numMFMA++;

            // Count the number of ds_write_b128 instructions and make sure they have
            // the expected offset values
            if(instruction.starts_with("ds_write_b128"))
            {
                if(expectedLocalWriteOffset > 0)
                    EXPECT_TRUE(instruction.ends_with("offset:"
                                                      + std::to_string(expectedLocalWriteOffset)));
                expectedLocalWriteOffset += 1024;
            }

            if(instruction.starts_with("ds_read_u8"))
            {
                numLocalRead++;

                if(expectedLocalReadOffset > 0)
                    EXPECT_TRUE(
                        instruction.ends_with("offset:" + std::to_string(expectedLocalReadOffset)));

                if(numLocalRead % 8 == 0)
                {
                    expectedLocalReadOffset = numLocalRead / 8 * 512;
                }
                else
                {
                    expectedLocalReadOffset += 32;
                }
            }
        }

        EXPECT_EQ(expectedLocalWriteOffset, 1024);
        EXPECT_EQ(numLocalRead, 16);
        EXPECT_EQ(numMFMA, 2);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyMacroTileF8_16x16x32_TN)
    {
        bool const isFP8 = std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8;
        if(isFP8)
            matrixMultiplyMacroTile<FP8, FP8, float>(16, 16, 32, 1, true, "T", "N");
        else
            matrixMultiplyMacroTile<BF8, BF8, float>(16, 16, 32, 1, true, "T", "N");
    }

    TEST_P(MatrixMultiplyF8F6F4TestGPU, GPU_MatrixMultiplyMacroTileF8F6F4)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        auto [typeAB, MFMAK, transOp] = std::get<1>(GetParam());

        uint const waveM = (MFMAK == 128) ? 16 : 32;
        uint const waveN = (MFMAK == 128) ? 16 : 32;
        uint const waveK = MFMAK;

        std::string const mfmaMnemonic{
            fmt::format("v_mfma_f32_{}x{}x{}_f8f6f4", waveM, waveN, waveK)};

        auto const [transA, transB] = transOp;

        auto const& arch = m_context->targetArchitecture();

        uint const        elementBits = DataTypeInfo::Get(typeAB).elementBits;
        std::string const trLoadMnemonic{transposeLoadMnemonic(arch, elementBits)};

        std::string modifiers{"cbsz:0b000 blgp:0b000"};

        switch(typeAB)
        {
        case DataType::FP8:
            matrixMultiplyMacroTile<FP8, FP8, float>(waveM, waveN, waveK, 1, true, transA, transB);
            break;
        case DataType::BF8:
            matrixMultiplyMacroTile<BF8, BF8, float>(waveM, waveN, waveK, 1, true, transA, transB);
            modifiers = "cbsz:0b001 blgp:0b001";
            break;
        case DataType::FP6:
            matrixMultiplyMacroTile<FP6, FP6, float>(waveM, waveN, waveK, 1, true, transA, transB);
            modifiers = "cbsz:0b010 blgp:0b010";
            break;
        case DataType::BF6:
            matrixMultiplyMacroTile<BF6, BF6, float>(waveM, waveN, waveK, 1, true, transA, transB);
            modifiers = "cbsz:0b011 blgp:0b011";
            break;
        case DataType::FP4:
            matrixMultiplyMacroTile<FP4, FP4, float>(waveM, waveN, waveK, 1, true, transA, transB);
            modifiers = "cbsz:0b100 blgp:0b100";
            break;
        default:
            Throw<FatalError>(
                fmt::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }

        uint const numMFMAs            = 2;
        uint const elementsPerWavetile = waveM * waveK / 64;
        uint const elementsPerTrLoad   = bitsPerTransposeLoad(arch, elementBits) / elementBits;
        uint const trLoadsPerMFMA      = elementsPerWavetile / elementsPerTrLoad;
        uint       expectedTrLoads     = 0;
        if(transA == "N")
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;
        if(transB == "T")
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "v_mfma"), 2);
        EXPECT_EQ(countSubstring(generatedCode, mfmaMnemonic), 2);
        EXPECT_EQ(countSubstring(generatedCode, trLoadMnemonic), expectedTrLoads);
        EXPECT_EQ(countSubstring(generatedCode, modifiers), 2);
    }

    TEST_P(MatrixMultiplyF8F6F4TestGPU, GPU_ScaledMatrixMultiplyMacroTileF8F6F4)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);

        auto [typeAB, MFMAK, transOp] = std::get<1>(GetParam());

        uint const waveM = (MFMAK == 128) ? 16 : 32;
        uint const waveN = (MFMAK == 128) ? 16 : 32;
        uint const waveK = MFMAK;

        std::string const mfmaMnemonic{
            fmt::format("v_mfma_scale_f32_{}x{}x{}_f8f6f4", waveM, waveN, waveK)};

        auto const [transA, transB] = transOp;

        auto const& arch = m_context->targetArchitecture();

        uint        elementBits = DataTypeInfo::Get(typeAB).elementBits;
        std::string trLoadMnemonic{transposeLoadMnemonic(arch, elementBits)};

        // TODO: enable non-TN 16x16x128 tests
        if((transA != "T" || transB != "N") && MFMAK == 128)
        {
            GTEST_SKIP() << "FIXME: Skipping scaled non-TN 16x16x128 tests";
        }

        std::string modifiers{"cbsz:0b000 blgp:0b000"};

        ScaleParams const scaleParams
            = {.scaleTypeA = DataType::E8M0, .scaleTypeB = DataType::E8M0, .scaleBlockSize = 32};

        switch(typeAB)
        {
        case DataType::FP8:
            matrixMultiplyMacroTile<FP8, FP8, float>(
                waveM, waveN, waveK, 1, true, transA, transB, scaleParams);
            break;
        case DataType::BF8:
            matrixMultiplyMacroTile<BF8, BF8, float>(
                waveM, waveN, waveK, 1, true, transA, transB, scaleParams);
            modifiers = "cbsz:0b001 blgp:0b001";
            break;
        case DataType::FP6:
            matrixMultiplyMacroTile<FP6, FP6, float>(
                waveM, waveN, waveK, 1, true, transA, transB, scaleParams);
            modifiers = "cbsz:0b010 blgp:0b010";
            break;
        case DataType::BF6:
            matrixMultiplyMacroTile<BF6, BF6, float>(
                waveM, waveN, waveK, 1, true, transA, transB, scaleParams);
            modifiers = "cbsz:0b011 blgp:0b011";
            break;
        case DataType::FP4:
            matrixMultiplyMacroTile<FP4, FP4, float>(
                waveM, waveN, waveK, 1, true, transA, transB, scaleParams);
            modifiers = "cbsz:0b100 blgp:0b100";
            break;
        default:
            Throw<FatalError>(
                fmt::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }

        uint const numMFMAs            = 2;
        uint const elementsPerWavetile = waveM * waveK / 64;
        uint const elementsPerTrLoad   = bitsPerTransposeLoad(arch, elementBits) / elementBits;
        uint const trLoadsPerMFMA      = elementsPerWavetile / elementsPerTrLoad;
        uint       expectedTrLoads     = 0;
        if(transA == "N")
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;
        if(transB == "T")
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "v_mfma"), 2);
        EXPECT_EQ(countSubstring(generatedCode, mfmaMnemonic), 2);
        EXPECT_EQ(countSubstring(generatedCode, trLoadMnemonic), expectedTrLoads);
        EXPECT_EQ(countSubstring(generatedCode, modifiers), 2);
    }

    TEST_P(MatrixMultiplyMixedTestGPU, GPU_MatrixMultiplyMacroTileMixed)
    {
        auto [typeA, typeB, MFMAK, transOp] = std::get<1>(GetParam());

        int wave_m = (MFMAK == 128) ? 16 : 32;
        int wave_n = (MFMAK == 128) ? 16 : 32;
        int wave_k = MFMAK;

        auto [transA, transB] = transOp;

        matrixMultiplyMacroTileMixed(typeA, typeB, wave_m, wave_n, wave_k, 1, true, transA, transB);
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyAB)
    {
        matrixMultiplyAB<float, float, float>(32, 32, 2, 1);
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyABFP16)
    {
        matrixMultiplyAB<Half, Half, Half>(32, 32, 8, 1);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyABF8_16x16x32)
    {
        if(std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8)
            matrixMultiplyAB<FP8, FP8, float>(16, 16, 32, 1);
        else
            matrixMultiplyAB<BF8, BF8, float>(16, 16, 32, 1);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyABF8_32x32x16)
    {
        if(std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8)
            matrixMultiplyAB<FP8, FP8, float>(32, 32, 16, 1);
        else
            matrixMultiplyAB<BF8, BF8, float>(32, 32, 16, 1);
    }

    TEST_P(MatrixMultiplyF8F6F4TestGPU, GPU_MatrixMultiplyABF8F6F4)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        auto [typeAB, MFMAK, transOp] = std::get<1>(GetParam());

        uint const waveM = (MFMAK == 128) ? 16 : 32;
        uint const waveN = (MFMAK == 128) ? 16 : 32;
        uint const waveK = MFMAK;

        std::string const mfmaMnemonic{
            fmt::format("v_mfma_f32_{}x{}x{}_f8f6f4", waveM, waveN, waveK)};

        auto const transA = transOp.first == "T";
        auto const transB = transOp.second == "T";

        auto const& arch = m_context->targetArchitecture();

        uint const        elementBits = DataTypeInfo::Get(typeAB).elementBits;
        std::string const trLoadMnemonic{transposeLoadMnemonic(arch, elementBits)};

        std::string modifiers{"cbsz:0b000 blgp:0b000"};

        switch(typeAB)
        {
        case DataType::FP8:
            matrixMultiplyAB<FP8, FP8, float>(waveM, waveN, waveK, 1, true, transA, transB);
            break;
        case DataType::BF8:
            matrixMultiplyAB<BF8, BF8, float>(waveM, waveN, waveK, 1, true, transA, transB);
            modifiers = "cbsz:0b001 blgp:0b001";
            break;
        case DataType::FP6:
            matrixMultiplyAB<FP6, FP6, float>(waveM, waveN, waveK, 1, true, transA, transB);
            modifiers = "cbsz:0b010 blgp:0b010";
            break;
        case DataType::BF6:
            matrixMultiplyAB<BF6, BF6, float>(waveM, waveN, waveK, 1, true, transA, transB);
            modifiers = "cbsz:0b011 blgp:0b011";
            break;
        case DataType::FP4:
            matrixMultiplyAB<FP4, FP4, float>(waveM, waveN, waveK, 1, true, transA, transB);
            modifiers = "cbsz:0b100 blgp:0b100";
            break;
        default:
            Throw<FatalError>(
                fmt::format("Unexpected data type: {}. (Allowed FP8, BF8, FP6, BF6, and FP4)",
                            toString(typeAB)));
        }

        uint const numMFMAs            = 2;
        uint const elementsPerWavetile = waveM * waveK / 64;
        uint const elementsPerTrLoad   = bitsPerTransposeLoad(arch, elementBits) / elementBits;
        uint const trLoadsPerMFMA      = elementsPerWavetile / elementsPerTrLoad;
        uint       expectedTrLoads     = 0;
        if(!transA)
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;
        if(transB)
            expectedTrLoads += numMFMAs * trLoadsPerMFMA;

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "v_mfma"), numMFMAs);
        EXPECT_EQ(countSubstring(generatedCode, mfmaMnemonic), numMFMAs);
        EXPECT_EQ(countSubstring(generatedCode, modifiers), numMFMAs);
        EXPECT_EQ(countSubstring(generatedCode, trLoadMnemonic), expectedTrLoads);
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyABC)
    {
        matrixMultiplyABC<float>(32, 32, 2, 1);
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyABCFP16)
    {
        matrixMultiplyABC<Half>(32, 32, 8, 1);
    }

    INSTANTIATE_TEST_SUITE_P(MatrixMultiplyTest, MatrixMultiplyTestGPU, mfmaSupportedISATuples());

    INSTANTIATE_TEST_SUITE_P(MatrixMultiplyTest,
                             MatrixMultiplyTestGPUF8,
                             ::testing::Combine(mfmaSupportedISAValues(),
                                                ::testing::Values(rocRoller::DataType::FP8,
                                                                  rocRoller::DataType::BF8)));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiplyTest,
        MatrixMultiplyTestGPUF16,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX950},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.sramecc = true}},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.xnack = true}}),
            ::testing::Combine(::testing::Values(rocRoller::DataType::Half,
                                                 rocRoller::DataType::BFloat16),
                               ::testing::Values(16, 32),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiplyTest,
        MatrixMultiplyF8F6F4TestGPU,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX950},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.sramecc = true}},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.xnack = true}}),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(64, 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiplyTest,
        MatrixMultiplyMixedTestGPU,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX950},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.sramecc = true}},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.xnack = true}}),
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
                               ::testing::Values(64, 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    // Params are: A type, B type, scale pair, K tile size
    class ScaledMatrixMultiplyMixedTestGPU
        : public BaseMatrixMultiplyContextFixture<std::tuple<rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             int,
                                                             std::pair<std::string, std::string>>>
    {
    };

    TEST_P(ScaledMatrixMultiplyMixedTestGPU, GPU_ScaledMatrixMultiplyMacroTileMixed)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);

        auto [typeA, typeB, MFMAK, transOp] = std::get<1>(GetParam());

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        auto [transA, transB] = transOp;

        // TODO: enable non-TN 16x16x128 tests
        if((transA != "T" || transB != "N") && MFMAK == 128)
        {
            GTEST_SKIP() << "FIXME: Skipping scaled non-TN 16x16x128 tests";
        }

        ScaleParams const scaleParams
            = {.scaleTypeA = DataType::E8M0, .scaleTypeB = DataType::E8M0, .scaleBlockSize = 32};

        matrixMultiplyMacroTileMixed(
            typeA, typeB, waveM, waveN, waveK, 1, true, transA, transB, scaleParams);
    }

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiplyTest,
        ScaledMatrixMultiplyMixedTestGPU,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX950},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.sramecc = true}},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.xnack = true}}),
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
                               ::testing::Values(64, 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    class ScaledMMTest
        : public BaseMatrixMultiplyContextFixture<std::tuple<rocRoller::DataType,
                                                             rocRoller::DataType,
                                                             std::pair<uint8_t, uint8_t>,
                                                             int,
                                                             std::pair<std::string, std::string>>>
    {
    };

    template <typename TA, typename TB>
    void exeScaledCPUMM(unsigned    M,
                        unsigned    N,
                        unsigned    K,
                        const float scaleA,
                        const float scaleB,
                        float       alpha,
                        double      err,
                        bool        transA,
                        bool        transB,
                        const uint  scaleBlockSize)
    {
        auto dataTypeA = TypeInfo<TA>::Var.dataType;
        auto dataTypeB = TypeInfo<TB>::Var.dataType;

        TensorDescriptor descA(dataTypeA, {M, K}, "T");
        TensorDescriptor descB(dataTypeB, {K, N}, "T");

        auto A     = DGenVector<TA>(descA, -1.0, 1.0, 9861u);
        auto B     = DGenVector<TB>(descB, -1.0, 1.0, 9861u);
        auto C     = std::vector<float>(M * N);
        auto D     = std::vector<float>(M * N);
        auto ref_D = std::vector<float>(M * N);

        auto AX = std::vector<uint8_t>(M * K / scaleBlockSize);
        auto BX = std::vector<uint8_t>(K * N / scaleBlockSize);
        std::fill(AX.begin(), AX.end(), scaleA);
        std::fill(BX.begin(), BX.end(), scaleB);

        // TODO: now only works for _TN for A and B, need to enable other data layout
        ScaledCPUMM(D, C, A, B, AX, BX, M, N, K, alpha, 0.0, transA, transB, scaleBlockSize);

        alpha *= std::pow(2.0f, int(scaleA) - 127) * std::pow(2.0f, int(scaleB) - 127);

        CPUMM(ref_D, C, A, B, M, N, K, alpha, 0.0, transA, transB);

        double rnorm = relativeNormL2(D, ref_D);
        Log::info("RNorm is {}", rnorm);
        ASSERT_LT(rnorm, err);
    }

    template <typename TA>
    void scaledCPUMMMixed(rocRoller::DataType typeB,
                          const int           m,
                          const int           n,
                          const int           k,
                          const float         scaleA,
                          const float         scaleB,
                          float               alpha,
                          double              err,
                          bool                transA,
                          bool                transB,
                          const uint          scaleBlockSize = 32)
    {
        if(typeB == rocRoller::DataType::FP8)
            exeScaledCPUMM<TA, FP8>(
                m, n, k, scaleA, scaleB, alpha, err, transA, transB, scaleBlockSize);
        else if(typeB == rocRoller::DataType::BF8)
            exeScaledCPUMM<TA, BF8>(
                m, n, k, scaleA, scaleB, alpha, err, transA, transB, scaleBlockSize);
        else if(typeB == rocRoller::DataType::FP6)
            exeScaledCPUMM<TA, FP6>(
                m, n, k, scaleA, scaleB, alpha, err, transA, transB, scaleBlockSize);
        else if(typeB == rocRoller::DataType::BF6)
            exeScaledCPUMM<TA, BF6>(
                m, n, k, scaleA, scaleB, alpha, err, transA, transB, scaleBlockSize);
        else if(typeB == rocRoller::DataType::FP4)
            exeScaledCPUMM<TA, FP4>(
                m, n, k, scaleA, scaleB, alpha, err, transA, transB, scaleBlockSize);
        else
            Throw<FatalError>("Invalid type.");
    }

    void scaledCPUMMMixed(rocRoller::DataType typeA,
                          rocRoller::DataType typeB,
                          const int           m,
                          const int           n,
                          const int           k,
                          const float         scaleA,
                          const float         scaleB,
                          float               alpha,
                          double              err,
                          bool                transA,
                          bool                transB,
                          const uint          scaleBlockSize = 32)
    {
        if(typeA == rocRoller::DataType::FP8)
            scaledCPUMMMixed<FP8>(
                typeB, m, n, k, scaleA, scaleB, alpha, err, transA, transB, scaleBlockSize);
        else if(typeA == rocRoller::DataType::BF8)
            scaledCPUMMMixed<BF8>(
                typeB, m, n, k, scaleA, scaleB, alpha, err, transA, transB, scaleBlockSize);
        else if(typeA == rocRoller::DataType::FP6)
            scaledCPUMMMixed<FP6>(
                typeB, m, n, k, scaleA, scaleB, alpha, err, transA, transB, scaleBlockSize);
        else if(typeA == rocRoller::DataType::BF6)
            scaledCPUMMMixed<BF6>(
                typeB, m, n, k, scaleA, scaleB, alpha, err, transA, transB, scaleBlockSize);
        else if(typeA == rocRoller::DataType::FP4)
            scaledCPUMMMixed<FP4>(
                typeB, m, n, k, scaleA, scaleB, alpha, err, transA, transB, scaleBlockSize);
        else
            Throw<FatalError>("Invalid type.");
    }

    TEST_P(ScaledMMTest, ScaledMMTestCPU)
    {
        auto [typeA, typeB, scales, MFMAK, transOp] = std::get<1>(GetParam());

        auto [scaleA, scaleB] = scales;
        auto [transA, transB] = transOp;

        int M = (MFMAK == 128) ? 16 : 32;
        int N = (MFMAK == 128) ? 16 : 32;
        int K = MFMAK;

        float alpha = 1.0f;

        scaledCPUMMMixed(
            typeA, typeB, M, N, K, scaleA, scaleB, alpha, 1.e-5, transA == "T", transB == "T");
    }

    INSTANTIATE_TEST_SUITE_P(
        ScaledMMCPU,
        ScaledMMTest,
        ::testing::Combine(
            ::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX950},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.sramecc = true}},
                              GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.xnack = true}}),
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
                               ::testing::Values(std::pair<uint8_t, uint8_t>{125u, 125u},
                                                 std::pair<uint8_t, uint8_t>{125u, 128u},
                                                 std::pair<uint8_t, uint8_t>{128u, 125u},
                                                 std::pair<uint8_t, uint8_t>{128u, 128u}),
                               ::testing::Values(64, 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        MatrixMultiplyTestGPUBFloat16,
        MatrixMultiplyTestGPUBFloat16,
        ::testing::Combine(mfmaSupportedISAValues(),
                           ::testing::Values(std::tuple<int, int, int>{32, 32, 4},
                                             std::tuple<int, int, int>{16, 16, 8})));
} // namespace MatrixMultiplyTest
