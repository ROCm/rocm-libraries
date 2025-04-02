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

#include <memory>

#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply.hpp>
#include <rocRoller/CodeGen/Arithmetic/Utility.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    namespace InstructionGenerators
    {
        RegisterComponent(MatrixMultiplyGenerator);

        const std::string MatrixMultiply::Basename = "MatrixMultiply";

        std::string typeStr(auto dtype)
        {
            switch(dtype)
            {
            case DataType::Float:
                return "f32";
            case DataType::Halfx2:
                return "f16";
            case DataType::BFloat16x2:
                return "bf16";
            case DataType::FP8x4:
                return "_fp8_fp8";
            case DataType::BF8x4:
                return "_bf8_bf8";
            case DataType::FP6x16:
            case DataType::BF6x16:
            case DataType::FP4x8:
                return "_f8f6f4";
            default:
                Throw<FatalError>("Unable to determine MFMA type: unhandled data type.",
                                  ShowValue(dtype));
            }
        }

        Generator<Instruction> MatrixMultiplyGenerator::mul(Register::ValuePtr D,
                                                            Register::ValuePtr A,
                                                            Register::ValuePtr B,
                                                            Register::ValuePtr C,
                                                            int                M,
                                                            int                N,
                                                            int                K,
                                                            int                BATCH)
        {
            AssertFatal(A != nullptr);
            AssertFatal(B != nullptr);
            AssertFatal(C != nullptr);

            auto typeA = A->variableType().dataType;
            auto typeB = B->variableType().dataType;
            auto typeC = C->variableType().dataType;
            auto typeD = C->variableType().dataType;
            if(D != nullptr)
                typeD = D->variableType().dataType;

            auto const lanesPerWavefront = m_context->targetArchitecture().GetCapability(
                GPUCapability::DefaultWavefrontSize);
            auto const packingA = DataTypeInfo::Get(typeA).packing;
            auto const packingB = DataTypeInfo::Get(typeB).packing;
            AssertFatal(M > 0 && N > 0 && K > 0 && BATCH > 0 && lanesPerWavefront > 0,
                        "Invalid inputs",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(K),
                        ShowValue(BATCH),
                        ShowValue(lanesPerWavefront));
            AssertFatal(A->valueCount() * packingA == (size_t)M * K * BATCH / lanesPerWavefront,
                        "A matrix size mismatch",
                        ShowValue(M),
                        ShowValue(K),
                        ShowValue(BATCH),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * K * BATCH / lanesPerWavefront),
                        ShowValue(A->valueCount()),
                        ShowValue(packingA));
            AssertFatal(B->valueCount() * packingB == (size_t)K * N * BATCH / lanesPerWavefront,
                        "B matrix size mismatch",
                        ShowValue(K),
                        ShowValue(N),
                        ShowValue(BATCH),
                        ShowValue(lanesPerWavefront),
                        ShowValue(K * N * BATCH / lanesPerWavefront),
                        ShowValue(B->valueCount()),
                        ShowValue(packingB));
            AssertFatal(C->valueCount() == (size_t)M * N * BATCH / lanesPerWavefront,
                        "C matrix size mismatch",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(BATCH),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * N * BATCH / lanesPerWavefront),
                        ShowValue(C->valueCount()));
            AssertFatal(D->valueCount() == (size_t)M * N * BATCH / lanesPerWavefront,
                        "D matrix size mismatch",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(BATCH),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * N * BATCH / lanesPerWavefront),
                        ShowValue(D->valueCount()));
            AssertFatal(A->regType() == Register::Type::Vector,
                        "Invalid LHS (A) register type",
                        ShowValue(A->regType()));
            AssertFatal(B->regType() == Register::Type::Vector,
                        "Invalid B (B) register type",
                        ShowValue(B->regType()));
            AssertFatal(C->variableType() == D->variableType(),
                        "Invalid D/R2HS (D/C) data types",
                        ShowValue(C->variableType()));

            auto const& arch = m_context->targetArchitecture();
            if(arch.HasCapability(GPUCapability::HasAccCD))
            {
                AssertFatal(D->regType() == Register::Type::Accumulator,
                            "Invalid DEST (D) register type",
                            ShowValue(D->regType()));
            }
            else
            {
                AssertFatal(D->regType() == Register::Type::Vector,
                            "Invalid DEST (D) register type",
                            ShowValue(D->regType()));
            }

            std::string inputType;
            std::string modifier;

            const auto isF8
                = [](DataType type) { return type == DataType::FP8x4 || type == DataType::BF8x4; };

            const auto isFP8 = [](DataType type) { return type == DataType::FP8x4; };

            const auto isBF8 = [](DataType type) { return type == DataType::BF8x4; };

            const auto isF16 = [](DataType type) {
                return type == DataType::Halfx2 || type == DataType::BFloat16x2;
            };

            const auto isFP16 = [](DataType type) { return type == DataType::Halfx2; };

            const auto isBF16 = [](DataType type) { return type == DataType::BFloat16x2; };

            if(arch.HasCapability(GPUCapability::HasWMMA))
            {
                AssertFatal((M == 16) && (N == 16) && (K == 16 || K == 32),
                            "Invalid inputs",
                            ShowValue(M),
                            ShowValue(N),
                            ShowValue(K));

                if(isF8(typeA) && isF8(typeB))
                {
                    inputType = isFP8(typeA) ? "fp8" : "bf8";
                    inputType += isFP8(typeB) ? "_fp8" : "_bf8";
                }
                else if(typeA == typeB && isF16(typeA))
                {
                    inputType = isFP16(typeA) ? "f16" : "bf16";
                }
                else
                {
                    Throw<FatalError>("Matrix Multiplication is not supported for",
                                      arch.target().toString(),
                                      " with A=",
                                      typeA,
                                      "and B=",
                                      typeB);
                }

                auto wmma = concatenate(
                    "v_wmma_", typeStr(typeD), "_", M, "x", N, "x", K, "_", inputType);
                co_yield_(Instruction(wmma, {D}, {A, B, C}, {}, ""));
            }
            else if(arch.HasCapability(GPUCapability::HasMFMA))
            {
                if(typeA == typeB)
                {
                    // Uniform type for A and B.  Result will be similar
                    // to "f16", and may be "_f8f6f4".
                    inputType = typeStr(typeA);

                    // For F8 types, result will be "_fp8_fp8" (or "bf8").
                    // Change this to "_f8f6f4" for 32x32x64 and 16x16x128
                    // tile sizes.
                    if(isF8(typeA))
                    {
                        if((M == 32 && N == 32 && K == 64) || (M == 16 && N == 16 && K == 128))
                            inputType = "_f8f6f4";
                    }

                    if(isBF16(typeA))
                    {
                        if(((M == 32) && (N == 32) && (K == 8))
                           || ((M == 16) && (N == 16) && (K == 16)))
                        {
                            inputType = "bf16_1k";
                        }
                    }
                    // For F16 types, result will be "f16" (or "bf16").
                    if((M == 16 && N == 16 && K == 32) || (M == 32 && N == 32 && K == 16))
                    {
                        if(isFP16(typeA))
                        {
                            inputType = "_f16";
                        }
                        if(isBF16(typeA))
                        {
                            inputType = "_bf16";
                        }
                    }
                }
                else
                {
                    // Mixed types for A and B.  Only works for lower
                    // precisions.
                    auto segA = DataTypeInfo::Get(typeA).segmentVariableType;
                    auto segB = DataTypeInfo::Get(typeB).segmentVariableType;
                    AssertFatal(DataTypeInfo::Get(segA).elementBits <= 8,
                                "Mixed MFMA inputs (A) must be low precision.",
                                ShowValue(typeA));
                    AssertFatal(DataTypeInfo::Get(segB).elementBits <= 8,
                                "Mixed MFMA inputs (B) must be low precision.",
                                ShowValue(typeB));
                    inputType = "_f8f6f4";
                }

                // TODO: _fp8_bf8 not handled
                if(inputType == "_f8f6f4")
                {
                    if(!((M == 32 && N == 32 && K == 64) || (M == 16 && N == 16 && K == 128)))
                    {
                        Throw<FatalError>(
                            "Invalid F8F6F4 MFMA size.", ShowValue(M), ShowValue(N), ShowValue(K));
                    }

                    modifier = concatenate("cbsz:",
                                           Arithmetic::getModifier(typeA),
                                           " blgp:",
                                           Arithmetic::getModifier(typeB));
                }

                auto mfma
                    = concatenate("v_mfma_", typeStr(typeD), "_", M, "x", N, "x", K, inputType);

                co_yield_(Instruction(mfma, {D}, {A, B, C}, {modifier}, ""));
            }
            else
            {
                Throw<FatalError>("Matrix Multiplication is not supported for",
                                  arch.target().toString());
            }
        }
    }
}
