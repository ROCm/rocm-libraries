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

#include <rocRoller/CodeGen/Arithmetic/ScaledMatrixMultiply.hpp>
#include <rocRoller/CodeGen/Arithmetic/Utility.hpp>
#include <rocRoller/CodeGen/CrashKernelGenerator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    namespace InstructionGenerators
    {
        RegisterComponent(ScaledMatrixMultiplyGenerator);

        const std::string ScaledMatrixMultiply::Basename = "ScaledMatrixMultiply";

        Generator<Instruction>
            ScaledMatrixMultiplyGenerator::mul(Register::ValuePtr  dest,
                                               Register::ValuePtr  matA,
                                               Register::ValuePtr  matB,
                                               Register::ValuePtr  matC,
                                               Register::ValuePtr  scaleA,
                                               Register::ValuePtr  scaleB,
                                               int                 M,
                                               int                 N,
                                               int                 K,
                                               std::optional<uint> maybeScaleBlockSize)
        {
            AssertFatal(matA != nullptr);
            AssertFatal(matB != nullptr);
            AssertFatal(matC != nullptr);
            AssertFatal(scaleA != nullptr);
            AssertFatal(scaleB != nullptr);

            auto const lanesPerWavefront = m_context->targetArchitecture().GetCapability(
                GPUCapability::DefaultWavefrontSize);
            AssertFatal(M > 0 && N > 0 && K > 0 && lanesPerWavefront > 0,
                        "Invalid inputs",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(K),
                        ShowValue(lanesPerWavefront));
            auto const packingA = DataTypeInfo::Get(matA->variableType()).packing;
            AssertFatal(matA->valueCount() * packingA == (size_t)M * K / lanesPerWavefront,
                        "A matrix size mismatch",
                        ShowValue(M),
                        ShowValue(K),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * K / lanesPerWavefront),
                        ShowValue(matA->valueCount()),
                        ShowValue(packingA));
            auto const packingB = DataTypeInfo::Get(matB->variableType()).packing;
            AssertFatal(matB->valueCount() * packingB == (size_t)K * N / lanesPerWavefront,
                        "B matrix size mismatch",
                        ShowValue(K),
                        ShowValue(N),
                        ShowValue(lanesPerWavefront),
                        ShowValue(K * N / lanesPerWavefront),
                        ShowValue(matB->valueCount()),
                        ShowValue(packingB));
            AssertFatal(matC->valueCount() == (size_t)M * N / lanesPerWavefront,
                        "C matrix size mismatch",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * N / lanesPerWavefront),
                        ShowValue(matC->valueCount()));
            AssertFatal(dest->valueCount() == (size_t)M * N / lanesPerWavefront,
                        "D matrix size mismatch",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * N / lanesPerWavefront),
                        ShowValue(dest->valueCount()));
            AssertFatal(isValidInputType(matA->variableType()),
                        "Invalid matrix A data type",
                        ShowValue(matA->variableType()));
            AssertFatal(matA->regType() == Register::Type::Vector,
                        "Invalid matrix A register type",
                        ShowValue(matA->regType()));
            AssertFatal(isValidInputType(matB->variableType()),
                        "Invalid matrix B data type",
                        ShowValue(matB->variableType()));
            AssertFatal(matB->regType() == Register::Type::Vector,
                        "Invalid matrix B register type",
                        ShowValue(matB->regType()));
            AssertFatal(isValidOutputType(matC->variableType()),
                        "Invalid matrix C data type",
                        ShowValue(matC->variableType()));
            auto const& arch = m_context->targetArchitecture();
            if(arch.HasCapability(GPUCapability::HasAccCD))
            {
                AssertFatal(dest->regType() == Register::Type::Accumulator,
                            "Invalid matrix D register type",
                            ShowValue(dest->regType()));
            }
            else
            {
                AssertFatal(dest->regType() == Register::Type::Vector,
                            "Invalid matrix D register type",
                            ShowValue(dest->regType()));
            }
            AssertFatal(isValidOutputType(dest->variableType()),
                        "Invalid matrix D data type",
                        ShowValue(dest->variableType()));

            auto typeA = matA->variableType().dataType;
            auto typeB = matB->variableType().dataType;

            std::string mi;
            std::string modifiers;

            if(arch.HasCapability(GPUCapability::HasMFMA_scale_f8f6f4))
            {
                AssertFatal((M == 16 && N == 16 && K == 128) || (M == 32 && N == 32 && K == 64),
                            "Invalid wavetile {}x{}x{} for scaled MFMA instruction for {}.",
                            M,
                            N,
                            K,
                            arch.target().toString());

                if(maybeScaleBlockSize)
                {
                    auto scaleBlockSize = maybeScaleBlockSize.value();
                    AssertFatal(scaleBlockSize == 32,
                                fmt::format("Scale block size expected: 32, got: {}, on: {}",
                                            scaleBlockSize,
                                            arch.target().toString()));
                }

                mi = concatenate("v_mfma_scale_f32_", M, "x", N, "x", K, "_f8f6f4");

                modifiers += "cbsz:" + Arithmetic::getModifier(typeA);
                modifiers += " blgp:" + Arithmetic::getModifier(typeB);
            }
            else
            {
                Throw<FatalError>("Scaled Matrix Multiplication is not supported for",
                                  arch.target().toString());
            }

            co_yield_(Instruction(mi, {dest}, {matA, matB, matC, scaleA, scaleB}, {modifiers}, ""));
        }
    }
}
