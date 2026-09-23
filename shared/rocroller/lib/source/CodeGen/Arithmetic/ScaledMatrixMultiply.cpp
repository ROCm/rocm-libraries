// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
        const std::string ScaledMatrixMultiply::Basename = "ScaledMatrixMultiply";

        Generator<Instruction>
            ScaledMatrixMultiplyGenerator::mul(Register::ValuePtr  dest,
                                               Register::ValuePtr  matA,
                                               Register::ValuePtr  matB,
                                               Register::ValuePtr  matC,
                                               Register::ValuePtr  scaleA,
                                               Register::ValuePtr  scaleB,
                                               MatrixMultiplySizes miSizes,
                                               std::optional<uint> maybeScaleBlockSize)
        {
            AssertFatal(matA != nullptr);
            AssertFatal(matB != nullptr);
            AssertFatal(matC != nullptr);
            AssertFatal(scaleA != nullptr);
            AssertFatal(scaleB != nullptr);

            // Special case for VGPR indexing:
            //     v_wmma_ld_scale ignores the MODE register therefore
            //     the scale data must be allocated sub v256.
            AssertFatal(scaleA->getRegisterIds().begin()->regIndex < 256,
                        ShowValue(scaleA->getRegisterIds().begin()->regIndex));
            AssertFatal(scaleB->getRegisterIds().begin()->regIndex < 256,
                        ShowValue(scaleB->getRegisterIds().begin()->regIndex));

            auto const lanesPerWavefront = m_context->targetArchitecture().GetCapability(
                GPUCapability::DefaultWavefrontSize);
            AssertFatal(miSizes.m > 0 && miSizes.n > 0 && miSizes.k > 0 && lanesPerWavefront > 0,
                        "Invalid inputs",
                        ShowValue(miSizes),
                        ShowValue(lanesPerWavefront));
            auto const packingA = DataTypeInfo::Get(matA->variableType()).packing;
            AssertFatal(matA->valueCount() * packingA
                            == (size_t)miSizes.m * miSizes.k / lanesPerWavefront,
                        "A matrix size mismatch",
                        ShowValue(miSizes),
                        ShowValue(lanesPerWavefront),
                        ShowValue(miSizes.m * miSizes.k / lanesPerWavefront),
                        ShowValue(matA->valueCount()),
                        ShowValue(packingA));
            auto const packingB = DataTypeInfo::Get(matB->variableType()).packing;
            AssertFatal(matB->valueCount() * packingB
                            == (size_t)miSizes.k * miSizes.n / lanesPerWavefront,
                        "B matrix size mismatch",
                        ShowValue(miSizes),
                        ShowValue(lanesPerWavefront),
                        ShowValue(miSizes.k * miSizes.n / lanesPerWavefront),
                        ShowValue(matB->valueCount()),
                        ShowValue(packingB));
            AssertFatal(matC->valueCount() == (size_t)miSizes.m * miSizes.n / lanesPerWavefront,
                        "C matrix size mismatch",
                        ShowValue(miSizes),
                        ShowValue(lanesPerWavefront),
                        ShowValue(miSizes.m * miSizes.n / lanesPerWavefront),
                        ShowValue(matC->valueCount()));
            AssertFatal(dest->valueCount() == (size_t)miSizes.m * miSizes.n / lanesPerWavefront,
                        "D matrix size mismatch",
                        ShowValue(miSizes),
                        ShowValue(lanesPerWavefront),
                        ShowValue(miSizes.m * miSizes.n / lanesPerWavefront),
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

            if(arch.HasCapability(GPUCapability::HasMFMA_scale_f8f6f4))
            {
                std::string mi;
                std::string aType, bType;

                auto M = miSizes.m;
                auto N = miSizes.n;
                auto K = miSizes.k;

                AssertFatal(
                    (M == 16 && N == 16 && K == 128) || (M == 32 && N == 32 && K == 64),
                    fmt::format("Invalid wavetile {}x{}x{} for scaled MFMA instruction for {}.",
                                M,
                                N,
                                K,
                                arch.target().toString()));

                if(maybeScaleBlockSize)
                {
                    auto scaleBlockSize = maybeScaleBlockSize.value();
                    AssertFatal(scaleBlockSize == 32,
                                fmt::format("Scale block size expected: 32, got: {}, on: {}",
                                            scaleBlockSize,
                                            arch.target().toString()));
                }

                mi = concatenate("v_mfma_scale_f32_", M, "x", N, "x", K, "_f8f6f4");

                aType = "cbsz:" + Arithmetic::getModifier(typeA);
                bType = "blgp:" + Arithmetic::getModifier(typeB);

                AssertFatal(scaleA->getBitOffset() % 8 == 0 && scaleA->getBitOffset() < 32,
                            ShowValue(scaleA->getBitOffset()));

                AssertFatal(scaleB->getBitOffset() % 8 == 0 && scaleB->getBitOffset() < 32,
                            ShowValue(scaleB->getBitOffset()));

                auto aScaleByte = scaleA->getBitOffset() / 8;
                auto bScaleByte = scaleB->getBitOffset() / 8;

                auto [opselLo, opselHi]
                    = Arithmetic::getOpselModifiers2xByte(aScaleByte, bScaleByte);

                Instruction inst(mi,
                                 {dest},
                                 {matA, matB, matC, scaleA, scaleB},
                                 {opselLo, opselHi, aType, bType},
                                 "");

                co_yield inst;
            }
            else if(arch.HasCapability(GPUCapability::HasWMMA_scale_f8f6f4))
            {
                AssertFatal((miSizes.m == 16 && miSizes.n == 16 && miSizes.k == 128)
                                || (miSizes.m == 32 && miSizes.n == 16 && miSizes.k == 128),
                            "Invalid wavetile {}x{}x{} for scaled WMMA instruction for {}.",
                            miSizes.m,
                            miSizes.n,
                            miSizes.k,
                            arch.target().toString());

                std::string miSuffix = "_f8f6f4";

                if(miSizes.m == 32 && miSizes.n == 16 && miSizes.k == 128)
                {
                    AssertFatal(
                        isF4(typeA) && isF4(typeB),
                        fmt::format(
                            "Invalid types for A and B. Scaled {}x{}x{} WMMA instructions only "
                            "support FP4.",
                            miSizes.m,
                            miSizes.n,
                            miSizes.k),
                        ShowValue(typeA),
                        ShowValue(typeB));
                    miSuffix = "_f4";
                }

                std::string mi;

                if(maybeScaleBlockSize)
                {
                    auto scaleBlockSize = maybeScaleBlockSize.value();
                    AssertFatal(arch.isSupportedScaleBlockSize(scaleBlockSize),
                                fmt::format("Scale block size {} not supported on {}",
                                            scaleBlockSize,
                                            arch.target().toString()));

                    if(scaleBlockSize == 32)
                    {
                        mi = concatenate("v_wmma_scale_f32_",
                                         miSizes.m,
                                         "x",
                                         miSizes.n,
                                         "x",
                                         miSizes.k,
                                         miSuffix);
                    }
                    else if(scaleBlockSize == 16)
                    {
                        AssertFatal(arch.HasCapability(GPUCapability::HasWMMA_scale16_f8f6f4));
                        mi = concatenate("v_wmma_scale16_f32_",
                                         miSizes.m,
                                         "x",
                                         miSizes.n,
                                         "x",
                                         miSizes.k,
                                         miSuffix);
                    }
                    else
                    {
                        Throw<FatalError>(fmt::format(
                            "Scaled Matrix Multiplication with block size {} is unimplemented {}",
                            scaleBlockSize,
                            arch.target().toString()));
                    }
                }
                else
                {
                    mi = concatenate(
                        "v_wmma_scale_f32_", miSizes.m, "x", miSizes.n, "x", miSizes.k, miSuffix);
                }

                auto scaleAType = scaleA->variableType().dataType;
                auto scaleBType = scaleB->variableType().dataType;

                AssertFatal(
                    isValidDataTypeScaleTypeCombination(typeA, typeB, scaleAType, scaleBType),
                    ShowValue(typeA),
                    ShowValue(typeB),
                    ShowValue(scaleAType),
                    ShowValue(scaleBType));

                auto aFmt = "matrix_a_fmt:" + Arithmetic::getModifier(typeA);
                auto bFmt = "matrix_b_fmt:" + Arithmetic::getModifier(typeB);
                auto aScaleFmt
                    = "matrix_a_scale_fmt:" + Arithmetic::getScaleTypeModifier(scaleAType);
                auto bScaleFmt
                    = "matrix_b_scale_fmt:" + Arithmetic::getScaleTypeModifier(scaleBType);

                if(miSuffix == "_f4")
                {
                    // _f4 only supports FP4 input types, so no need to specify formats
                    aFmt = "";
                    bFmt = "";
                }

                Instruction inst(mi,
                                 {dest},
                                 {matA, matB, matC, scaleA, scaleB},
                                 {aFmt, bFmt, aScaleFmt, bScaleFmt},
                                 "");

                co_yield inst;
            }
            else
            {
                Throw<FatalError>("Scaled Matrix Multiplication is not supported for",
                                  arch.target().toString());
            }
        }
    }
}
