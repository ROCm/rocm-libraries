
#include <memory>

#include <rocRoller/CodeGen/Arithmetic/ScaledMatrixMultiply.hpp>
#include <rocRoller/CodeGen/Arithmetic/Utility.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    namespace InstructionGenerators
    {
        RegisterComponent(ScaledMatrixMultiplyGenerator);

        const std::string ScaledMatrixMultiply::Basename = "ScaledMatrixMultiply";

        Generator<Instruction> ScaledMatrixMultiplyGenerator::mul(Register::ValuePtr dest,
                                                                  Register::ValuePtr matA,
                                                                  Register::ValuePtr matB,
                                                                  Register::ValuePtr matC,
                                                                  Register::ValuePtr scaleA,
                                                                  Register::ValuePtr scaleB,
                                                                  int                M,
                                                                  int                N,
                                                                  int                K)
        {
            AssertFatal(matA != nullptr);
            AssertFatal(matB != nullptr);
            AssertFatal(matC != nullptr);
            AssertFatal(scaleA != nullptr);
            AssertFatal(scaleB != nullptr);

            auto const lanesPerWavefront = m_context->targetArchitecture().GetCapability(
                GPUCapability::DefaultWavefrontSize);
            auto const packing = DataTypeInfo::Get(matA->variableType()).packing;
            AssertFatal(M > 0 && N > 0 && K > 0 && lanesPerWavefront > 0,
                        "Invalid inputs",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(K),
                        ShowValue(lanesPerWavefront));
            AssertFatal(matA->valueCount() * packing == (size_t)M * K / lanesPerWavefront,
                        "A matrix size mismatch",
                        ShowValue(M),
                        ShowValue(K),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * K / lanesPerWavefront),
                        ShowValue(matA->valueCount()),
                        ShowValue(packing));
            AssertFatal(matB->valueCount() * packing == (size_t)K * N / lanesPerWavefront,
                        "B matrix size mismatch",
                        ShowValue(K),
                        ShowValue(N),
                        ShowValue(lanesPerWavefront),
                        ShowValue(K * N / lanesPerWavefront),
                        ShowValue(matB->valueCount()),
                        ShowValue(packing));
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
            AssertFatal(dest->regType() == Register::Type::Accumulator,
                        "Invalid matrix D register type",
                        ShowValue(dest->regType()));
            AssertFatal(isValidOutputType(dest->variableType()),
                        "Invalid matrix D data type",
                        ShowValue(dest->variableType()));

            auto        typeA    = matA->variableType().dataType;
            auto        typeB    = matB->variableType().dataType;
            std::string cbsz     = "cbsz:" + Arithmetic::getModifier(typeA); // Matrix A type
            std::string blgp     = "blgp:" + Arithmetic::getModifier(typeB); // Matrix B type
            std::string abid     = "abid:1"; // Enable scale
            std::string modifier = concatenate(cbsz, " ", abid, " ", blgp);

            auto mfma = concatenate("v_mfma_scale_f32_", M, "x", N, "x", K, "_f8f6f4");

            // Currently compiler has a bug that the number of VGPRs must be 8 regardless
            // the data types of input matrices, and the following two lines are to workaround
            // this bug.
            // TODO: Remove these two lines when the compiler bug gets fixed.
            matA = matA->withFixedRegisters(8);
            matB = matB->withFixedRegisters(8);
            co_yield_(
                Instruction(mfma, {dest}, {matA, matB, matC, scaleA, scaleB}, {modifier}, ""));
        }
    }
}
