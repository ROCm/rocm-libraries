
#include <memory>

#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    namespace InstructionGenerators
    {
        RegisterComponentTemplateSpec(MatrixMultiplyGenerator, DataType::Float, DataType::Float);
        RegisterComponentTemplateSpec(MatrixMultiplyGenerator, DataType::Float, DataType::Halfx2);
        RegisterComponentTemplateSpec(MatrixMultiplyGenerator, DataType::Float, DataType::FP8x4);
        RegisterComponentTemplateSpec(MatrixMultiplyGenerator, DataType::Float, DataType::BF8x4);

        const std::string MatrixMultiply::Basename = "MatrixMultiply";

        template <DataType DATATYPE>
        std::string typeStr()
        {
            if constexpr(DATATYPE == DataType::Float)
                return "f32";
            else if constexpr(DATATYPE == DataType::Halfx2)
                return "f16";
            else if constexpr(DATATYPE == DataType::FP8x4)
                return "_fp8_fp8";
            else if constexpr(DATATYPE == DataType::BF8x4)
                return "_bf8_bf8";
            else
                return "unknown";
        }

        template <DataType ACC, DataType INPUT>
        Generator<Instruction> MatrixMultiplyGenerator<ACC, INPUT>::mul(Register::ValuePtr dest,
                                                                        Register::ValuePtr lhs,
                                                                        Register::ValuePtr r1hs,
                                                                        Register::ValuePtr r2hs,
                                                                        int                M,
                                                                        int                N,
                                                                        int                K,
                                                                        int                B)
        {
            AssertFatal(lhs != nullptr);
            AssertFatal(r1hs != nullptr);
            AssertFatal(r2hs != nullptr);

            auto const lanesPerWavefront = m_context->targetArchitecture().GetCapability(
                GPUCapability::DefaultWavefrontSize);
            auto const packing = DataTypeInfo::Get(INPUT).packing;
            AssertFatal(M > 0 && N > 0 && K > 0 && B > 0 && lanesPerWavefront > 0,
                        "Invalid inputs",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(K),
                        ShowValue(B),
                        ShowValue(lanesPerWavefront));
            AssertFatal(lhs->valueCount() * packing == (size_t)M * K * B / lanesPerWavefront,
                        "A matrix size mismatch",
                        ShowValue(M),
                        ShowValue(K),
                        ShowValue(B),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * K * B / lanesPerWavefront),
                        ShowValue(lhs->valueCount()),
                        ShowValue(packing));
            AssertFatal(r1hs->valueCount() * packing == (size_t)K * N * B / lanesPerWavefront,
                        "B matrix size mismatch",
                        ShowValue(K),
                        ShowValue(N),
                        ShowValue(B),
                        ShowValue(lanesPerWavefront),
                        ShowValue(K * N * B / lanesPerWavefront),
                        ShowValue(r1hs->valueCount()),
                        ShowValue(packing));
            AssertFatal(r2hs->valueCount() == (size_t)M * N * B / lanesPerWavefront,
                        "C matrix size mismatch",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(B),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * N * B / lanesPerWavefront),
                        ShowValue(r2hs->valueCount()));
            AssertFatal(dest->valueCount() == (size_t)M * N * B / lanesPerWavefront,
                        "D matrix size mismatch",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(B),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * N * B / lanesPerWavefront),
                        ShowValue(dest->valueCount()));
            AssertFatal(lhs->variableType() == INPUT,
                        "Invalid LHS (A) data type",
                        ShowValue(lhs->variableType()));
            AssertFatal(lhs->regType() == Register::Type::Vector,
                        "Invalid LHS (A) register type",
                        ShowValue(lhs->regType()));
            AssertFatal(r1hs->variableType() == INPUT,
                        "Invalid R1HS (B) data type",
                        ShowValue(lhs->variableType()));
            AssertFatal(r1hs->regType() == Register::Type::Vector,
                        "Invalid R1HS (B) register type",
                        ShowValue(r1hs->regType()));
            AssertFatal(r2hs->variableType() == ACC,
                        "Invalid R2HS (C) data type",
                        ShowValue(r2hs->variableType()));
            AssertFatal(dest->regType() == Register::Type::Accumulator,
                        "Invalid DEST (D) register type",
                        ShowValue(lhs->regType()));
            AssertFatal(dest->variableType() == ACC,
                        "Invalid DEST (D) data type",
                        ShowValue(lhs->variableType()));

            std::string inputType;
            std::string modifier;
            if constexpr(INPUT == DataType::FP8x4)
            {
                if((M == 32 && N == 32 && K == 64) || (M == 16 && N == 16 && K == 128))
                {
                    inputType = "_f8f6f4";
                    // TODO: input matrix types of f8f6f4 can be fp8/bf8, fp6/bf6, fp4 or
                    //       mixed, where the types are indicated in cbsz and blgp. Here
                    //       only fp8 is handled and has to be enhanced to support rest types.
                    std::string cbsz = "cbsz:[0]"; // Matrix A type
                    std::string blgp = "blgp:[0]"; // Matrix B type
                    modifier         = concatenate(cbsz, " ", blgp);
                }
                else
                {
                    inputType = "_fp8_fp8";
                }
            }
            else
                inputType = typeStr<INPUT>();

            auto mfma = concatenate(
                "v_mfma_", typeStr<ACC>(), "_", M, "x", N, "x", K, std::move(inputType));
            co_yield_(Instruction(mfma, {dest}, {lhs, r1hs, r2hs}, {modifier}, ""));
        }
    }
}
