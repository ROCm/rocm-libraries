
#include <memory>

#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/InstructionValues/RegisterUtils.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    namespace InstructionGenerators
    {
        RegisterComponentTemplateSpec(MatrixMultiplyGenerator, DataType::Float, DataType::Float);
        RegisterComponentTemplateSpec(MatrixMultiplyGenerator, DataType::Float, DataType::Halfx2);

        const std::string MatrixMultiply::Name = "MatrixMultiply";

        template <DataType ACC, DataType INPUT>
        Generator<Instruction>
            MatrixMultiplyGenerator<ACC, INPUT>::zero(std::shared_ptr<Register::Value> dest)
        {
            co_yield Register::AllocateIfNeeded(dest);
            for(size_t i = 0; i < dest->valueCount(); ++i)
                co_yield_(Instruction("v_accvgpr_write",
                                      {dest->subset({i})},
                                      {Register::Value::Special("0x0")},
                                      {},
                                      "initialise to zero"));
        }

        template <DataType DATATYPE>
        std::string typeStr()
        {
            if constexpr(DATATYPE == DataType::Float)
                return "f32";
            else if constexpr(DATATYPE == DataType::Halfx2)
                return "f16";
            else
                return "unknown";
        }

        template <DataType ACC, DataType INPUT>
        Generator<Instruction>
            MatrixMultiplyGenerator<ACC, INPUT>::mul(std::shared_ptr<Register::Value> dest,
                                                     std::shared_ptr<Register::Value> lhs,
                                                     std::shared_ptr<Register::Value> rhs,
                                                     int                              M,
                                                     int                              N,
                                                     int                              K,
                                                     int                              B)
        {
            AssertFatal(lhs != nullptr);
            AssertFatal(rhs != nullptr);

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
                        ShowValue(lhs->valueCount()));
            AssertFatal(rhs->valueCount() * packing == (size_t)K * N * B / lanesPerWavefront,
                        "B matrix size mismatch",
                        ShowValue(K),
                        ShowValue(N),
                        ShowValue(B),
                        ShowValue(lanesPerWavefront),
                        ShowValue(K * N * B / lanesPerWavefront),
                        ShowValue(rhs->valueCount()));
            AssertFatal(dest->valueCount() == (size_t)M * N * B / lanesPerWavefront,
                        "D matrix size mismatch",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(B),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * N * B / lanesPerWavefront),
                        ShowValue(dest->valueCount()));
            AssertFatal(lhs->variableType() == INPUT,
                        "Invalid LHS data type",
                        ShowValue(lhs->variableType()));
            AssertFatal(lhs->regType() == Register::Type::Vector,
                        "Invalid LHS register type",
                        ShowValue(lhs->regType()));
            AssertFatal(rhs->variableType() == INPUT,
                        "Invalid RHS data type",
                        ShowValue(lhs->variableType()));
            AssertFatal(rhs->regType() == Register::Type::Vector,
                        "Invalid RHS register type",
                        ShowValue(rhs->regType()));
            AssertFatal(dest->variableType() == ACC,
                        "Invalid DEST data type",
                        ShowValue(lhs->variableType()));
            AssertFatal(dest->regType() == Register::Type::Accumulator,
                        "Invalid DEST register type",
                        ShowValue(lhs->regType()));

            auto mfma
                = concatenate("v_mfma_", typeStr<ACC>(), "_", M, "x", N, "x", K, typeStr<INPUT>());
            co_yield_(Instruction(mfma, {dest}, {lhs, rhs, dest}, {}, ""));
        }
    }
}
