
#include <memory>

#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/InstructionValues/RegisterUtils.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    namespace InstructionGenerators
    {
        RegisterComponent(MatrixMultiply_Float_Float);

        const std::string MatrixMultiply::Name = "MatrixMultiply";

        bool MatrixMultiply_Float_Float::Match(Argument const& arg)
        {
            auto atype = std::get<1>(arg);
            auto vtype = std::get<2>(arg);
            return atype == DataType::Float && vtype == DataType::Float;
        }

        std::shared_ptr<MatrixMultiply> MatrixMultiply_Float_Float::Build(Argument const& arg)
        {
            auto context = std::get<0>(arg);
            return std::make_shared<MatrixMultiply_Float_Float>(context);
        }

        Generator<Instruction>
            MatrixMultiply_Float_Float::zero(std::shared_ptr<Register::Value> dest)
        {
            co_yield Register::AllocateIfNeeded(dest);
            for(int i = 0; i < dest->valueCount(); ++i)
                co_yield_(Instruction("v_accvgpr_write",
                                      {dest->subset({i})},
                                      {Register::Value::Special("0x0")},
                                      {},
                                      "initialise to zero"));
        }

        Generator<Instruction>
            MatrixMultiply_Float_Float::mul(std::shared_ptr<Register::Value> dest,
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

            AssertFatal(lhs->valueCount() == M * K * B / lanesPerWavefront,
                        "A matrix size mismatch",
                        ShowValue(M),
                        ShowValue(K),
                        ShowValue(B),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * K * B / lanesPerWavefront),
                        ShowValue(lhs->valueCount()));
            AssertFatal(rhs->valueCount() == K * N * B / lanesPerWavefront,
                        "B matrix size mismatch",
                        ShowValue(K),
                        ShowValue(N),
                        ShowValue(B),
                        ShowValue(lanesPerWavefront),
                        ShowValue(K * N * B / lanesPerWavefront),
                        ShowValue(rhs->valueCount()));
            AssertFatal(dest->valueCount() == M * N * B / lanesPerWavefront,
                        "D matrix size mismatch",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(B),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * N * B / lanesPerWavefront),
                        ShowValue(dest->valueCount()));
            AssertFatal(lhs->variableType() == DataType::Float,
                        "Invalid LHS data type",
                        ShowValue(lhs->variableType()));
            AssertFatal(lhs->regType() == Register::Type::Vector,
                        "Invalid LHS register type",
                        ShowValue(lhs->regType()));
            AssertFatal(rhs->variableType() == DataType::Float,
                        "Invalid RHS data type",
                        ShowValue(lhs->variableType()));
            AssertFatal(rhs->regType() == Register::Type::Vector,
                        "Invalid RHS register type",
                        ShowValue(rhs->regType()));
            AssertFatal(dest->variableType() == DataType::Float,
                        "Invalid DEST data type",
                        ShowValue(lhs->variableType()));
            AssertFatal(dest->regType() == Register::Type::Accumulator,
                        "Invalid DEST register type",
                        ShowValue(lhs->regType()));

            auto mfma = concatenate("v_mfma_f32_", M, "x", N, "x", K, "f32");
            co_yield_(Instruction(mfma, {dest}, {lhs, rhs, dest}, {}, ""));
        }
    }
}
