#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitwiseXor.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(BitwiseXorGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::BitwiseXor>>
        GetGenerator<Expression::BitwiseXor>(Register::ValuePtr dst,
                                             Register::ValuePtr lhs,
                                             Register::ValuePtr rhs)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::BitwiseXor>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> BitwiseXorGenerator::generate(std::shared_ptr<Register::Value> dest,
                                                         std::shared_ptr<Register::Value> lhs,
                                                         std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto dataType = promoteDataType(dest, lhs, lhs);

        if(dest->regType() == Register::Type::Scalar)
        {
            switch(dataType)
            {
            case DataType::Int32:
            case DataType::UInt32:
                co_yield_(Instruction("s_xor_b32", {dest}, {lhs, rhs}, {}, ""));
                break;
            case DataType::Int64:
            case DataType::UInt64:
                co_yield_(Instruction("s_xor_b64", {dest}, {lhs, rhs}, {}, ""));
                break;
            default:
                Throw<FatalError>("Unsupported datatype for arithmetic shift right operation: ",
                                  ShowValue(dataType));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            switch(dataType)
            {
            case DataType::Int32:
            case DataType::UInt32:
                co_yield_(Instruction("v_xor_b32", {dest}, {lhs, rhs}, {}, ""));
                break;
            case DataType::Int64:
            case DataType::UInt64:
                co_yield_(Instruction("v_xor_b32",
                                      {dest->subset({0})},
                                      {lhs->subset({0}), rhs->subset({0})},
                                      {},
                                      ""));
                co_yield_(Instruction("v_xor_b32",
                                      {dest->subset({1})},
                                      {lhs->subset({1}), rhs->subset({1})},
                                      {},
                                      ""));
                break;
            default:
                Throw<FatalError>("Unsupported datatype for bitwiseXor operation: ",
                                  ShowValue(dataType));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for bitwiseXor operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
