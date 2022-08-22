#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitwiseAnd.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(BitwiseAndGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::BitwiseAnd>>
        GetGenerator<Expression::BitwiseAnd>(Register::ValuePtr dst,
                                             Register::ValuePtr lhs,
                                             Register::ValuePtr rhs)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::BitwiseAnd>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> BitwiseAndGenerator::generate(std::shared_ptr<Register::Value> dest,
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
                co_yield_(Instruction("s_and_b32", {dest}, {lhs, rhs}, {}, ""));
                break;
            case DataType::Int64:
            case DataType::UInt64:
                co_yield_(Instruction("s_and_b64", {dest}, {lhs, rhs}, {}, ""));
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
                co_yield_(Instruction("v_and_b32", {dest}, {lhs, rhs}, {}, ""));
                break;
            case DataType::Int64:
            case DataType::UInt64:
                co_yield_(Instruction("v_and_b32",
                                      {dest->subset({0})},
                                      {lhs->subset({0}), rhs->subset({0})},
                                      {},
                                      ""));
                co_yield_(Instruction("v_and_b32",
                                      {dest->subset({1})},
                                      {lhs->subset({1}), rhs->subset({1})},
                                      {},
                                      ""));
                break;
            default:
                Throw<FatalError>("Unsupported datatype for bitwiseAnd operation: ",
                                  ShowValue(dataType));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for bitwiseAnd operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
