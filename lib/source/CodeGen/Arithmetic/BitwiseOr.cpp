#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitwiseOr.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(BitwiseOrGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::BitwiseOr>>
        GetGenerator<Expression::BitwiseOr>(Register::ValuePtr dst,
                                            Register::ValuePtr lhs,
                                            Register::ValuePtr rhs)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::BitwiseOr>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> BitwiseOrGenerator::generate(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto dataType = promoteDataType(dest, lhs, rhs);

        if(dest->regType() == Register::Type::Scalar)
        {
            switch(dataType)
            {
            case DataType::Int32:
            case DataType::UInt32:
            case DataType::Raw32:
                co_yield_(Instruction("s_or_b32", {dest}, {lhs, rhs}, {}, ""));
                break;
            case DataType::Int64:
            case DataType::UInt64:
                co_yield_(Instruction("s_or_b64", {dest}, {lhs, rhs}, {}, ""));
                break;
            default:
                Throw<FatalError>("Unsupported datatype for bitwise or operation: ",
                                  ShowValue(dataType));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            switch(dataType)
            {
            case DataType::Int32:
            case DataType::UInt32:
            case DataType::Raw32:
                co_yield_(Instruction("v_or_b32", {dest}, {lhs, rhs}, {}, ""));
                break;
            case DataType::Int64:
            case DataType::UInt64:
                co_yield_(Instruction(
                    "v_or_b32", {dest->subset({0})}, {lhs->subset({0}), rhs->subset({0})}, {}, ""));
                co_yield_(Instruction(
                    "v_or_b32", {dest->subset({1})}, {lhs->subset({1}), rhs->subset({1})}, {}, ""));
                break;
            default:
                Throw<FatalError>("Unsupported datatype for arithmetic shift right operation: ",
                                  ShowValue(dataType));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for bitwise operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
