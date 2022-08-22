#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/ShiftR.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(ShiftRGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::ShiftR>> GetGenerator<Expression::ShiftR>(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::ShiftR>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> ShiftRGenerator::generate(std::shared_ptr<Register::Value> dest,
                                                     std::shared_ptr<Register::Value> value,
                                                     std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        auto dataType = promoteDataType(dest, value, value);

        if(dest->regType() == Register::Type::Scalar)
        {
            switch(dataType)
            {
            case DataType::Int32:
            case DataType::UInt32:
                co_yield_(Instruction("s_lshr_b32", {dest}, {value, shiftAmount}, {}, ""));
                break;
            case DataType::Int64:
            case DataType::UInt64:
                co_yield_(
                    Instruction("s_ashr_i64", {dest}, {value, shiftAmount->subset({0})}, {}, ""));
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
                co_yield_(Instruction("v_lshrrev_b32", {dest}, {shiftAmount, value}, {}, ""));
                break;
            case DataType::Int64:
            case DataType::UInt64:
                co_yield_(Instruction(
                    "v_ashrrev_i64", {dest}, {shiftAmount->subset({0}), value}, {}, ""));
                break;
            default:
                Throw<FatalError>("Unsupported datatype for arithmetic shift right operation: ",
                                  ShowValue(dataType));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for arithmetic shift right operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
