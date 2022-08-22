#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/ShiftL.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(ShiftLGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::ShiftL>> GetGenerator<Expression::ShiftL>(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::ShiftL>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> ShiftLGenerator::generate(std::shared_ptr<Register::Value> dest,
                                                     std::shared_ptr<Register::Value> value,
                                                     std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        auto dataType = promoteDataType(dest, value, value);

        auto toShift = shiftAmount->regType() == Register::Type::Literal ? shiftAmount
                                                                         : shiftAmount->subset({0});

        if(dest->regType() == Register::Type::Scalar)
        {
            switch(dataType)
            {
            case DataType::Int32:
            case DataType::UInt32:
                co_yield_(Instruction("s_lshl_b32", {dest}, {value, toShift}, {}, ""));
                break;
            case DataType::Int64:
            case DataType::UInt64:
                co_yield_(Instruction("s_lshl_b64", {dest}, {value, toShift}, {}, ""));
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
                co_yield_(Instruction("v_lshlrev_b32", {dest}, {toShift, value}, {}, ""));
                break;
            case DataType::Int64:
            case DataType::UInt64:
                co_yield_(
                    Instruction("v_lshlrev_b64", {dest}, {toShift->subset({0}), value}, {}, ""));
                break;
            default:
                Throw<FatalError>("Unsupported datatype for arithmetic shift left operation: ",
                                  ShowValue(dataType));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for arithmetic shift left operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
