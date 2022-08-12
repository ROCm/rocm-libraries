#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/SignedShiftR.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(SignedShiftRGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::SignedShiftR>>
        GetGenerator<Expression::SignedShiftR>(Register::ValuePtr dst,
                                               Register::ValuePtr lhs,
                                               Register::ValuePtr rhs)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::SignedShiftR>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction>
        SignedShiftRGenerator::generate(std::shared_ptr<Register::Value> dest,
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
                co_yield_(Instruction("s_ashr_i32", {dest}, {value, toShift}, {}, ""));
                break;
            case DataType::Int64:
            case DataType::UInt64:
                co_yield_(Instruction("s_ashr_i64", {dest}, {value, toShift}, {}, ""));
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
                co_yield_(Instruction("v_ashrrev_i32", {dest}, {toShift, value}, {}, ""));
                break;
            case DataType::Int64:
            case DataType::UInt64:
                co_yield_(Instruction("v_ashrrev_i64", {dest}, {toShift, value}, {}, ""));
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
