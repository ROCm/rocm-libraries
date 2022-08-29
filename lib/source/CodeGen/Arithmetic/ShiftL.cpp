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

        auto toShift = shiftAmount->regType() == Register::Type::Literal ? shiftAmount
                                                                         : shiftAmount->subset({0});

        auto elementSize = std::max({DataTypeInfo::Get(dest->variableType()).elementSize,
                                     DataTypeInfo::Get(value->variableType()).elementSize});

        if(dest->regType() == Register::Type::Scalar)
        {
            if(elementSize <= 4)
            {
                co_yield_(Instruction("s_lshl_b32", {dest}, {value, toShift}, {}, ""));
            }
            else if(elementSize == 8)
            {
                co_yield_(Instruction("s_lshl_b64", {dest}, {value, toShift}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for shift left operation:: ",
                                  ShowValue(elementSize * 8));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            if(elementSize <= 4)
            {
                co_yield_(Instruction("v_lshlrev_b32", {dest}, {toShift, value}, {}, ""));
            }
            else if(elementSize == 8)
            {
                co_yield_(
                    Instruction("v_lshlrev_b64", {dest}, {toShift->subset({0}), value}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for shift left operation:: ",
                                  ShowValue(elementSize * 8));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for shift left operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
