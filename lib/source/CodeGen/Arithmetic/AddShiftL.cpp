#include <rocRoller/CodeGen/Arithmetic/AddShiftL.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(AddShiftLGenerator);

    template <>
    std::shared_ptr<TernaryArithmeticGenerator<Expression::AddShiftL>>
        GetGenerator<Expression::AddShiftL>(Register::ValuePtr dst,
                                            Register::ValuePtr lhs,
                                            Register::ValuePtr rhs,
                                            Register::ValuePtr shiftAmount)
    {
        return Component::Get<TernaryArithmeticGenerator<Expression::AddShiftL>>(
            getContextFromValues(dst, lhs, rhs, shiftAmount),
            dst->regType(),
            dst->variableType().dataType);
    }

    Generator<Instruction> AddShiftLGenerator::generate(Register::ValuePtr dest,
                                                        Register::ValuePtr lhs,
                                                        Register::ValuePtr rhs,
                                                        Register::ValuePtr shiftAmount)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);
        AssertFatal(shiftAmount != nullptr);

        if(dest->regType() == Register::Type::Vector
           && (dest->variableType() == DataType::UInt32 || dest->variableType() == DataType::Int32))
        {
            auto toShift = shiftAmount->regType() == Register::Type::Literal
                               ? shiftAmount
                               : shiftAmount->subset({0});

            co_yield_(Instruction("v_add_lshl_u32", {dest}, {lhs, rhs, toShift}, {}, ""));
        }
        else
        {
            co_yield generateOp<Expression::Add>(dest, lhs, rhs);
            co_yield generateOp<Expression::ShiftL>(dest, dest, shiftAmount);
        }
    }
}
