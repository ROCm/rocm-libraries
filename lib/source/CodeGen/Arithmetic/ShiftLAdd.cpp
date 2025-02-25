#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/ShiftLAdd.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(ShiftLAddGenerator);

    template <>
    std::shared_ptr<TernaryArithmeticGenerator<Expression::ShiftLAdd>>
        GetGenerator<Expression::ShiftLAdd>(Register::ValuePtr dst,
                                            Register::ValuePtr lhs,
                                            Register::ValuePtr shiftAmount,
                                            Register::ValuePtr rhs,
                                            Expression::ShiftLAdd const&)
    {
        return Component::Get<TernaryArithmeticGenerator<Expression::ShiftLAdd>>(
            getContextFromValues(dst, lhs, rhs, shiftAmount),
            dst->regType(),
            dst->variableType().dataType);
    }

    Generator<Instruction> ShiftLAddGenerator::generate(Register::ValuePtr dest,
                                                        Register::ValuePtr lhs,
                                                        Register::ValuePtr shiftAmount,
                                                        Register::ValuePtr rhs,
                                                        Expression::ShiftLAdd const&)
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

            co_yield_(Instruction("v_lshl_add_u32", {dest}, {lhs, toShift, rhs}, {}, ""));
        }
        else
        {
            co_yield generateOp<Expression::ShiftL>(dest, lhs, shiftAmount);
            co_yield generateOp<Expression::Add>(dest, dest, rhs);
        }
    }
}
