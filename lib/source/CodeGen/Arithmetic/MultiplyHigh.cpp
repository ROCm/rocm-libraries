#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/MultiplyHigh.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(MultiplyHighGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::MultiplyHigh>>
        GetGenerator<Expression::MultiplyHigh>(Register::ValuePtr dst,
                                               Register::ValuePtr lhs,
                                               Register::ValuePtr rhs)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::MultiplyHigh>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> MultiplyHighGenerator::generate(Register::ValuePtr dest,
                                                           Register::ValuePtr lhs,
                                                           Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto const& dataTypeInfo = DataTypeInfo::Get(dest->variableType());

        if(dest->regType() == Register::Type::Scalar)
        {
            if(dataTypeInfo.isSigned)
            {
                co_yield_(Instruction("s_mul_hi_i32", {dest}, {lhs, rhs}, {}, ""));
            }
            else
            {
                co_yield_(Instruction("s_mul_hi_u32", {dest}, {lhs, rhs}, {}, ""));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            if(dataTypeInfo.isSigned)
            {
                co_yield_(Instruction("v_mul_hi_i32", {dest}, {lhs, rhs}, {}, ""));
            }
            else
            {
                co_yield_(Instruction("v_mul_hi_u32", {dest}, {lhs, rhs}, {}, ""));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for multiply high operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
