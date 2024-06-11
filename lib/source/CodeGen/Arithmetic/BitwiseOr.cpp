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

    Generator<Instruction> BitwiseOrGenerator::generate(Register::ValuePtr dest,
                                                        Register::ValuePtr lhs,
                                                        Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto elementBits = std::max({DataTypeInfo::Get(dest->variableType()).elementBits,
                                     DataTypeInfo::Get(lhs->variableType()).elementBits,
                                     DataTypeInfo::Get(rhs->variableType()).elementBits});

        if(dest->regType() == Register::Type::Scalar)
        {
            if(elementBits <= 32u)
            {
                co_yield_(Instruction("s_or_b32", {dest}, {lhs, rhs}, {}, ""));
            }
            else if(elementBits == 64u)
            {
                co_yield_(Instruction("s_or_b64", {dest}, {lhs, rhs}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported elementBits for bitwiseOr operation:: ",
                                  ShowValue(elementBits));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            co_yield swapIfRHSLiteral(lhs, rhs);

            if(elementBits <= 32u)
            {
                co_yield_(Instruction("v_or_b32", {dest}, {lhs, rhs}, {}, ""));
            }
            else if(elementBits == 64u)
            {
                co_yield_(Instruction(
                    "v_or_b32", {dest->subset({0})}, {lhs->subset({0}), rhs->subset({0})}, {}, ""));
                co_yield_(Instruction(
                    "v_or_b32", {dest->subset({1})}, {lhs->subset({1}), rhs->subset({1})}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported elementBits for bitwiseOr operation:: ",
                                  ShowValue(elementBits));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for bitwiseOr operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
