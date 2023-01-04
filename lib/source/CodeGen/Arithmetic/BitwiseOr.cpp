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

        auto elementSize = std::max({DataTypeInfo::Get(dest->variableType()).elementSize,
                                     DataTypeInfo::Get(lhs->variableType()).elementSize,
                                     DataTypeInfo::Get(rhs->variableType()).elementSize});

        if(dest->regType() == Register::Type::Scalar)
        {
            if(elementSize <= 4)
            {
                co_yield_(Instruction("s_or_b32", {dest}, {lhs, rhs}, {}, ""));
            }
            else if(elementSize == 8)
            {
                co_yield_(Instruction("s_or_b64", {dest}, {lhs, rhs}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for bitwiseOr operation:: ",
                                  ShowValue(elementSize * 8));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            co_yield swapIfRHSLiteral(lhs, rhs);

            if(elementSize <= 4)
            {
                co_yield_(Instruction("v_or_b32", {dest}, {lhs, rhs}, {}, ""));
            }
            else if(elementSize == 8)
            {
                co_yield_(Instruction(
                    "v_or_b32", {dest->subset({0})}, {lhs->subset({0}), rhs->subset({0})}, {}, ""));
                co_yield_(Instruction(
                    "v_or_b32", {dest->subset({1})}, {lhs->subset({1}), rhs->subset({1})}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for bitwiseOr operation:: ",
                                  ShowValue(elementSize * 8));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for bitwiseOr operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
