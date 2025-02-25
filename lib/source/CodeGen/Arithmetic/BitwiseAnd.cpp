#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitwiseAnd.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(BitwiseAndGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::BitwiseAnd>>
        GetGenerator<Expression::BitwiseAnd>(Register::ValuePtr dst,
                                             Register::ValuePtr lhs,
                                             Register::ValuePtr rhs,
                                             Expression::BitwiseAnd const&)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::BitwiseAnd>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> BitwiseAndGenerator::generate(Register::ValuePtr dest,
                                                         Register::ValuePtr lhs,
                                                         Register::ValuePtr rhs,
                                                         Expression::BitwiseAnd const&)
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
                co_yield_(Instruction("s_and_b32", {dest}, {lhs, rhs}, {}, ""));
            }
            else if(elementBits == 64u)
            {
                co_yield_(Instruction("s_and_b64", {dest}, {lhs, rhs}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported elementBits for bitwiseAnd operation:: ",
                                  ShowValue(elementBits));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            co_yield swapIfRHSLiteral(lhs, rhs);

            if(elementBits <= 32u)
            {
                co_yield_(Instruction("v_and_b32", {dest}, {lhs, rhs}, {}, ""));
            }
            else if(elementBits == 64u)
            {
                co_yield_(Instruction("v_and_b32",
                                      {dest->subset({0})},
                                      {lhs->subset({0}), rhs->subset({0})},
                                      {},
                                      ""));
                co_yield_(Instruction("v_and_b32",
                                      {dest->subset({1})},
                                      {lhs->subset({1}), rhs->subset({1})},
                                      {},
                                      ""));
            }
            else
            {
                Throw<FatalError>("Unsupported elementBits for bitwiseAnd operation:: ",
                                  ShowValue(elementBits));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for bitwiseAnd operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
