// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitwiseAnd.hpp>
#include <rocRoller/CodeGen/Arithmetic/Utility.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
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

        auto destNumBits = DataTypeInfo::Get(dest->variableType()).elementBits;
        auto lhsNumBits  = DataTypeInfo::Get(lhs->variableType()).elementBits;
        auto rhsNumBits  = DataTypeInfo::Get(rhs->variableType()).elementBits;

        auto elementBits = std::max({destNumBits, lhsNumBits, rhsNumBits});

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
            co_yield m_context->copier()->ensureTypeCommutative(
                {Register::Type::Vector, Register::Type::Literal},
                lhs,
                {Register::Type::Vector},
                rhs);

            if(elementBits <= 32u)
            {
                co_yield_(Instruction("v_and_b32", {dest}, {lhs, rhs}, {}, ""));
            }
            else if(elementBits == 64u)
            {
                Register::ValuePtr l0, l1, r0, r1;
                if(lhs->regType() == Register::Type::Scalar)
                {
                    co_yield get2DwordsScalar(l0, l1, lhs);
                }
                else
                {
                    co_yield get2DwordsVector(l0, l1, lhs);
                }
                co_yield get2DwordsVector(r0, r1, rhs);

                co_yield_(Instruction("v_and_b32", {dest->subset({0})}, {l0, r0}, {}, ""));
                co_yield_(Instruction("v_and_b32", {dest->subset({1})}, {l1, r1}, {}, ""));
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
