// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitFieldExtract.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::BitFieldExtract>> GetGenerator(
        Register::ValuePtr dst, Register::ValuePtr arg, Expression::BitFieldExtract const&)
    {
        return Component::Get<UnaryArithmeticGenerator<Expression::BitFieldExtract>>(
            getContextFromValues(dst, arg), dst->regType(), dst->variableType().dataType);
    }
}