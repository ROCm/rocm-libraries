// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>

namespace rocRoller
{
    template <>
    std::shared_ptr<TernaryArithmeticGenerator<Expression::MultiplyAdd>>
        GetGenerator<Expression::MultiplyAdd>(Register::ValuePtr dst,
                                              Register::ValuePtr a,
                                              Register::ValuePtr x,
                                              Register::ValuePtr y,
                                              Expression::MultiplyAdd const&);

    struct MultiplyAddGenerator : public TernaryArithmeticGenerator<Expression::MultiplyAdd>
    {
        MultiplyAddGenerator(ContextPtr c)
            : TernaryArithmeticGenerator<Expression::MultiplyAdd>(c)
        {
        }

        // Match function required by Component system for selecting the correct
        // generator.
        static bool Match(Argument const& arg)
        {
            return true;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<Base> Build(Argument const& arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<MultiplyAddGenerator>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr dest,
                                        Register::ValuePtr a,
                                        Register::ValuePtr x,
                                        Register::ValuePtr y,
                                        Expression::MultiplyAdd const&);

        inline static const std::string Name = "MultiplyAddGenerator";
    };
}
