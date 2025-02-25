#pragma once

#include "ArithmeticGenerator.hpp"

namespace rocRoller
{

    // GetGenerator function will return the Generator to use based on the provided arguments.
    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::BitwiseOr>>
        GetGenerator<Expression::BitwiseOr>(Register::ValuePtr dst,
                                            Register::ValuePtr lhs,
                                            Register::ValuePtr rhs,
                                            Expression::BitwiseOr const&);

    // Generator for all register types and datatypes.
    class BitwiseOrGenerator : public BinaryArithmeticGenerator<Expression::BitwiseOr>
    {
    public:
        BitwiseOrGenerator(ContextPtr c)
            : BinaryArithmeticGenerator<Expression::BitwiseOr>(c)
        {
        }

        // Match function required by Component system for selecting the correct
        // generator.
        static bool Match(Argument const& arg)
        {
            ContextPtr     ctx;
            Register::Type registerType;
            DataType       dataType;

            std::tie(ctx, registerType, dataType) = arg;

            return true;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<Base> Build(Argument const& arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<BitwiseOrGenerator>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr dest,
                                        Register::ValuePtr value,
                                        Register::ValuePtr shiftAmount,
                                        Expression::BitwiseOr const&);

        static const std::string Name;
    };
}
