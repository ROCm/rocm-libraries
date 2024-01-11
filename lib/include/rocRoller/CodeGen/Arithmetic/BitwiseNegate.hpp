#pragma once

#include "ArithmeticGenerator.hpp"

namespace rocRoller
{

    // GetGenerator function will return the Generator to use based on the provided arguments.
    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::BitwiseNegate>>
        GetGenerator<Expression::BitwiseNegate>(Register::ValuePtr dst, Register::ValuePtr arg);

    // Templated Generator class based on the return type.
    class BitwiseNegateGenerator : public UnaryArithmeticGenerator<Expression::BitwiseNegate>
    {
    public:
        BitwiseNegateGenerator(ContextPtr c)
            : UnaryArithmeticGenerator<Expression::BitwiseNegate>(c)
        {
        }

        // Match function required by Component system for selecting the correct
        // generator.
        static bool
            Match(typename UnaryArithmeticGenerator<Expression::BitwiseNegate>::Argument const& arg)
        {
            ContextPtr     ctx;
            Register::Type registerType;
            DataType       dataType;

            std::tie(ctx, registerType, dataType) = arg;

            return true;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<typename UnaryArithmeticGenerator<Expression::BitwiseNegate>::Base>
            Build(typename UnaryArithmeticGenerator<Expression::BitwiseNegate>::Argument const& arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<BitwiseNegateGenerator>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr dst, Register::ValuePtr arg);

        static const std::string Name;
    };
}
