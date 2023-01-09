#pragma once

#include "ArithmeticGenerator.hpp"

namespace rocRoller
{

    // GetGenerator function will return the Generator to use based on the provided arguments.
    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::Negate>>
        GetGenerator<Expression::Negate>(Register::ValuePtr dst, Register::ValuePtr arg);

    // Templated Generator class based on the return type.
    class NegateGenerator : public UnaryArithmeticGenerator<Expression::Negate>
    {
    public:
        NegateGenerator(std::shared_ptr<Context> c)
            : UnaryArithmeticGenerator<Expression::Negate>(c)
        {
        }

        // Match function required by Component system for selecting the correct
        // generator.
        static bool
            Match(typename UnaryArithmeticGenerator<Expression::Negate>::Argument const& arg)
        {
            std::shared_ptr<Context> ctx;
            Register::Type           registerType;
            DataType                 dataType;

            std::tie(ctx, registerType, dataType) = arg;

            return true;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<typename UnaryArithmeticGenerator<Expression::Negate>::Base>
            Build(typename UnaryArithmeticGenerator<Expression::Negate>::Argument const& arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<NegateGenerator>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr dst, Register::ValuePtr arg);

        static const std::string Name;
    };
}
