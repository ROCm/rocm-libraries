#pragma once

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>

namespace rocRoller
{
    // GetGenerator function will return the Generator to use based on the provided arguments.
    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::Exponential2>>
        GetGenerator<Expression::Exponential2>(Register::ValuePtr dst,
                                               Register::ValuePtr arg,
                                               Expression::Exponential2 const&);

    // Templated Generator class based on the return type.
    template <Register::Type REGISTER_TYPE, DataType DATATYPE>
    class Exponential2Generator : public UnaryArithmeticGenerator<Expression::Exponential2>
    {
    public:
        Exponential2Generator(ContextPtr c)
            : UnaryArithmeticGenerator<Expression::Exponential2>(c)
        {
        }

        // Match function required by Component system for selecting the correct
        // generator.
        static bool
            Match(typename UnaryArithmeticGenerator<Expression::Exponential2>::Argument const& arg)
        {
            ContextPtr     ctx;
            Register::Type registerType;
            DataType       dataType;

            std::tie(ctx, registerType, dataType) = arg;

            return true;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<typename UnaryArithmeticGenerator<Expression::Exponential2>::Base>
            Build(typename UnaryArithmeticGenerator<Expression::Exponential2>::Argument const& arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<Exponential2Generator>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr dst,
                                        Register::ValuePtr arg,
                                        Expression::Exponential2 const&);

        static const std::string Name;
    };

    template <>
    Generator<Instruction> Exponential2Generator<Register::Type::Vector, DataType::Float>::generate(
        Register::ValuePtr dst, Register::ValuePtr arg, Expression::Exponential2 const&);

    template <>
    Generator<Instruction> Exponential2Generator<Register::Type::Vector, DataType::Half>::generate(
        Register::ValuePtr dst, Register::ValuePtr arg, Expression::Exponential2 const&);
}
