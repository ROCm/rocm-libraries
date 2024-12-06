#pragma once

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>

namespace rocRoller
{

    // GetGenerator function will return the Generator to use based on the provided arguments.
    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::BitFieldExtract>> GetGenerator(
        Register::ValuePtr dst, Register::ValuePtr arg, Expression::BitFieldExtract const& expr);

    template <DataType DATATYPE>
    class BitFieldExtractGenerator : public UnaryArithmeticGenerator<Expression::BitFieldExtract>
    {
    public:
        BitFieldExtractGenerator(ContextPtr c)
            : UnaryArithmeticGenerator<Expression::BitFieldExtract>(c)
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

            return dataType == DATATYPE;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<Base> Build(Argument const& arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<BitFieldExtractGenerator>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr                 dst,
                                        Register::ValuePtr                 arg,
                                        Expression::BitFieldExtract const& expr);

        static const std::string Name;
    };
}
