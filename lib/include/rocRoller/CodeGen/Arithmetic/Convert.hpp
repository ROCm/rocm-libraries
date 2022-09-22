#pragma once

#include "ArithmeticGenerator.hpp"

namespace rocRoller
{

    // GetGenerator function will return the Generator to use based on the provided arguments.
    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::Convert<DataType::Float>>>
        GetGenerator<Expression::Convert<DataType::Float>>(Register::ValuePtr dst,
                                                           Register::ValuePtr arg);

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::Convert<DataType::Half>>>
        GetGenerator<Expression::Convert<DataType::Half>>(Register::ValuePtr dst,
                                                          Register::ValuePtr arg);

    /**
     * @brief Generates instructions to convert register value to a new datatype.
     *
     * @param dataType The new datatype
     * @param dest The destination register
     * @param arg The value to be converted
     * @return Generator<Instruction>
     */
    Generator<Instruction>
        generateConvertOp(DataType dataType, Register::ValuePtr dest, Register::ValuePtr arg);

    // Templated Generator class based on the return type.
    template <DataType DATATYPE>
    class ConvertGenerator : public UnaryArithmeticGenerator<Expression::Convert<DATATYPE>>
    {
    public:
        ConvertGenerator<DATATYPE>(std::shared_ptr<Context> c)
            : UnaryArithmeticGenerator<Expression::Convert<DATATYPE>>(c)
        {
        }

        // Match function required by Component system for selecting the correct
        // generator.
        static bool Match(
            typename UnaryArithmeticGenerator<Expression::Convert<DATATYPE>>::Argument const& arg)
        {
            std::shared_ptr<Context> ctx;
            Register::Type           registerType;
            DataType                 dataType;

            std::tie(ctx, registerType, dataType) = arg;

            return true;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<
            typename UnaryArithmeticGenerator<Expression::Convert<DATATYPE>>::Base>
            Build(typename UnaryArithmeticGenerator<Expression::Convert<DATATYPE>>::Argument const&
                      arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<ConvertGenerator<DATATYPE>>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr dst, Register::ValuePtr arg);

        inline static const std::string Name;
        inline static const std::string Basename;
    };
}
