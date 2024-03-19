#pragma once

#include "ArithmeticGenerator.hpp"

namespace rocRoller
{
    // GetGenerator function will return the Generator to use based on the provided arguments.
    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::LogicalAnd>>
        GetGenerator<Expression::LogicalAnd>(Register::ValuePtr dst,
                                             Register::ValuePtr lhs,
                                             Register::ValuePtr rhs);

    // Templated Generator class based on the register type and datatype.
    template <Register::Type REGISTER_TYPE, DataType DATATYPE>
    class LogicalAndGenerator : public BinaryArithmeticGenerator<Expression::LogicalAnd>
    {
    public:
        LogicalAndGenerator(ContextPtr c)
            : BinaryArithmeticGenerator<Expression::LogicalAnd>(c)
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

            if constexpr(DATATYPE == DataType::Bool32)
            {
                return registerType == REGISTER_TYPE
                       && (dataType == DataType::Bool || dataType == DataType::Bool32
                           || dataType == DataType::Raw32);
            }
            else
            {
                return registerType == REGISTER_TYPE && dataType == DATATYPE;
            }
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<Base> Build(Argument const& arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<LogicalAndGenerator<REGISTER_TYPE, DATATYPE>>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction>
            generate(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);

        static const std::string Name;
    };

    // Specializations for supported Register Type / DataType combinations
    template <>
    Generator<Instruction> LogicalAndGenerator<Register::Type::Scalar, DataType::Bool32>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
    template <>
    Generator<Instruction> LogicalAndGenerator<Register::Type::Scalar, DataType::UInt64>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
}
