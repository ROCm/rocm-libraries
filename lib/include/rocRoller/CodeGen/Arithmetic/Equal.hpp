#pragma once

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>

namespace rocRoller
{
    // GetGenerator function will return the Generator to use based on the provided arguments.
    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::Equal>>
        GetGenerator<Expression::Equal>(Register::ValuePtr dst,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        Expression::Equal const&);

    // Templated Generator class based on the register type and datatype.
    template <Register::Type REGISTER_TYPE, DataType DATATYPE>
    class EqualGenerator : public BinaryArithmeticGenerator<Expression::Equal>
    {
    public:
        EqualGenerator(ContextPtr c)
            : BinaryArithmeticGenerator<Expression::Equal>(c)
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

            if constexpr(DATATYPE == DataType::Int32)
                return registerType == REGISTER_TYPE
                       && (dataType == DataType::Int32 || dataType == DataType::UInt32);

            else if constexpr(DATATYPE == DataType::Int64)
                return registerType == REGISTER_TYPE
                       && (dataType == DataType::Int64 || dataType == DataType::UInt64);
            else
                return registerType == REGISTER_TYPE && dataType == DATATYPE;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<Base> Build(Argument const& arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<EqualGenerator<REGISTER_TYPE, DATATYPE>>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr dst,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        Expression::Equal const&);

        static const std::string Name;
    };

    // Specializations for supported Register Type / DataType combinations
    template <>
    Generator<Instruction>
        EqualGenerator<Register::Type::Scalar, DataType::Int32>::generate(Register::ValuePtr dst,
                                                                          Register::ValuePtr lhs,
                                                                          Register::ValuePtr rhs,
                                                                          Expression::Equal const&);
    template <>
    Generator<Instruction>
        EqualGenerator<Register::Type::Vector, DataType::Int32>::generate(Register::ValuePtr dst,
                                                                          Register::ValuePtr lhs,
                                                                          Register::ValuePtr rhs,
                                                                          Expression::Equal const&);
    template <>
    Generator<Instruction>
        EqualGenerator<Register::Type::Scalar, DataType::Int64>::generate(Register::ValuePtr dst,
                                                                          Register::ValuePtr lhs,
                                                                          Register::ValuePtr rhs,
                                                                          Expression::Equal const&);
    template <>
    Generator<Instruction>
        EqualGenerator<Register::Type::Vector, DataType::Int64>::generate(Register::ValuePtr dst,
                                                                          Register::ValuePtr lhs,
                                                                          Register::ValuePtr rhs,
                                                                          Expression::Equal const&);
    template <>
    Generator<Instruction>
        EqualGenerator<Register::Type::Vector, DataType::Float>::generate(Register::ValuePtr dst,
                                                                          Register::ValuePtr lhs,
                                                                          Register::ValuePtr rhs,
                                                                          Expression::Equal const&);
    template <>
    Generator<Instruction> EqualGenerator<Register::Type::Vector, DataType::Double>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::Equal const&);
}
