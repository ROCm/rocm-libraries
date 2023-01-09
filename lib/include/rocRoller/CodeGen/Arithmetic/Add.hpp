#pragma once

#include "ArithmeticGenerator.hpp"

namespace rocRoller
{

    // GetGenerator function will return the Generator to use based on the provided arguments.
    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::Add>> GetGenerator<Expression::Add>(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);

    // Templated Generator class based on the register type and datatype.
    template <Register::Type REGISTER_TYPE, DataType DATATYPE>
    class AddGenerator : public BinaryArithmeticGenerator<Expression::Add>
    {
    public:
        AddGenerator<REGISTER_TYPE, DATATYPE>(std::shared_ptr<Context> c)
            : BinaryArithmeticGenerator<Expression::Add>(c)
        {
        }

        // Match function required by Component system for selecting the correct
        // generator.
        static bool Match(Argument const& arg)
        {
            std::shared_ptr<Context> ctx;
            Register::Type           registerType;
            DataType                 dataType;

            std::tie(ctx, registerType, dataType) = arg;

            // Int64 and UInt64 will generate the same instructions
            if constexpr(DATATYPE == DataType::Int64)
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

            return std::make_shared<AddGenerator<REGISTER_TYPE, DATATYPE>>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction>
            generate(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);

        inline static const std::string Name;
    };

    // Specializations for supported Register Type / DataType combinations
    template <>
    Generator<Instruction> AddGenerator<Register::Type::Scalar, DataType::Int32>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
    template <>
    Generator<Instruction> AddGenerator<Register::Type::Vector, DataType::Int32>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
    template <>
    Generator<Instruction> AddGenerator<Register::Type::Scalar, DataType::UInt32>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
    template <>
    Generator<Instruction> AddGenerator<Register::Type::Vector, DataType::UInt32>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
    template <>
    Generator<Instruction> AddGenerator<Register::Type::Scalar, DataType::Int64>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
    template <>
    Generator<Instruction> AddGenerator<Register::Type::Vector, DataType::Int64>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
    template <>
    Generator<Instruction> AddGenerator<Register::Type::Vector, DataType::Halfx2>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
    template <>
    Generator<Instruction> AddGenerator<Register::Type::Vector, DataType::Float>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
    template <>
    Generator<Instruction> AddGenerator<Register::Type::Vector, DataType::Double>::generate(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);
}
