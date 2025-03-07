/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>

namespace rocRoller
{
    // GetGenerator function will return the Generator to use based on the provided arguments.
    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::LessThan>>
        GetGenerator<Expression::LessThan>(Register::ValuePtr dst,
                                           Register::ValuePtr lhs,
                                           Register::ValuePtr rhs,
                                           Expression::LessThan const&);

    // Templated Generator class based on the register type and datatype.
    template <Register::Type REGISTER_TYPE, DataType DATATYPE>
    class LessThanGenerator : public BinaryArithmeticGenerator<Expression::LessThan>
    {
    public:
        LessThanGenerator(ContextPtr c)
            : BinaryArithmeticGenerator<Expression::LessThan>(c)
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

            return registerType == REGISTER_TYPE && dataType == DATATYPE;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<Base> Build(Argument const& arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<LessThanGenerator<REGISTER_TYPE, DATATYPE>>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr dst,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        Expression::LessThan const&);

        static const std::string Name;
    };

    // Specializations for supported Register Type / DataType combinations
    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Scalar, DataType::Int32>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&);
    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Scalar, DataType::UInt32>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&);
    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Vector, DataType::Int32>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&);
    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Vector, DataType::UInt32>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&);
    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Scalar, DataType::Int64>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&);
    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Scalar, DataType::UInt64>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&);
    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Vector, DataType::Int64>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&);
    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Vector, DataType::UInt64>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&);
    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Vector, DataType::Float>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&);
    template <>
    Generator<Instruction> LessThanGenerator<Register::Type::Vector, DataType::Double>::generate(
        Register::ValuePtr dst,
        Register::ValuePtr lhs,
        Register::ValuePtr rhs,
        Expression::LessThan const&);
}
