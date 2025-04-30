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
    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::Convert>>
        GetGenerator<Expression::Convert>(Register::ValuePtr dst,
                                          Register::ValuePtr arg,
                                          Expression::Convert const&);
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

    class ConvertGenerator : public UnaryArithmeticGenerator<Expression::Convert>
    {
    public:
        ConvertGenerator(ContextPtr c)
            : UnaryArithmeticGenerator<Expression::Convert>(c)
        {
        }

        // Match function required by Component system for selecting the correct
        // generator.
        static bool Match(typename UnaryArithmeticGenerator<Expression::Convert>::Argument const&)
        {

            return true;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<typename UnaryArithmeticGenerator<Expression::Convert>::Base>
            Build(typename UnaryArithmeticGenerator<Expression::Convert>::Argument const& arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<ConvertGenerator>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr         dst,
                                        Register::ValuePtr         arg,
                                        Expression::Convert const& expr) override;

    private:
        Generator<Instruction> generateFloat(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateHalf(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateHalfx2(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateBFloat16(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateBFloat16x2(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateFP8x4(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateBF8x4(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateFP8(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateBF8(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateFP6x16(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateBF6x16(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateFP4x8(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateInt32(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateInt64(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateUInt32(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateUInt64(Register::ValuePtr dest, Register::ValuePtr arg);

        Generator<Instruction> generateDouble(Register::ValuePtr dest, Register::ValuePtr arg);

    public:
        static const std::string Name;
    };

    /**
     *  Below are for Stochastic Rounding (SR) conversion.
     *  SR conversion takes two args: the first arg is the value to be converted, and
     *  the second arg is a seed for stochastic rounding.
     */
    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::SRConvert<DataType::FP8>>>
        GetGenerator<Expression::SRConvert<DataType::FP8>>(
            Register::ValuePtr dst,
            Register::ValuePtr lhs,
            Register::ValuePtr rhs,
            Expression::SRConvert<DataType::FP8> const&);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::SRConvert<DataType::BF8>>>
        GetGenerator<Expression::SRConvert<DataType::BF8>>(
            Register::ValuePtr dst,
            Register::ValuePtr lhs,
            Register::ValuePtr rhs,
            Expression::SRConvert<DataType::BF8> const&);

    // Templated Generator class based on the return type.
    template <DataType DATATYPE>
    class SRConvertGenerator : public BinaryArithmeticGenerator<Expression::SRConvert<DATATYPE>>
    {
    public:
        SRConvertGenerator(ContextPtr c)
            : BinaryArithmeticGenerator<Expression::SRConvert<DATATYPE>>(c)
        {
        }

        // Match function required by Component system for selecting the correct
        // generator.
        static bool Match(
            typename BinaryArithmeticGenerator<Expression::SRConvert<DATATYPE>>::Argument const&)
        {

            return true;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<
            typename BinaryArithmeticGenerator<Expression::SRConvert<DATATYPE>>::Base>
            Build(
                typename BinaryArithmeticGenerator<Expression::SRConvert<DATATYPE>>::Argument const&
                    arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<SRConvertGenerator<DATATYPE>>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr dst,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        Expression::SRConvert<DATATYPE> const&) override;

        inline static const std::string Name;
    };

}
