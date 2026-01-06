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

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitFieldExtract.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitFieldExtract_detail.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitwiseAnd.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitwiseNegate.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitwiseOr.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitwiseXor.hpp>
#include <rocRoller/CodeGen/Arithmetic/Convert.hpp>
#include <rocRoller/CodeGen/Arithmetic/Equal.hpp>
#include <rocRoller/CodeGen/Arithmetic/GreaterThan.hpp>
#include <rocRoller/CodeGen/Arithmetic/LessThanEqual.hpp>
#include <rocRoller/CodeGen/Arithmetic/LogicalAnd.hpp>
#include <rocRoller/CodeGen/Arithmetic/LogicalShiftR.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::Add>>::registerImplementations()
    {
        registerComponent<AddGenerator<Register::Type::Scalar, DataType::Int32>>();
        registerComponent<AddGenerator<Register::Type::Vector, DataType::Int32>>();
        registerComponent<AddGenerator<Register::Type::M0, DataType::UInt32>>();
        registerComponent<AddGenerator<Register::Type::Scalar, DataType::UInt32>>();
        registerComponent<AddGenerator<Register::Type::Vector, DataType::UInt32>>();
        registerComponent<AddGenerator<Register::Type::Scalar, DataType::Int64>>();
        registerComponent<AddGenerator<Register::Type::Vector, DataType::Int64>>();
        registerComponent<AddGenerator<Register::Type::Vector, DataType::Half>>();
        registerComponent<AddGenerator<Register::Type::Vector, DataType::Halfx2>>();
        registerComponent<AddGenerator<Register::Type::Vector, DataType::BFloat16>>();
        registerComponent<AddGenerator<Register::Type::Vector, DataType::Float>>();
        registerComponent<AddGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        TernaryArithmeticGenerator<Expression::AddShiftL>>::registerImplementations()
    {
        registerComponent<AddShiftLGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::ArithmeticShiftR>>::registerImplementations()
    {
        registerComponent<ArithmeticShiftRGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        UnaryArithmeticGenerator<Expression::BitFieldExtract>>::registerImplementations()
    {
        registerComponent<BitFieldExtractGenerator<DataType::Half>>();
        registerComponent<BitFieldExtractGenerator<DataType::BFloat16>>();
        registerComponent<BitFieldExtractGenerator<DataType::FP8>>();
        registerComponent<BitFieldExtractGenerator<DataType::BF8>>();
        registerComponent<BitFieldExtractGenerator<DataType::FP6>>();
        registerComponent<BitFieldExtractGenerator<DataType::BF6>>();
        registerComponent<BitFieldExtractGenerator<DataType::FP4>>();
        registerComponent<BitFieldExtractGenerator<DataType::Int8>>();
        registerComponent<BitFieldExtractGenerator<DataType::Int16>>();
        registerComponent<BitFieldExtractGenerator<DataType::Int32>>();
        registerComponent<BitFieldExtractGenerator<DataType::Int64>>();
        registerComponent<BitFieldExtractGenerator<DataType::Raw32>>();
        registerComponent<BitFieldExtractGenerator<DataType::UInt8>>();
        registerComponent<BitFieldExtractGenerator<DataType::UInt16>>();
        registerComponent<BitFieldExtractGenerator<DataType::UInt32>>();
        registerComponent<BitFieldExtractGenerator<DataType::UInt64>>();
        registerComponent<BitFieldExtractGenerator<DataType::E8M0>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::BitwiseAnd>>::registerImplementations()
    {
        registerComponent<BitwiseAndGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        UnaryArithmeticGenerator<Expression::BitwiseNegate>>::registerImplementations()
    {
        registerComponent<BitwiseNegateGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::BitwiseOr>>::registerImplementations()
    {
        registerComponent<BitwiseOrGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::BitwiseXor>>::registerImplementations()
    {
        registerComponent<BitwiseXorGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        TernaryArithmeticGenerator<Expression::Conditional>>::registerImplementations()
    {
        registerComponent<ConditionalGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        UnaryArithmeticGenerator<Expression::Convert>>::registerImplementations()
    {
        registerComponent<ConvertGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::Divide>>::registerImplementations()
    {
        registerComponent<DivideGenerator<Register::Type::Scalar, DataType::Int32>>();
        registerComponent<DivideGenerator<Register::Type::Vector, DataType::Int32>>();
        registerComponent<DivideGenerator<Register::Type::Scalar, DataType::Int64>>();
        registerComponent<DivideGenerator<Register::Type::Vector, DataType::Int64>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::Equal>>::registerImplementations()
    {
        registerComponent<EqualGenerator<Register::Type::Scalar, DataType::Int32>>();
        registerComponent<EqualGenerator<Register::Type::Vector, DataType::Int32>>();
        registerComponent<EqualGenerator<Register::Type::Scalar, DataType::Int64>>();
        registerComponent<EqualGenerator<Register::Type::Vector, DataType::Int64>>();
        registerComponent<EqualGenerator<Register::Type::Vector, DataType::Float>>();
        registerComponent<EqualGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        UnaryArithmeticGenerator<Expression::Exponential2>>::registerImplementations()
    {
        registerComponent<Exponential2Generator<Register::Type::Vector, DataType::Float>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::GreaterThan>>::registerImplementations()
    {
        registerComponent<GreaterThanGenerator<Register::Type::Scalar, DataType::Int32>>();
        registerComponent<GreaterThanGenerator<Register::Type::Scalar, DataType::UInt32>>();
        registerComponent<GreaterThanGenerator<Register::Type::Vector, DataType::Int32>>();
        registerComponent<GreaterThanGenerator<Register::Type::Vector, DataType::UInt32>>();
        registerComponent<GreaterThanGenerator<Register::Type::Scalar, DataType::Int64>>();
        registerComponent<GreaterThanGenerator<Register::Type::Scalar, DataType::UInt64>>();
        registerComponent<GreaterThanGenerator<Register::Type::Vector, DataType::Int64>>();
        registerComponent<GreaterThanGenerator<Register::Type::Vector, DataType::UInt64>>();
        registerComponent<GreaterThanGenerator<Register::Type::Vector, DataType::Float>>();
        registerComponent<GreaterThanGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::GreaterThanEqual>>::registerImplementations()
    {
        registerComponent<GreaterThanEqualGenerator<Register::Type::Scalar, DataType::Int32>>();
        registerComponent<GreaterThanEqualGenerator<Register::Type::Scalar, DataType::UInt32>>();
        registerComponent<GreaterThanEqualGenerator<Register::Type::Vector, DataType::Int32>>();
        registerComponent<GreaterThanEqualGenerator<Register::Type::Vector, DataType::UInt32>>();
        registerComponent<GreaterThanEqualGenerator<Register::Type::Scalar, DataType::Int64>>();
        registerComponent<GreaterThanEqualGenerator<Register::Type::Scalar, DataType::UInt64>>();
        registerComponent<GreaterThanEqualGenerator<Register::Type::Vector, DataType::Int64>>();
        registerComponent<GreaterThanEqualGenerator<Register::Type::Vector, DataType::UInt64>>();
        registerComponent<GreaterThanEqualGenerator<Register::Type::Vector, DataType::Float>>();
        registerComponent<GreaterThanEqualGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::LessThan>>::registerImplementations()
    {
        registerComponent<LessThanGenerator<Register::Type::Scalar, DataType::Int32>>();
        registerComponent<LessThanGenerator<Register::Type::Scalar, DataType::UInt32>>();
        registerComponent<LessThanGenerator<Register::Type::Vector, DataType::Int32>>();
        registerComponent<LessThanGenerator<Register::Type::Vector, DataType::UInt32>>();
        registerComponent<LessThanGenerator<Register::Type::Scalar, DataType::Int64>>();
        registerComponent<LessThanGenerator<Register::Type::Scalar, DataType::UInt64>>();
        registerComponent<LessThanGenerator<Register::Type::Vector, DataType::Int64>>();
        registerComponent<LessThanGenerator<Register::Type::Vector, DataType::UInt64>>();
        registerComponent<LessThanGenerator<Register::Type::Vector, DataType::Float>>();
        registerComponent<LessThanGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::LessThanEqual>>::registerImplementations()
    {
        registerComponent<LessThanEqualGenerator<Register::Type::Scalar, DataType::Int32>>();
        registerComponent<LessThanEqualGenerator<Register::Type::Scalar, DataType::UInt32>>();
        registerComponent<LessThanEqualGenerator<Register::Type::Vector, DataType::Int32>>();
        registerComponent<LessThanEqualGenerator<Register::Type::Vector, DataType::UInt32>>();
        registerComponent<LessThanEqualGenerator<Register::Type::Scalar, DataType::Int64>>();
        registerComponent<LessThanEqualGenerator<Register::Type::Scalar, DataType::UInt64>>();
        registerComponent<LessThanEqualGenerator<Register::Type::Vector, DataType::Int64>>();
        registerComponent<LessThanEqualGenerator<Register::Type::Vector, DataType::UInt64>>();
        registerComponent<LessThanEqualGenerator<Register::Type::Vector, DataType::Float>>();
        registerComponent<LessThanEqualGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::LogicalAnd>>::registerImplementations()
    {
        registerComponent<LogicalAndGenerator<Register::Type::Scalar, DataType::Bool32>>();
        registerComponent<LogicalAndGenerator<Register::Type::Scalar, DataType::Bool64>>();
    }

    template <>
    void Component::ComponentFactory<
        UnaryArithmeticGenerator<Expression::LogicalNot>>::registerImplementations()
    {
        registerComponent<LogicalNotGenerator<Register::Type::Scalar, DataType::Bool>>();
        registerComponent<LogicalNotGenerator<Register::Type::Scalar, DataType::Bool32>>();
        registerComponent<LogicalNotGenerator<Register::Type::Scalar, DataType::Bool64>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::LogicalOr>>::registerImplementations()
    {
        registerComponent<LogicalOrGenerator<Register::Type::Scalar, DataType::Bool32>>();
        registerComponent<LogicalOrGenerator<Register::Type::Scalar, DataType::Bool64>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::LogicalShiftR>>::registerImplementations()
    {
        registerComponent<LogicalShiftRGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::Modulo>>::registerImplementations()
    {
        registerComponent<ModuloGenerator<Register::Type::Scalar, DataType::Int32>>();
        registerComponent<ModuloGenerator<Register::Type::Vector, DataType::Int32>>();
        registerComponent<ModuloGenerator<Register::Type::Scalar, DataType::Int64>>();
        registerComponent<ModuloGenerator<Register::Type::Vector, DataType::Int64>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::Multiply>>::registerImplementations()
    {
        registerComponent<MultiplyGenerator<Register::Type::Scalar, DataType::Int32>>();
        registerComponent<MultiplyGenerator<Register::Type::Vector, DataType::Int32>>();
        registerComponent<MultiplyGenerator<Register::Type::Scalar, DataType::Int64>>();
        registerComponent<MultiplyGenerator<Register::Type::Vector, DataType::Int64>>();
        registerComponent<MultiplyGenerator<Register::Type::Vector, DataType::Halfx2>>();
        registerComponent<MultiplyGenerator<Register::Type::Vector, DataType::Float>>();
        registerComponent<MultiplyGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        TernaryArithmeticGenerator<Expression::MultiplyAdd>>::registerImplementations()
    {
        registerComponent<MultiplyAddGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::MultiplyHigh>>::registerImplementations()
    {
        registerComponent<MultiplyHighGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        UnaryArithmeticGenerator<Expression::Negate>>::registerImplementations()
    {
        registerComponent<NegateGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::NotEqual>>::registerImplementations()
    {
        registerComponent<NotEqualGenerator<Register::Type::Scalar, DataType::Int32>>();
        registerComponent<NotEqualGenerator<Register::Type::Vector, DataType::Int32>>();
        registerComponent<NotEqualGenerator<Register::Type::Scalar, DataType::Int64>>();
        registerComponent<NotEqualGenerator<Register::Type::Vector, DataType::Int64>>();
        registerComponent<NotEqualGenerator<Register::Type::Vector, DataType::Float>>();
        registerComponent<NotEqualGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        UnaryArithmeticGenerator<Expression::RandomNumber>>::registerImplementations()
    {
        registerComponent<RandomNumberGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::ShiftL>>::registerImplementations()
    {
        registerComponent<ShiftLGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        TernaryArithmeticGenerator<Expression::ShiftLAdd>>::registerImplementations()
    {
        registerComponent<ShiftLAddGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::Subtract>>::registerImplementations()
    {
        registerComponent<SubtractGenerator<Register::Type::Scalar, DataType::Int32>>();
        registerComponent<SubtractGenerator<Register::Type::Vector, DataType::Int32>>();
        registerComponent<SubtractGenerator<Register::Type::Scalar, DataType::UInt32>>();
        registerComponent<SubtractGenerator<Register::Type::Vector, DataType::UInt32>>();
        registerComponent<SubtractGenerator<Register::Type::Scalar, DataType::Int64>>();
        registerComponent<SubtractGenerator<Register::Type::Vector, DataType::Int64>>();
        registerComponent<SubtractGenerator<Register::Type::Vector, DataType::Float>>();
        registerComponent<SubtractGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::SRConvert<DataType::FP8>>>::registerImplementations()
    {
        registerComponent<SRConvertGenerator<DataType::FP8>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::SRConvert<DataType::BF8>>>::registerImplementations()
    {
        registerComponent<SRConvertGenerator<DataType::BF8>>();
    }
}
