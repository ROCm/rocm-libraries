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
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Add>>::registerComponent<
            AddGenerator<Register::Type::Scalar, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Add>>::registerComponent<
            AddGenerator<Register::Type::Vector, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Add>>::registerComponent<
            AddGenerator<Register::Type::M0, DataType::UInt32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Add>>::registerComponent<
            AddGenerator<Register::Type::Scalar, DataType::UInt32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Add>>::registerComponent<
            AddGenerator<Register::Type::Vector, DataType::UInt32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Add>>::registerComponent<
            AddGenerator<Register::Type::Scalar, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Add>>::registerComponent<
            AddGenerator<Register::Type::Vector, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Add>>::registerComponent<
            AddGenerator<Register::Type::Vector, DataType::Half>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Add>>::registerComponent<
            AddGenerator<Register::Type::Vector, DataType::Halfx2>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Add>>::registerComponent<
            AddGenerator<Register::Type::Vector, DataType::BFloat16>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Add>>::registerComponent<
            AddGenerator<Register::Type::Vector, DataType::Float>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Add>>::registerComponent<
            AddGenerator<Register::Type::Vector, DataType::Double>>();
        // auto addGenerator = [this]<int regTypeIdx = 0>()
        // {
        //     constexpr auto regType       = static_cast<Register::Type>(regTypeIdx);
        //     constexpr auto maxRegTypeIdx = static_cast<int>(Register::Type::Count);
        //     if constexpr(regTypeIdx < maxRegTypeIdx)
        //     {
        //         auto addGeneratorDataType = [this]<int dataTypeIdx = 0>()
        //         {
        //             constexpr auto dataType       = static_cast<DataType>(dataTypeIdx);
        //             constexpr auto maxDataTypeIdx = static_cast<int>(DataType::Count);
        //             if constexpr(dataTypeIdx < maxDataTypeIdx)
        //             {
        //                 Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Add>>::
        //                     registerComponent<AddGenerator<regType, dataType>>();
        //                 self<dataTypeIdx + 1>();
        //             }
        //         };
        //         self<regTypeIdx + 1>();
        //     }
        // };
        //
        // addGenerator();
    }

    template <>
    void Component::ComponentFactory<
        TernaryArithmeticGenerator<Expression::AddShiftL>>::registerImplementations()
    {
        Component::ComponentFactory<TernaryArithmeticGenerator<Expression::AddShiftL>>::
            registerComponent<AddShiftLGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::ArithmeticShiftR>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::ArithmeticShiftR>>::
            registerComponent<ArithmeticShiftRGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        UnaryArithmeticGenerator<Expression::BitFieldExtract>>::registerImplementations()
    {
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::Half>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::BFloat16>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::FP8>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::BF8>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::FP6>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::BF6>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::FP4>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::Int8>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::Int16>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::Int32>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::Int64>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::Raw32>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::UInt8>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::UInt16>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::UInt32>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::UInt64>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitFieldExtract>>::
            registerComponent<BitFieldExtractGenerator<DataType::E8M0>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::BitwiseAnd>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::BitwiseAnd>>::
            registerComponent<BitwiseAndGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        UnaryArithmeticGenerator<Expression::BitwiseNegate>>::registerImplementations()
    {
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::BitwiseNegate>>::
            registerComponent<BitwiseNegateGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::BitwiseOr>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::BitwiseOr>>::
            registerComponent<BitwiseOrGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::BitwiseXor>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::BitwiseXor>>::
            registerComponent<BitwiseXorGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        TernaryArithmeticGenerator<Expression::Conditional>>::registerImplementations()
    {
        Component::ComponentFactory<TernaryArithmeticGenerator<Expression::Conditional>>::
            registerComponent<ConditionalGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        UnaryArithmeticGenerator<Expression::Convert>>::registerImplementations()
    {
        Component::ComponentFactory<
            UnaryArithmeticGenerator<Expression::Convert>>::registerComponent<ConvertGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::Divide>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Divide>>::
            registerComponent<DivideGenerator<Register::Type::Scalar, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Divide>>::
            registerComponent<DivideGenerator<Register::Type::Vector, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Divide>>::
            registerComponent<DivideGenerator<Register::Type::Scalar, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Divide>>::
            registerComponent<DivideGenerator<Register::Type::Vector, DataType::Int64>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::Equal>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Equal>>::
            registerComponent<EqualGenerator<Register::Type::Scalar, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Equal>>::
            registerComponent<EqualGenerator<Register::Type::Vector, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Equal>>::
            registerComponent<EqualGenerator<Register::Type::Scalar, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Equal>>::
            registerComponent<EqualGenerator<Register::Type::Vector, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Equal>>::
            registerComponent<EqualGenerator<Register::Type::Vector, DataType::Float>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Equal>>::
            registerComponent<EqualGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        UnaryArithmeticGenerator<Expression::Exponential2>>::registerImplementations()
    {
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::Exponential2>>::
            registerComponent<Exponential2Generator<Register::Type::Vector, DataType::Float>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::GreaterThan>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThan>>::
            registerComponent<GreaterThanGenerator<Register::Type::Scalar, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThan>>::
            registerComponent<GreaterThanGenerator<Register::Type::Scalar, DataType::UInt32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThan>>::
            registerComponent<GreaterThanGenerator<Register::Type::Vector, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThan>>::
            registerComponent<GreaterThanGenerator<Register::Type::Vector, DataType::UInt32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThan>>::
            registerComponent<GreaterThanGenerator<Register::Type::Scalar, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThan>>::
            registerComponent<GreaterThanGenerator<Register::Type::Scalar, DataType::UInt64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThan>>::
            registerComponent<GreaterThanGenerator<Register::Type::Vector, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThan>>::
            registerComponent<GreaterThanGenerator<Register::Type::Vector, DataType::UInt64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThan>>::
            registerComponent<GreaterThanGenerator<Register::Type::Vector, DataType::Float>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThan>>::
            registerComponent<GreaterThanGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::GreaterThanEqual>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThanEqual>>::
            registerComponent<GreaterThanEqualGenerator<Register::Type::Scalar, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThanEqual>>::
            registerComponent<
                GreaterThanEqualGenerator<Register::Type::Scalar, DataType::UInt32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThanEqual>>::
            registerComponent<GreaterThanEqualGenerator<Register::Type::Vector, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThanEqual>>::
            registerComponent<
                GreaterThanEqualGenerator<Register::Type::Vector, DataType::UInt32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThanEqual>>::
            registerComponent<GreaterThanEqualGenerator<Register::Type::Scalar, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThanEqual>>::
            registerComponent<
                GreaterThanEqualGenerator<Register::Type::Scalar, DataType::UInt64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThanEqual>>::
            registerComponent<GreaterThanEqualGenerator<Register::Type::Vector, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThanEqual>>::
            registerComponent<
                GreaterThanEqualGenerator<Register::Type::Vector, DataType::UInt64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThanEqual>>::
            registerComponent<GreaterThanEqualGenerator<Register::Type::Vector, DataType::Float>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::GreaterThanEqual>>::
            registerComponent<
                GreaterThanEqualGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::LessThan>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThan>>::
            registerComponent<LessThanGenerator<Register::Type::Scalar, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThan>>::
            registerComponent<LessThanGenerator<Register::Type::Scalar, DataType::UInt32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThan>>::
            registerComponent<LessThanGenerator<Register::Type::Vector, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThan>>::
            registerComponent<LessThanGenerator<Register::Type::Vector, DataType::UInt32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThan>>::
            registerComponent<LessThanGenerator<Register::Type::Scalar, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThan>>::
            registerComponent<LessThanGenerator<Register::Type::Scalar, DataType::UInt64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThan>>::
            registerComponent<LessThanGenerator<Register::Type::Vector, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThan>>::
            registerComponent<LessThanGenerator<Register::Type::Vector, DataType::UInt64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThan>>::
            registerComponent<LessThanGenerator<Register::Type::Vector, DataType::Float>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThan>>::
            registerComponent<LessThanGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::LessThanEqual>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThanEqual>>::
            registerComponent<LessThanEqualGenerator<Register::Type::Scalar, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThanEqual>>::
            registerComponent<LessThanEqualGenerator<Register::Type::Scalar, DataType::UInt32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThanEqual>>::
            registerComponent<LessThanEqualGenerator<Register::Type::Vector, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThanEqual>>::
            registerComponent<LessThanEqualGenerator<Register::Type::Vector, DataType::UInt32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThanEqual>>::
            registerComponent<LessThanEqualGenerator<Register::Type::Scalar, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThanEqual>>::
            registerComponent<LessThanEqualGenerator<Register::Type::Scalar, DataType::UInt64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThanEqual>>::
            registerComponent<LessThanEqualGenerator<Register::Type::Vector, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThanEqual>>::
            registerComponent<LessThanEqualGenerator<Register::Type::Vector, DataType::UInt64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThanEqual>>::
            registerComponent<LessThanEqualGenerator<Register::Type::Vector, DataType::Float>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LessThanEqual>>::
            registerComponent<LessThanEqualGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::LogicalAnd>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LogicalAnd>>::
            registerComponent<LogicalAndGenerator<Register::Type::Scalar, DataType::Bool32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LogicalAnd>>::
            registerComponent<LogicalAndGenerator<Register::Type::Scalar, DataType::Bool64>>();
    }

    template <>
    void Component::ComponentFactory<
        UnaryArithmeticGenerator<Expression::LogicalNot>>::registerImplementations()
    {
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::LogicalNot>>::
            registerComponent<LogicalNotGenerator<Register::Type::Scalar, DataType::Bool>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::LogicalNot>>::
            registerComponent<LogicalNotGenerator<Register::Type::Scalar, DataType::Bool32>>();
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::LogicalNot>>::
            registerComponent<LogicalNotGenerator<Register::Type::Scalar, DataType::Bool64>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::LogicalOr>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LogicalOr>>::
            registerComponent<LogicalOrGenerator<Register::Type::Scalar, DataType::Bool32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LogicalOr>>::
            registerComponent<LogicalOrGenerator<Register::Type::Scalar, DataType::Bool64>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::LogicalShiftR>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::LogicalShiftR>>::
            registerComponent<LogicalShiftRGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::Modulo>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Modulo>>::
            registerComponent<ModuloGenerator<Register::Type::Scalar, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Modulo>>::
            registerComponent<ModuloGenerator<Register::Type::Vector, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Modulo>>::
            registerComponent<ModuloGenerator<Register::Type::Scalar, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Modulo>>::
            registerComponent<ModuloGenerator<Register::Type::Vector, DataType::Int64>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::Multiply>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Multiply>>::
            registerComponent<MultiplyGenerator<Register::Type::Scalar, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Multiply>>::
            registerComponent<MultiplyGenerator<Register::Type::Vector, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Multiply>>::
            registerComponent<MultiplyGenerator<Register::Type::Scalar, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Multiply>>::
            registerComponent<MultiplyGenerator<Register::Type::Vector, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Multiply>>::
            registerComponent<MultiplyGenerator<Register::Type::Vector, DataType::Halfx2>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Multiply>>::
            registerComponent<MultiplyGenerator<Register::Type::Vector, DataType::Float>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Multiply>>::
            registerComponent<MultiplyGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        TernaryArithmeticGenerator<Expression::MultiplyAdd>>::registerImplementations()
    {
        Component::ComponentFactory<TernaryArithmeticGenerator<Expression::MultiplyAdd>>::
            registerComponent<MultiplyAddGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::MultiplyHigh>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::MultiplyHigh>>::
            registerComponent<MultiplyHighGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        UnaryArithmeticGenerator<Expression::Negate>>::registerImplementations()
    {
        Component::ComponentFactory<
            UnaryArithmeticGenerator<Expression::Negate>>::registerComponent<NegateGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::NotEqual>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::NotEqual>>::
            registerComponent<NotEqualGenerator<Register::Type::Scalar, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::NotEqual>>::
            registerComponent<NotEqualGenerator<Register::Type::Vector, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::NotEqual>>::
            registerComponent<NotEqualGenerator<Register::Type::Scalar, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::NotEqual>>::
            registerComponent<NotEqualGenerator<Register::Type::Vector, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::NotEqual>>::
            registerComponent<NotEqualGenerator<Register::Type::Vector, DataType::Float>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::NotEqual>>::
            registerComponent<NotEqualGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        UnaryArithmeticGenerator<Expression::RandomNumber>>::registerImplementations()
    {
        Component::ComponentFactory<UnaryArithmeticGenerator<Expression::RandomNumber>>::
            registerComponent<RandomNumberGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::ShiftL>>::registerImplementations()
    {
        Component::ComponentFactory<
            BinaryArithmeticGenerator<Expression::ShiftL>>::registerComponent<ShiftLGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        TernaryArithmeticGenerator<Expression::ShiftLAdd>>::registerImplementations()
    {
        Component::ComponentFactory<TernaryArithmeticGenerator<Expression::ShiftLAdd>>::
            registerComponent<ShiftLAddGenerator>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::Subtract>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Subtract>>::
            registerComponent<SubtractGenerator<Register::Type::Scalar, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Subtract>>::
            registerComponent<SubtractGenerator<Register::Type::Vector, DataType::Int32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Subtract>>::
            registerComponent<SubtractGenerator<Register::Type::Scalar, DataType::UInt32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Subtract>>::
            registerComponent<SubtractGenerator<Register::Type::Vector, DataType::UInt32>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Subtract>>::
            registerComponent<SubtractGenerator<Register::Type::Scalar, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Subtract>>::
            registerComponent<SubtractGenerator<Register::Type::Vector, DataType::Int64>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Subtract>>::
            registerComponent<SubtractGenerator<Register::Type::Vector, DataType::Float>>();
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::Subtract>>::
            registerComponent<SubtractGenerator<Register::Type::Vector, DataType::Double>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::SRConvert<DataType::FP8>>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::SRConvert<
            DataType::FP8>>>::registerComponent<SRConvertGenerator<DataType::FP8>>();
    }

    template <>
    void Component::ComponentFactory<
        BinaryArithmeticGenerator<Expression::SRConvert<DataType::BF8>>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::SRConvert<
            DataType::BF8>>>::registerComponent<SRConvertGenerator<DataType::BF8>>();
    }
}
