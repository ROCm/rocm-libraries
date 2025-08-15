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

#include "rocRoller/CodeGen/Arithmetic/BitFieldExtract.hpp"
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
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
        BinaryArithmeticGenerator<Expression::>>::registerImplementations()
    {
        Component::ComponentFactory<BinaryArithmeticGenerator<Expression::>>::registerComponent<>();
    }
}
