/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2025 AMD ROCm(TM) Software
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

#include <rocRoller/Expression.hpp>
#include <rocRoller/Expression_evaluate_detail.hpp>
#include <rocRoller/Expression_evaluate_detail_binary.hpp>

namespace rocRoller::Expression::EvaluateDetail
{
    SIMPLE_BINARY_OP(ShiftL, <<);
    SIMPLE_BINARY_OP(ArithmeticShiftR, >>);

    template <>
    struct OperationEvaluatorVisitor<LogicalShiftR> : public BinaryEvaluatorVisitor<LogicalShiftR>
    {
        template <CIntegral T, CIntegral U>
        constexpr T evaluate(T const& lhs, U const& rhs) const
        {
            return static_cast<typename std::make_unsigned<T>::type>(lhs) >> rhs;
        }
    };

    template <DataType T_DataType>
    struct OperationEvaluatorVisitor<SRConvert<T_DataType>>
        : public BinaryEvaluatorVisitor<SRConvert<T_DataType>>
    {
        using Base       = BinaryEvaluatorVisitor<SRConvert<T_DataType>>;
        using ResultType = typename EnumTypeInfo<T_DataType>::Type;

        template <CArithmeticType T>
        requires CCanStaticCastTo<ResultType, T> ResultType evaluate(T const& arg)
        const
        {
            return static_cast<ResultType>(arg);
        }
    };

    INSTANTIATE_BINARY_OP(ShiftL);
    INSTANTIATE_BINARY_OP(LogicalShiftR);
    INSTANTIATE_BINARY_OP(ArithmeticShiftR);
    INSTANTIATE_BINARY_OP(SRConvert<DataType::FP8>);
    INSTANTIATE_BINARY_OP(SRConvert<DataType::BF8>);
}
