// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
