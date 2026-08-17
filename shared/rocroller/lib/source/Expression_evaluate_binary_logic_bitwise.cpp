// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Expression.hpp>
#include <rocRoller/Expression_evaluate_detail.hpp>
#include <rocRoller/Expression_evaluate_detail_binary.hpp>

namespace rocRoller::Expression::EvaluateDetail
{
    SIMPLE_BINARY_OP(BitwiseAnd, &);
    SIMPLE_BINARY_OP(BitwiseOr, |);
    SIMPLE_BINARY_OP(BitwiseXor, ^);
    SIMPLE_BINARY_OP(LogicalAnd, &&);
    SIMPLE_BINARY_OP(LogicalOr, ||);

    INSTANTIATE_BINARY_OP(BitwiseAnd);
    INSTANTIATE_BINARY_OP(BitwiseOr);
    INSTANTIATE_BINARY_OP(BitwiseXor);
    INSTANTIATE_BINARY_OP(LogicalAnd);
    INSTANTIATE_BINARY_OP(LogicalOr);
}
