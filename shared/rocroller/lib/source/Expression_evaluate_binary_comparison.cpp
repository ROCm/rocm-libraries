// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Expression.hpp>
#include <rocRoller/Expression_evaluate_detail.hpp>
#include <rocRoller/Expression_evaluate_detail_binary.hpp>

namespace rocRoller::Expression::EvaluateDetail
{
    SIMPLE_BINARY_OP(GreaterThan, >);
    SIMPLE_BINARY_OP(GreaterThanEqual, >=);
    SIMPLE_BINARY_OP(LessThan, <);
    SIMPLE_BINARY_OP(LessThanEqual, <=);
    SIMPLE_BINARY_OP(Equal, ==);
    SIMPLE_BINARY_OP(NotEqual, !=);

    INSTANTIATE_BINARY_OP(GreaterThan);
    INSTANTIATE_BINARY_OP(GreaterThanEqual);
    INSTANTIATE_BINARY_OP(LessThan);
    INSTANTIATE_BINARY_OP(LessThanEqual);
    INSTANTIATE_BINARY_OP(Equal);
    INSTANTIATE_BINARY_OP(NotEqual);
}
