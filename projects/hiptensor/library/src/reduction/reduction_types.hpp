/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp>
#include <ck/utility/reduction_enums.hpp>
#include <hiptensor/hiptensor_types.h>

namespace hiptensor
{

    template <hiptensorOperator_t opReduce>
    struct convert_to_ck_reduce_operator;

    template <hiptensorOperator_t opReduce>
    using convert_to_ck_reduce_operator_v = typename convert_to_ck_reduce_operator<opReduce>::value;

    template <>
    struct convert_to_ck_reduce_operator<HIPTENSOR_OP_ADD>
    {
        static constexpr auto value = ck::ReduceTensorOp::ADD;
    };

    template <>
    struct convert_to_ck_reduce_operator<HIPTENSOR_OP_MUL>
    {
        static constexpr auto value = ck::ReduceTensorOp::MUL;
    };
    template <>
    struct convert_to_ck_reduce_operator<HIPTENSOR_OP_MIN>
    {
        static constexpr auto value = ck::ReduceTensorOp::MIN;
    };
    template <>
    struct convert_to_ck_reduce_operator<HIPTENSOR_OP_MAX>
    {
        static constexpr auto value = ck::ReduceTensorOp::MAX;
    };
} // namespace hiptensor
