/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_REDUCTION_TYPES_HPP
#define HIPTENSOR_REDUCTION_TYPES_HPP

#include <ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp>
#include <ck/utility/reduction_enums.hpp>
#include <hiptensor/hiptensor_types.hpp>

namespace hiptensor
{

    template <hiptensorOperator_t opReduce>
    constexpr inline auto convertHiptensorReduceOperatorToCk()
    {
        static_assert((opReduce == HIPTENSOR_OP_ADD) || (opReduce == HIPTENSOR_OP_MUL)
                          || (opReduce == HIPTENSOR_OP_MIN) || (opReduce == HIPTENSOR_OP_MAX),
                      "opReduce is not supported");

        constexpr auto reduceOpId = (opReduce == HIPTENSOR_OP_ADD)   ? ck::ReduceTensorOp::ADD
                                    : (opReduce == HIPTENSOR_OP_MUL) ? ck::ReduceTensorOp::MUL
                                    : (opReduce == HIPTENSOR_OP_MIN) ? ck::ReduceTensorOp::MIN
                                                                     : ck::ReduceTensorOp::MAX;
        return reduceOpId;
    }

    template <ck::ReduceTensorOp opReduce>
    constexpr inline auto convertCkReduceOperatorToHiptensor()
    {
        static_assert((opReduce == ck::ReduceTensorOp::ADD) || (opReduce == ck::ReduceTensorOp::MUL)
                          || (opReduce == ck::ReduceTensorOp::MIN)
                          || (opReduce == ck::ReduceTensorOp::MAX),
                      "opReduce is not supported");

        constexpr auto reduceOpId = (opReduce == ck::ReduceTensorOp::ADD)   ? HIPTENSOR_OP_ADD
                                    : (opReduce == ck::ReduceTensorOp::MUL) ? HIPTENSOR_OP_MUL
                                    : (opReduce == ck::ReduceTensorOp::MIN) ? HIPTENSOR_OP_MIN
                                                                            : HIPTENSOR_OP_MAX;
        return reduceOpId;
    }
    inline auto reductionUnaryOperators(hiptensorOperator_t opReduce, int32_t reduceLength)
    {
        if(opReduce == HIPTENSOR_OP_ADD)
        {
            return ck::reduce_unary_operator<ck::ReduceTensorOp::ADD, true, true>::
                GetElementwiseOperator(reduceLength);
        }
        else if(opReduce == HIPTENSOR_OP_MUL)
        {
            return ck::reduce_unary_operator<ck::ReduceTensorOp::MUL, true, true>::
                GetElementwiseOperator(reduceLength);
        }
        else if(opReduce == HIPTENSOR_OP_MIN)
        {
            return ck::reduce_unary_operator<ck::ReduceTensorOp::MIN, true, true>::
                GetElementwiseOperator(reduceLength);
        }
        else
        {
            return ck::reduce_unary_operator<ck::ReduceTensorOp::MAX, true, true>::
                GetElementwiseOperator(reduceLength);
        }
    }

} // namespace hiptensor
#endif // HIPTENSOR_REDUCTION_TYPES_HPP
