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

#ifndef HIPTENSOR_REDUCTION_META_TRAITS_HPP
#define HIPTENSOR_REDUCTION_META_TRAITS_HPP

#include <device_reduce.hpp>
#include <hiptensor/internal/types.hpp>

#include "ck/tensor_operation/gpu/device/impl/device_reduce_multiblock.hpp"
#include "meta_traits.hpp"

namespace hiptensor
{
    // Meta traits for Scalar reduction
    template <typename InDataType,
              typename AccDataType,
              typename OutDataType,
              ck::index_t Rank,
              ck::index_t NumReduceDim,
              typename ReduceOperation,
              typename InElementwiseOp,
              typename AccElementwiseOp,
              bool PropagateNan,
              bool OutputIndex>
    struct MetaTraits<ck::tensor_operation::device::DeviceReduce<InDataType,
                                                                 AccDataType,
                                                                 OutDataType,
                                                                 Rank,
                                                                 NumReduceDim,
                                                                 ReduceOperation,
                                                                 InElementwiseOp,
                                                                 AccElementwiseOp,
                                                                 PropagateNan,
                                                                 OutputIndex>>
    {
        constexpr static ck::index_t TensorRank         = Rank;
        constexpr static ck::index_t TensorNumReduceDim = NumReduceDim;
        constexpr static bool        TensorPropagateNan = PropagateNan;
        constexpr static bool        TensorOutputIndex  = OutputIndex;

        /*
         * CK does not use hip_bfloat16, instead it use ushort(ck::bhalf_t) for cuda bhalf_t type.
         * What we want here is that we can use ck::bhalf_t with ck instances and use hip_bfloat16
         * with hiptensor classes.
         *
         * When creating a solution, ck::bhalf_t was passed in to create ck instance.
         * When registering the solution, MetaTraits will returen hip_bfloat16 to create key.
         */
        using TensorInDataType       = std::conditional_t<std::is_same_v<InDataType, ck::bhalf_t>,
                                                          hiptensor::bfloat16_t,
                                                          InDataType>;
        using TensorAccDataType      = std::conditional_t<std::is_same_v<AccDataType, ck::bhalf_t>,
                                                          hiptensor::bfloat16_t,
                                                          AccDataType>;
        using TensorOutDataType      = std::conditional_t<std::is_same_v<OutDataType, ck::bhalf_t>,
                                                          hiptensor::bfloat16_t,
                                                          OutDataType>;
        using TensorReduceOperation  = ReduceOperation;
        using TensorInElementwiseOp  = InElementwiseOp;
        using TensorAccElementwiseOp = AccElementwiseOp;
        static_assert((std::is_same_v<TensorReduceOperation, typename ck::reduce::Add>)
                          || (std::is_same_v<TensorReduceOperation, typename ck::reduce::Mul>)
                          || (std::is_same_v<TensorReduceOperation, typename ck::reduce::Min>)
                          || (std::is_same_v<TensorReduceOperation, typename ck::reduce::Max>),
                      "Reduction Operator is not supported.");
    };
} // namespace hiptensor

#endif // HIPTENSOR_REDUCTION_META_TRAITS_HPP
