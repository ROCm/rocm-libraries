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

#include <array>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck
{
    namespace tensor_operation
    {
        namespace device
        {

            template <typename InDataType,
                      typename AccDataType,
                      typename OutDataType,
                      index_t Rank,
                      index_t NumReduceDim,
                      typename ReduceOperation,
                      typename InElementwiseOperation,
                      typename PriorDestElementwiseOperation,
                      typename AccElementwiseOperation,
                      bool PropagateNan,
                      bool OutputIndex>
            struct HipTensorDeviceReduce : public BaseOperator
            {
                static constexpr index_t NumOutDim
                    = (Rank - NumReduceDim == 0) ? 1 : Rank - NumReduceDim;

                virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
                    const std::array<index_t, Rank>      inLengths,
                    const std::array<index_t, Rank>      inStrides,
                    const std::array<index_t, NumOutDim> outLengths,
                    const std::array<index_t, NumOutDim> outStrides,
                    const std::array<int, NumReduceDim>  reduceDims,
                    double                               alpha,
                    double                               beta,
                    const void*                          in_dev,
                    const void*                          in_index_dev,
                    void*                                out_dev,
                    void*                                out_index_dev,
                    const InElementwiseOperation         in_elementwise_op,
                    const PriorDestElementwiseOperation  prior_dest_elementwise_op,
                    const AccElementwiseOperation        acc_elementwise_op)
                    = 0;

                virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
            };

            template <typename InDataType,
                      typename AccDataType,
                      typename OutDataType,
                      index_t Rank,
                      index_t NumReduceDim,
                      typename ReduceOperation,
                      typename InElementwiseOperation,
                      typename PriorDestElementwiseOperation,
                      typename AccElementwiseOperation,
                      bool PropagateNan,
                      bool OutputIndex>
            using HipTensorDeviceReducePtr
                = std::unique_ptr<HipTensorDeviceReduce<InDataType,
                                                        AccDataType,
                                                        OutDataType,
                                                        Rank,
                                                        NumReduceDim,
                                                        ReduceOperation,
                                                        InElementwiseOperation,
                                                        PriorDestElementwiseOperation,
                                                        AccElementwiseOperation,
                                                        PropagateNan,
                                                        OutputIndex>>;

        } // namespace device
    } // namespace tensor_operation
} // namespace ck
