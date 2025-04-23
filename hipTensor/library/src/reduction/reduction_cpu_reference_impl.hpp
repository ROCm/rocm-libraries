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

#ifndef HIPTENSOR_REDUCTION_CPU_REFERENCE_IMPL_HPP
#define HIPTENSOR_REDUCTION_CPU_REFERENCE_IMPL_HPP

// Std includes
#include <array>
#include <list>
#include <numeric>
#include <vector>

// CK includes
#include "ck/library/reference_tensor_operation/cpu/reference_reduce.hpp"

#include "reduction_meta_traits.hpp"
#include "reduction_solution.hpp"

namespace hiptensor
{

    template <typename InDataType,
              typename AccDataType,
              typename OutDataType,
              ck::index_t Rank,
              ck::index_t NumReduceDim,
              typename ReduceOperation,
              typename InElementwiseOperation,
              typename AccElementwiseOperation,
              bool PropagateNan,
              bool OutputIndex>
    using ReferenceReduction = ck::tensor_operation::host::ReferenceReduce<InDataType,
                                                                           AccDataType,
                                                                           OutDataType,
                                                                           Rank,
                                                                           NumReduceDim,
                                                                           ReduceOperation,
                                                                           InElementwiseOperation,
                                                                           AccElementwiseOperation,
                                                                           PropagateNan,
                                                                           OutputIndex>;

    // Partial specialize for reference reduction
    template <typename InDataType,
              typename AccDataType,
              typename OutDataType,
              ck::index_t Rank,
              ck::index_t NumReduceDim,
              typename ReduceOperation,
              typename InElementwiseOperation,
              typename AccElementwiseOperation,
              bool PropagateNan,
              bool OutputIndex>
    struct MetaTraits<ReferenceReduction<InDataType,
                                         AccDataType,
                                         OutDataType,
                                         Rank,
                                         NumReduceDim,
                                         ReduceOperation,
                                         InElementwiseOperation,
                                         AccElementwiseOperation,
                                         PropagateNan,
                                         OutputIndex>>
        : public MetaTraits<ck::tensor_operation::device::DeviceReduce<InDataType,
                                                                       AccDataType,
                                                                       OutDataType,
                                                                       Rank,
                                                                       NumReduceDim,
                                                                       ReduceOperation,
                                                                       InElementwiseOperation,
                                                                       AccElementwiseOperation,
                                                                       PropagateNan,
                                                                       OutputIndex>>
    {
    };

    template <typename InDataType,
              typename AccDataType,
              typename OutDataType,
              int                 Rank,
              int                 NumReduceDim,
              hiptensorOperator_t opReduce,
              bool                PropagateNan,
              bool                OutputIndex>
    auto enumerateReferenceSolutions()
    {
        constexpr auto ReduceOpId = convertHiptensorReduceOperatorToCk<opReduce>();

        using ReduceOperation = typename ck::reduce_binary_operator<ReduceOpId>::opType;
        using InElementwiseOperation =
            typename ck::reduce_unary_operator<ReduceOpId, true, true>::InElementwiseOperation;
        using AccElementwiseOperation =
            typename ck::reduce_unary_operator<ReduceOpId, true, true>::AccElementwiseOperation;
        using ReferenceOp = ReferenceReduction<InDataType,
                                               AccDataType,
                                               OutDataType,
                                               Rank,
                                               NumReduceDim,
                                               ReduceOperation,
                                               InElementwiseOperation,
                                               AccElementwiseOperation,
                                               PropagateNan,
                                               OutputIndex>;

        auto solution
            = std::make_unique<ReductionSolutionImpl<ReferenceOp>>(std::make_unique<ReferenceOp>());

        auto result = std::vector<std::unique_ptr<ReductionSolution>>();
        result.push_back(std::move(solution));

        return result;
    }

} // namespace hiptensor

#endif // HIPTENSOR_REDUCTION_CPU_REFERENCE_IMPL_HPP
