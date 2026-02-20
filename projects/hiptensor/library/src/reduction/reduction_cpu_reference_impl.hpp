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

// Std includes
#include <array>
#include <list>
#include <numeric>
#include <vector>

// CK includes
#include "device/hiptensor_reference_reduce.hpp"

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
              typename PriorDestElementwiseOperation,
              typename AccElementwiseOperation,
              bool PropagateNan,
              bool OutputIndex>
    using ReferenceReduction
        = ck::tensor_operation::host::ReferenceReduce<InDataType,
                                                      AccDataType,
                                                      OutDataType,
                                                      Rank,
                                                      NumReduceDim,
                                                      ReduceOperation,
                                                      InElementwiseOperation,
                                                      PriorDestElementwiseOperation,
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
              typename PriorDestElementwiseOperation,
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
                                         PriorDestElementwiseOperation,
                                         AccElementwiseOperation,
                                         PropagateNan,
                                         OutputIndex>>
        : public MetaTraits<
              ck::tensor_operation::device::HipTensorDeviceReduce<InDataType,
                                                                  AccDataType,
                                                                  OutDataType,
                                                                  Rank,
                                                                  NumReduceDim,
                                                                  ReduceOperation,
                                                                  InElementwiseOperation,
                                                                  PriorDestElementwiseOperation,
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
        constexpr auto ReduceOpId = convert_to_ck_reduce_operator<opReduce>::value;

        using ReduceOperation        = typename ck::reduce_binary_operator<ReduceOpId>::opType;
        using InElementwiseOperation = ck::tensor_operation::element_wise::HiptensorUnaryOp;
        using PriorDestElementwiseOperation = ck::tensor_operation::element_wise::HiptensorUnaryOp;
        using AccElementwiseOperation       = ck::tensor_operation::element_wise::PassThrough;
        using ReferenceOp                   = ReferenceReduction<InDataType,
                                               AccDataType,
                                               OutDataType,
                                               Rank,
                                               NumReduceDim,
                                               ReduceOperation,
                                               InElementwiseOperation,
                                               PriorDestElementwiseOperation,
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
