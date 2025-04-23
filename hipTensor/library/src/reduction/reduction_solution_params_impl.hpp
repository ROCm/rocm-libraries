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

#ifndef HIPTENSOR_REDUCTION_SOLUTION_PARAMS_IMPL_HPP
#define HIPTENSOR_REDUCTION_SOLUTION_PARAMS_IMPL_HPP

#include "data_types.hpp"
#include "hash.hpp"
#include "reduction_meta_traits.hpp"

namespace std
{
    template <>
    struct hash<hiptensor::ReductionSolutionParams>
    {
        size_t operator()(hiptensor::ReductionSolutionParams const& s) const noexcept
        {
            return hiptensor::Hash{}(s.typeIn(),
                                     s.typeAcc(),
                                     s.typeOut(),
                                     s.rankIn(),
                                     s.numReducedDim(),
                                     s.opReduce(),
                                     s.propagateNan(),
                                     s.outputIndex());
        }
    };
}

namespace hiptensor
{
    template <typename DeviceOp>
    struct ReductionSolutionParamsImpl : public ReductionSolutionParams
    {
        ReductionSolutionParamsImpl()                                              = default;
        ~ReductionSolutionParamsImpl()                                             = default;
        ReductionSolutionParamsImpl(ReductionSolutionParamsImpl const&)            = default;
        ReductionSolutionParamsImpl(ReductionSolutionParamsImpl&&)                 = default;
        ReductionSolutionParamsImpl& operator=(ReductionSolutionParamsImpl const&) = default;
        ReductionSolutionParamsImpl& operator=(ReductionSolutionParamsImpl&&)      = default;

        using MetaTraitsT = MetaTraits<DeviceOp>;

        // Map tensor dimension
        int32_t rankIn() const override
        {
            return MetaTraitsT::TensorRank;
        }

        int32_t numReducedDim() const override
        {
            return MetaTraitsT::TensorNumReduceDim;
        }

        bool propagateNan() const override
        {
            return MetaTraitsT::TensorPropagateNan;
        }

        bool outputIndex() const override
        {
            return MetaTraitsT::TensorOutputIndex;
        }

        hipDataType typeIn() const override
        {
            return HipDataType_v<typename MetaTraitsT::TensorInDataType>;
        }
        hiptensorComputeType_t typeAcc() const override
        {
            return convertToComputeType(HipDataType_v<typename MetaTraitsT::TensorAccDataType>);
        }
        hipDataType typeOut() const override
        {
            return HipDataType_v<typename MetaTraitsT::TensorOutDataType>;
        }

        hiptensorOperator_t opReduce() const override
        {
            if constexpr(std::is_same_v<typename MetaTraitsT::TensorReduceOperation,
                                        typename ck::reduce::Add>)
            {
                return HIPTENSOR_OP_ADD;
            }
            else if constexpr(std::is_same_v<typename MetaTraitsT::TensorReduceOperation,
                                             typename ck::reduce::Mul>)
            {
                return HIPTENSOR_OP_MUL;
            }
            else if constexpr(std::is_same_v<typename MetaTraitsT::TensorReduceOperation,
                                             typename ck::reduce::Min>)
            {
                return HIPTENSOR_OP_MIN;
            }
            else
            {
                // MetaTraitsT::TensorReduceOperation has only 4 options : ADD, MUL, MIN, MAX
                return HIPTENSOR_OP_MAX;
            }
        }
    };

} // namespace hiptensor

#endif // HIPTENSOR_REDUCTION_SOLUTION_PARAMS_IMPL_HPP
