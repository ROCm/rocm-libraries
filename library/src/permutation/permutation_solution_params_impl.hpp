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

#ifndef HIPTENSOR_PERMUTATION_SOLUTION_PARAMS_IMPL_HPP
#define HIPTENSOR_PERMUTATION_SOLUTION_PARAMS_IMPL_HPP

#include "data_types.hpp"
#include "hash.hpp"
#include "permutation_meta_traits.hpp"
#include "permutation_solution_params.hpp"

namespace std
{
    template <>
    struct hash<hiptensor::PermutationSolutionParams>
    {
        size_t operator()(hiptensor::PermutationSolutionParams const& s,
                          size_t                                      mPerThread) const noexcept
        {
            return hiptensor::Hash{}(s.dim(), s.typeIn(), s.typeOut(), s.opScale(), mPerThread);
        }
    };
}

namespace hiptensor
{
    template <typename DeviceOp>
    struct PermutationSolutionParamsImpl : public PermutationSolutionParams
    {
        PermutationSolutionParamsImpl()                                                = default;
        ~PermutationSolutionParamsImpl()                                               = default;
        PermutationSolutionParamsImpl(PermutationSolutionParamsImpl const&)            = default;
        PermutationSolutionParamsImpl(PermutationSolutionParamsImpl&&)                 = default;
        PermutationSolutionParamsImpl& operator=(PermutationSolutionParamsImpl const&) = default;
        PermutationSolutionParamsImpl& operator=(PermutationSolutionParamsImpl&&)      = default;

        using MetaTraitsT = MetaTraits<DeviceOp>;

        int32_t dim() const override
        {
            return MetaTraitsT::NDim;
        }

        hipDataType typeIn() const override
        {
            return HipDataType_v<typename ck::tuple_element_t<0, typename MetaTraitsT::InDataT>>;
        }

        hipDataType typeOut() const override
        {
            return HipDataType_v<typename ck::tuple_element_t<0, typename MetaTraitsT::OutDataT>>;
        }

        PermutationOpId_t opScale() const override
        {
            return PermutationOperatorType_v<typename MetaTraitsT::ScaleOp>;
        }
    };

} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_SOLUTION_PARAMS_IMPL_HPP
