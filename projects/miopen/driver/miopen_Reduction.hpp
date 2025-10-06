/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_MIOPEN_REDUCTION_HOST_HPP_
#define GUARD_MIOPEN_REDUCTION_HOST_HPP_

#include <functional>
#include <numeric>
#include <vector>
#include <type_traits>
#include <cassert>
#include <cmath>

#include "../test/cpu_reduce_util.hpp"

#include "tensor_driver.hpp"

using float16 = half_float::half;

template <typename Tgpu, typename Tref>
class miopenReductionHost
{
public:
    miopenReductionHost() = default;
    miopenReductionHost(const miopenReduceTensorDescriptor_t reduceDesc,
                        miopenTensorDescriptor_t inDesc,
                        miopenTensorDescriptor_t outDesc,
                        const std::vector<int>& invariantDims_,
                        const std::vector<int>& toReduceDims_)
    {
        miopenGetReduceTensorDescriptor(
            reduceDesc, &reduceOp, &compTypeVal, &nanOpt, &indicesOpt, &indicesType);

        this->inLengths  = GetTensorLengths(inDesc);
        this->outLengths = GetTensorLengths(outDesc);
        this->inStrides  = GetTensorStrides(inDesc);
        this->outStrides = GetTensorStrides(outDesc);

        this->invariantDims = invariantDims_;
        this->toReduceDims  = toReduceDims_;

        assert(this->inLengths.size() == this->outLengths.size());
        assert(!this->toReduceDims.empty());

        for(const auto dim : this->invariantDims)
            this->invariantLengths.push_back(this->inLengths[dim]);

        for(const auto dim : this->toReduceDims)
            toReduceLengths.push_back(this->inLengths[dim]);

        this->reduceAllDims = this->invariantDims.empty();
    };

    void
    Run(float alpha, const Tgpu* in_data, float beta, std::vector<Tref>& out_data, int* indices)
    ~miopenReductionHost(){};

    void Run(float alpha, const Tgpu* in_data, float beta, Tref* out_data, int* indices)
    {
        if(compTypeVal == miopenFloat)
        {
            if constexpr(std::is_same_v<Tref, double>)
                RunImpl<double>(alpha, in_data, beta, out_data, indices);
            else
                RunImpl<float>(alpha, in_data, beta, out_data, indices);
        }
        else if(compTypeVal == miopenHalf)
        {
            if constexpr(std::is_same_v<Tref, double> || std::is_same_v<Tref, float>)
                RunImpl<Tref>(alpha, in_data, beta, out_data, indices);
            else
                RunImpl<float16>(alpha, in_data, beta, out_data, indices);
        }
        else if(compTypeVal == miopenDouble)
            RunImpl<double>(alpha, in_data, beta, out_data, indices);
    };

private:
    miopenReduceTensorOp_t reduceOp;
    miopenDataType_t compTypeVal;
    miopenNanPropagation_t nanOpt;
    miopenReduceTensorIndices_t indicesOpt;
    miopenIndicesType_t indicesType;

    std::vector<int> inLengths;
    std::vector<int> outLengths;
    std::vector<int> inStrides;
    std::vector<int> outStrides;

    std::vector<int> invariantLengths;
    std::vector<int> toReduceLengths;

    std::vector<int> invariantDims;
    std::vector<int> toReduceDims;

    bool reduceAllDims;

    template <typename compType>
    void RunImpl(float alpha, const Tgpu* in_data, float beta, std::vector<Tref>& out_data, int* indices)
    {
        bool need_indices =
            (indicesOpt == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES) &&
            (reduceOp == MIOPEN_REDUCE_TENSOR_MIN || reduceOp == MIOPEN_REDUCE_TENSOR_MAX ||
             reduceOp == MIOPEN_REDUCE_TENSOR_AMAX);

        if(need_indices)
            RunImpl_generic<compType, true>(alpha, in_data, beta, out_data, indices);
        else
            RunImpl_generic<compType, false>(alpha, in_data, beta, out_data);
    };

    template <typename compType, bool UseIdx>
    void RunImpl_generic(float alpha,
                         const Tgpu* in_data,
                         float beta,
                         std::vector<Tref>& out_data,
                         [[maybe_unused]] int* indices = nullptr)
    {
        using reduce::binop_with_nan_check;
        using reduce::binop_with_nan_check2;
        using reduce::convert_type;
        using reduce::float_equal_one;
        using reduce::float_equal_zero;
        using reduce::PosUnaryOpFn;
        using reduce::PreUnaryOpFn;
        using reduce::ReduceOpFn;
        using reduce::ReduceOpFn2;
        using reduce::ReduceOpZeroVal;

        const auto divider = std::accumulate(
            toReduceLengths.begin(), toReduceLengths.end(), 1, std::multiplies<int>());

        auto PreUnaryOp = PreUnaryOpFn<compType>(reduceOp, divider);
        auto PosUnaryOp = PosUnaryOpFn<compType>(reduceOp, divider);

        // Select reducer
        [[maybe_unused]] auto opReduce_val = ReduceOpFn<compType>(this->reduceOp);
        [[maybe_unused]] auto opReduce_idx = ReduceOpFn2<compType>(this->reduceOp);

        if(reduceAllDims)
        {
            std::vector<std::vector<int>> idx_all;
            get_all_indexes(inLengths, 0, idx_all);

            compType accuVal = ReduceOpZeroVal<compType>(this->reduceOp);
            int accuIndex    = 0;

            for(const auto& src_index : idx_all)
            {
                const int src_offset = get_offset_from_index(this->inStrides, src_index);

                auto currVal = convert_type<compType>(in_data[src_offset]);
                PreUnaryOp(currVal);

                if constexpr(UseIdx)
                {
                    const int currIndex = get_flatten_offset(inLengths, src_index);
                    binop_with_nan_check2(
                        nanOpt, opReduce_idx, accuVal, currVal, accuIndex, currIndex);
                }
                else
                {
                    binop_with_nan_check(nanOpt, opReduce_val, accuVal, currVal);
                }
            }

            if constexpr(!UseIdx)
            {
                PosUnaryOp(accuVal);
            }

            if(!float_equal_one(alpha))
                accuVal *= convert_type<compType>(alpha);
            if(!float_equal_zero(beta))
                accuVal += convert_type<compType>(out_data[0]) * convert_type<compType>(beta);

            out_data[0] = convert_type<Tref>(accuVal);
            if constexpr(UseIdx)
            {
                indices[0] = accuIndex;
            }
        }
        else
        {
            std::vector<std::vector<int>> inv_idx, red_idx;
            get_all_indexes(this->invariantLengths, 0, inv_idx);
            get_all_indexes(this->toReduceLengths, 0, red_idx);

            for(const auto& i1 : inv_idx)
            {
                std::vector<int> src_index(this->inLengths.size(), 0);
                std::vector<int> dst_index(this->inLengths.size(), 0);

                for(int k = 0; k < invariantDims.size(); ++k)
                    dst_index[invariantDims[k]] = i1[k];

                const int dst_offset = get_offset_from_index(this->outStrides, dst_index);

                for(int k = 0; k < invariantDims.size(); ++k)
                    src_index[invariantDims[k]] = i1[k];

                compType accuVal = ReduceOpZeroVal<compType>(this->reduceOp);
                int accuIndex    = 0;

                for(const auto& i2 : red_idx)
                {
                    for(int k = 0; k < toReduceDims.size(); ++k)
                        src_index[toReduceDims[k]] = i2[k];

                    const int src_offset = get_offset_from_index(this->inStrides, src_index);

                    auto currVal = convert_type<compType>(in_data[src_offset]);
                    PreUnaryOp(currVal);

                    if constexpr(UseIdx)
                    {
                        const int currIndex = get_flatten_offset(toReduceLengths, i2);
                        binop_with_nan_check2(
                            nanOpt, opReduce_idx, accuVal, currVal, accuIndex, currIndex);
                    }
                    else
                    {
                        binop_with_nan_check(nanOpt, opReduce_val, accuVal, currVal);
                    }
                }

                if constexpr(!UseIdx)
                {
                    PosUnaryOp(accuVal);
                }

                if(!float_equal_one(alpha))
                    accuVal *= convert_type<compType>(alpha);
                if(!float_equal_zero(beta))
                    accuVal +=
                        convert_type<compType>(out_data[dst_offset]) * convert_type<compType>(beta);

                out_data[dst_offset] = convert_type<Tref>(accuVal);
                if constexpr(UseIdx)
                {
                    indices[dst_offset] = accuIndex;
                }
            }
        }
    }
};

#endif
