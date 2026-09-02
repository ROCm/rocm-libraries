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
    miopenReductionHost(const miopenReduceTensorDescriptor_t reduceDesc_,
                        miopenTensorDescriptor_t inDesc,
                        miopenTensorDescriptor_t outDesc,
                        const std::vector<int>& invariantDims_,
                        const std::vector<int>& toReduceDims_)
    {
        miopenGetReduceTensorDescriptor(
            reduceDesc_, &reduceOp, &compTypeVal, &nanOpt, &indicesOpt, &indicesType);

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

    void Run(float alpha,
             std::vector<Tgpu>& in_data,
             float beta,
             std::vector<Tref>& out_data,
             bool parallel,
             std::vector<int>& indices)
    {
        if(compTypeVal == miopenFloat)
        {
            if constexpr(std::is_same_v<Tref, double>)
                RunImpl<double>(alpha, in_data, beta, out_data, parallel, indices);
            else
                RunImpl<float>(alpha, in_data, beta, out_data, parallel, indices);
        }
        else if(compTypeVal == miopenHalf)
        {
            if constexpr(std::is_same_v<Tref, double> || std::is_same_v<Tref, float>)
                RunImpl<Tref>(alpha, in_data, beta, out_data, parallel, indices);
            else
                RunImpl<float16>(alpha, in_data, beta, out_data, parallel, indices);
        }
        else if(compTypeVal == miopenDouble)
            RunImpl<double>(alpha, in_data, beta, out_data, parallel, indices);
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
    void RunImpl(float alpha,
                 std::vector<Tgpu>& in_data,
                 float beta,
                 std::vector<Tref>& out_data,
                 bool parallel,
                 std::vector<int>& indices)
    {
        bool need_indices =
            (indicesOpt == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES) &&
            (reduceOp == MIOPEN_REDUCE_TENSOR_MIN || reduceOp == MIOPEN_REDUCE_TENSOR_MAX ||
             reduceOp == MIOPEN_REDUCE_TENSOR_AMAX);

        auto [out_data_, indices_] = reduce_cpu_common<Tgpu, Tref, compType, int>(reduceOp,
                                                                                  nanOpt,
                                                                                  inLengths,
                                                                                  outLengths,
                                                                                  in_data,
                                                                                  inStrides,
                                                                                  out_data,
                                                                                  outStrides,
                                                                                  alpha,
                                                                                  beta,
                                                                                  parallel,
                                                                                  need_indices);

        out_data = std::move(out_data_.data);
        indices  = std::move(indices_.data);
    };
};

#endif
