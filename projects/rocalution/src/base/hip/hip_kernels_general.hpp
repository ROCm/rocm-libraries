/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
 * ************************************************************************ */

#ifndef ROCALUTION_HIP_HIP_KERNELS_GENERAL_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_GENERAL_HPP_

#include <hip/hip_runtime.h>

namespace rocalution
{

    template <typename ValueType, typename IndexType>
    __global__ void kernel_set_to_value(IndexType n, ValueType* __restrict__ data, ValueType val)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        data[ind] = val;
    }

    template <typename IndexType>
    __global__ void kernel_reverse_index(IndexType n,
                                         const IndexType* __restrict__ perm,
                                         IndexType* __restrict__ out)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        out[perm[ind]] = ind;
    }

    template <typename ValueType, typename IndexType>
    __global__ void
        kernel_buffer_addscalar(IndexType n, ValueType scalar, ValueType* __restrict__ buff)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        buff[ind] = buff[ind] + scalar;
    }

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_GENERAL_HPP_
