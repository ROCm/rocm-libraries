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

#ifndef ROCALUTION_HIP_HIP_KERNELS_VECTOR_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_VECTOR_HPP_

#include <hip/hip_runtime.h>

#include "hip_atomics.hpp"

namespace rocalution
{
    template <typename ValueType, typename IndexType>
    __global__ void kernel_affine_transform(IndexType n,
                                            ValueType a,
                                            ValueType b,
                                            ValueType* __restrict__ inout)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        inout[ind] = (b - a) * inout[ind] + a;
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_scaleadd(IndexType n,
                                    ValueType alpha,
                                    const ValueType* __restrict__ x,
                                    ValueType* __restrict__ out)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        out[ind] = alpha * out[ind] + x[ind];
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_scaleaddscale(IndexType n,
                                         ValueType alpha,
                                         ValueType beta,
                                         const ValueType* __restrict__ x,
                                         ValueType* __restrict__ out)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        out[ind] = alpha * out[ind] + beta * x[ind];
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_scaleaddscale_offset(IndexType n,
                                                IndexType src_offset,
                                                IndexType dst_offset,
                                                ValueType alpha,
                                                ValueType beta,
                                                const ValueType* __restrict__ x,
                                                ValueType* __restrict__ out)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        out[ind + dst_offset] = alpha * out[ind + dst_offset] + beta * x[ind + src_offset];
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_scaleadd2(IndexType n,
                                     ValueType alpha,
                                     ValueType beta,
                                     ValueType gamma,
                                     const ValueType* __restrict__ x,
                                     const ValueType* __restrict__ y,
                                     ValueType* __restrict__ out)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        out[ind] = alpha * out[ind] + beta * x[ind] + gamma * y[ind];
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_pointwisemult(IndexType n,
                                         const ValueType* __restrict__ x,
                                         ValueType* __restrict__ out)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        out[ind] = out[ind] * x[ind];
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_pointwisemult2(IndexType n,
                                          const ValueType* __restrict__ x,
                                          const ValueType* __restrict__ y,
                                          ValueType* __restrict__ out)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        out[ind] = y[ind] * x[ind];
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_copy_offset_from(IndexType n,
                                            IndexType src_offset,
                                            IndexType dst_offset,
                                            const ValueType* __restrict__ in,
                                            ValueType* __restrict__ out)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        out[ind + dst_offset] = in[ind + src_offset];
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_permute(int64_t n,
                                   const IndexType* __restrict__ permute,
                                   const ValueType* __restrict__ in,
                                   ValueType* __restrict__ out)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        out[permute[ind]] = in[ind];
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_permute_backward(int64_t n,
                                            const IndexType* __restrict__ permute,
                                            const ValueType* __restrict__ in,
                                            ValueType* __restrict__ out)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        out[ind] = in[permute[ind]];
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_get_index_values(int64_t size,
                                            const IndexType* __restrict__ index,
                                            const ValueType* __restrict__ in,
                                            ValueType* __restrict__ out)
    {
        int64_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(i >= size)
        {
            return;
        }

        out[i] = in[index[i]];
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_set_index_values(int64_t size,
                                            const IndexType* __restrict__ index,
                                            const ValueType* __restrict__ in,
                                            ValueType* __restrict__ out)
    {
        int64_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(i >= size)
        {
            return;
        }

        out[index[i]] = in[i];
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_power(IndexType n, double power, ValueType* __restrict__ out)
    {
        IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        out[ind] = pow(out[ind], power);
    }

    template <typename ValueType>
    __global__ void
        kernel_copy_from_float(int64_t n, const float* __restrict__ in, ValueType* __restrict__ out)
    {
        int64_t ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        out[ind] = static_cast<ValueType>(in[ind]);
    }

    template <typename ValueType>
    __global__ void kernel_copy_from_double(int64_t n,
                                            const double* __restrict__ in,
                                            ValueType* __restrict__ out)
    {
        int64_t ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ind >= n)
        {
            return;
        }

        out[ind] = static_cast<ValueType>(in[ind]);
    }

    // Add index values from communication
    template <typename ValueType, typename IndexType>
    __global__ void kernel_add_index_values(int64_t size,
                                            const IndexType* __restrict__ index,
                                            const ValueType* __restrict__ in,
                                            ValueType* __restrict__ out)
    {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i >= size)
        {
            return;
        }

        // We need to use atomic add because we cannot be sure
        // that entries of index are unique
        atomicAdd(&out[index[i]], in[i]);
    }

    // Update and pack CF map for communication
    template <unsigned int BLOCKSIZE, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_rs_pmis_cf_update_pack(int64_t size,
                                           const IndexType* __restrict__ index,
                                           IndexType* __restrict__ in,
                                           IndexType* __restrict__ out)
    {
        int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i >= size)
        {
            return;
        }

        // Update from buffer
        if(in[i] == 0)
        {
            out[index[i]] = 0;
        }
        else
        {
            // Pack
            in[i] = out[index[i]];
        }
    }

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_VECTOR_HPP_
