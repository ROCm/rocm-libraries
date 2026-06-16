/* ************************************************************************
 * Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HIP_HIP_KERNELS_MCSR_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_MCSR_HPP_

#include <hip/hip_runtime.h>

namespace rocalution
{

    template <unsigned int WF_SIZE, typename ValueType, typename IndexType>
    __global__ void kernel_mcsr_spmv(IndexType nrow,
                                     const IndexType* __restrict__ row_offset,
                                     const IndexType* __restrict__ col,
                                     const ValueType* __restrict__ val,
                                     const ValueType* __restrict__ in,
                                     ValueType* __restrict__ out)
    {
        IndexType gid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        IndexType tid    = hipThreadIdx_x;
        IndexType laneid = tid & (WF_SIZE - 1);
        IndexType warpid = gid / WF_SIZE;
        IndexType nwarps = hipGridDim_x * hipBlockDim_x / WF_SIZE;

        for(IndexType ai = warpid; ai < nrow; ai += nwarps)
        {
            IndexType row_start = row_offset[ai];
            IndexType row_end   = row_offset[ai + 1];

            ValueType sum = static_cast<ValueType>(0);

            for(IndexType aj = row_start + laneid; aj < row_end; aj += WF_SIZE)
            {
                sum = sum + val[aj] * in[col[aj]];
            }

            wf_reduce_sum<WF_SIZE>(&sum);

            if(laneid == 0)
            {
                out[ai] = val[ai] * in[ai] + sum;
            }
        }
    }

    template <unsigned int WF_SIZE, typename ValueType, typename IndexType>
    __global__ void kernel_mcsr_add_spmv(IndexType nrow,
                                         const IndexType* __restrict__ row_offset,
                                         const IndexType* __restrict__ col,
                                         const ValueType* __restrict__ val,
                                         ValueType scalar,
                                         const ValueType* __restrict__ in,
                                         ValueType* __restrict__ out)
    {
        IndexType gid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        IndexType tid    = hipThreadIdx_x;
        IndexType laneid = tid % WF_SIZE;
        IndexType warpid = gid / WF_SIZE;
        IndexType nwarps = hipGridDim_x * hipBlockDim_x / WF_SIZE;

        for(IndexType ai = warpid; ai < nrow; ai += nwarps)
        {
            ValueType sum = static_cast<ValueType>(0);

            for(IndexType aj = row_offset[ai] + laneid; aj < row_offset[ai + 1]; aj += WF_SIZE)
            {
                sum = sum + scalar * val[aj] * in[col[aj]];
            }

            wf_reduce_sum<WF_SIZE>(&sum);

            if(laneid == 0)
            {
                out[ai] = out[ai] + scalar * val[ai] * in[ai] + sum;
            }
        }
    }

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_MCSR_HPP_
