/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once
#include "memory_gfx.h"
#include <hip/hip_runtime.h>

template <size_t Begin, size_t End, size_t Increment = 1>
struct static_for
{
    template <typename F>
    __host__ __device__ constexpr void operator()(F const& f) const
    {
        if constexpr(Begin < End)
        {
            f(Begin);
            static_for<Begin + Increment, End>()(f);
        }
    }
};

constexpr size_t max(size_t a, size_t b)
{
    return (a > b) ? a : b;
}

template <typename DataTypeCompute, typename DataTypeOut, size_t MT0, size_t MT1, size_t VW>
__device__ inline void
    reductionKernel_ijk(DataTypeCompute* in, DataTypeOut* out, int m, int n, int strideJ)
{
    __shared__ DataTypeCompute block[MT0 * MT1 * VW];

    int mod = hipThreadIdx_x % MT0;
    int row = hipThreadIdx_x / MT0;

    int soffset = hipBlockIdx_x * MT0 * VW;
    int voffset = mod * VW;
    int idx     = voffset + soffset;

    int soffsetBytes      = soffset * sizeof(DataTypeCompute);
    int soffsetBytes_bias = soffset * sizeof(DataTypeOut);
    int num_records       = strideJ * n * sizeof(DataTypeCompute);
    int num_records_bias  = m * sizeof(DataTypeOut);

    constexpr size_t sumLength      = max((size_t)1, VW - 1);
    DataTypeCompute  sum[sumLength] = {0};
    if(idx + (VW - 1) < m)
    {
        for(int i = 0; i < n; i += MT1)
        {
            int             currRow   = row + i;
            int             rowStride = currRow * strideJ + voffset;
            DataTypeCompute tmp[VW];
            static_for<0, VW>()([&](int vw) {
                buffer_load<DataTypeCompute, sizeof(DataTypeCompute)>(
                    tmp[vw],
                    reinterpret_cast<void*>(const_cast<DataTypeCompute*>(in)),
                    (uint32_t)((rowStride + vw) * sizeof(DataTypeCompute)),
                    (uint32_t)(soffsetBytes),
                    num_records);
            });
            static_for<0, VW>()([&](int vw) { sum[vw] += tmp[vw]; });
        }
        static_for<0, VW>()([&](int vw) { block[(mod * MT1 + row) * VW + vw] = sum[vw]; });
        __syncthreads();
        if(hipThreadIdx_x < MT0)
        {
            for(int i = 1; i < MT1; i += 1)
            {
                static_for<0, VW>()(
                    [&](int vw) { sum[vw] += block[(i + hipThreadIdx_x * MT1) * VW + vw]; });
            }
            // This is a 1D bias
            static_for<0, VW>()([&](int vw) {
                buffer_store<DataTypeOut, sizeof(DataTypeOut)>(
                    (DataTypeOut)sum[vw],
                    reinterpret_cast<void*>(const_cast<DataTypeOut*>(out)),
                    (uint32_t)(voffset + vw) * sizeof(DataTypeOut),
                    (uint32_t)(soffsetBytes_bias),
                    num_records_bias);
            });
        }
    }
    else if(idx < m)
    {
        for(int i = 0; i < n; i += MT1)
        {
            int              currRow   = row + i;
            int              rowStride = currRow * strideJ + voffset;
            constexpr size_t tmpLength = max((size_t)1, VW - 1);
            DataTypeCompute  tmp[tmpLength];
            static_for<0, VW - 1>()([&](int vw) {
                buffer_load<DataTypeCompute, sizeof(DataTypeCompute)>(
                    tmp[vw],
                    reinterpret_cast<void*>(const_cast<DataTypeCompute*>(in)),
                    (uint32_t)((rowStride + vw) * sizeof(DataTypeCompute)),
                    (uint32_t)(soffsetBytes),
                    num_records);
            });
            static_for<0, VW - 1>()([&](int vw) { sum[vw] += tmp[vw]; });
        }
        static_for<0, VW - 1>()([&](int vw) { block[(mod * MT1 + row) * VW + vw] = sum[vw]; });
        __syncthreads();
        if(hipThreadIdx_x < MT0)
        {

            for(int i = 1; i < MT1; i += 1)
            {
                static_for<0, VW - 1>()(
                    [&](int vw) { sum[vw] += block[(i + hipThreadIdx_x * MT1) * VW + vw]; });
            }
            // This is a 1D bias
            static_for<0, VW - 1>()([&](int vw) {
                buffer_store<DataTypeOut, sizeof(DataTypeOut)>(
                    (DataTypeOut)sum[vw],
                    reinterpret_cast<void*>(const_cast<DataTypeOut*>(out)),
                    (uint32_t)(voffset + vw) * sizeof(DataTypeOut),
                    (uint32_t)(soffsetBytes_bias),
                    num_records_bias);
            });
        }
    }
    else
    {
        static_for<0, VW>()([&](int vw) { block[(mod * MT1 + row) * VW + vw] = 0; });
        __syncthreads();
    }
}
