/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif
#include "miopen_cstdint.hpp"

#include "float_types.h"

template <typename TI, typename TO>
__device__ void layernormfwdcontiguous(const TI* __restrict__ x,
                                       const TI* __restrict__ weight,
                                       const TI* __restrict__ bias,
                                       TO* __restrict__ y,
                                       TO* __restrict__ mean,
                                       TO* __restrict__ rstd,
                                       float eps,
                                       uint64_t inner_size,
                                       int32_t mode)
{
    /*
     * Each group works on a single channel.
     * Example)
     * x dim = {N, C, L}, normalized shape = {C, L}
     * outer_size = N, inner_size = C * L
     *
     * Example2)
     * x dim = {N, C, L}, normalized shape = {L}
     * outer_size = N * C, inner_size = L
     *
     * => gws = {outer_size * LOCAL_SIZE}, lws = {LOCAL_SIZE}
     */

    /*
     * Reduction to calculate mean and rstd
     */

    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;

    FLOAT_ACCUM pmean = static_cast<FLOAT_ACCUM>(0);
    FLOAT_ACCUM pvar  = static_cast<FLOAT_ACCUM>(0);
    __shared__ FLOAT_ACCUM ltmp1[LOCAL_SIZE];
    __shared__ FLOAT_ACCUM ltmp2[LOCAL_SIZE];

    // reduce sum for mean and var
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t x_idx = gid * inner_size + i;

        FLOAT_ACCUM tmp = CVT_FLOAT2ACCUM(x[x_idx]);
        pmean += tmp;
        pvar += tmp * tmp;
    }

    ltmp1[lid] = pmean;
    ltmp2[lid] = pvar;
    __syncthreads();
    for(uint32_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp1[lid] += ltmp1[lid + i];
            ltmp2[lid] += ltmp2[lid + i];
        }
        __syncthreads();
    }
    pmean             = ltmp1[0] / inner_size;
    pvar              = ltmp2[0] / inner_size - pmean * pmean;
    FLOAT_ACCUM prstd = rsqrt(pvar + FLOAT_ACCUM(eps));

    if(lid == 0)
    {
        if(mean)
            mean[gid] = CVT_ACCUM2FLOAT(pmean);
        if(rstd)
            rstd[gid] = CVT_ACCUM2FLOAT(prstd);
    }

    // forward calculation
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t idx = gid * inner_size + i;

        FLOAT_ACCUM pweight;
        FLOAT_ACCUM pbias;

        pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE) ? CVT_FP32_2ACCUM(1.0f)
                                                      : CVT_FLOAT2ACCUM(weight[i]);
        pbias =
            (mode == MIOPEN_ELEMENTWISE_AFFINE) ? static_cast<FLOAT>(0) : CVT_FLOAT2ACCUM(bias[i]);

        FLOAT_ACCUM val = (CVT_FLOAT2ACCUM(x[idx]) - pmean) * prstd * pweight + pbias;
        y[idx]          = CVT_ACCUM2FLOAT(val);
    }
}

template <typename TI, typename TO>
__device__ void layernormfwdstride(const TI* __restrict__ x,
                                   const TI* __restrict__ weight,
                                   const TI* __restrict__ bias,
                                   TO* __restrict__ y,
                                   TO* __restrict__ mean,
                                   TO* __restrict__ rstd,
                                   float eps,
                                   uint64_t inner_size,
                                   uint64_t stride,
                                   int32_t mode)
{
    /*
     * Each group works on a single channel.
     * Example)
     * x dim = {N, C, L}, normalized shape = {C, L}
     * outer_size = N, inner_size = C * L, stride = 1
     *
     * Example2)
     * x dim = {N, C, L}, normalized shape = {L}
     * outer_size = N, inner_size = L, stride = C
     *
     * => gws = {outer_size * stride * LOCAL_SIZE}, lws = {LOCAL_SIZE}
     */

    /*
     * Reduction to calculate mean and rstd
     */

    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;
    const uint64_t o   = gid / stride;
    const uint64_t s   = gid % stride;

    FLOAT_ACCUM pmean = static_cast<FLOAT_ACCUM>(0);
    FLOAT_ACCUM pvar  = static_cast<FLOAT_ACCUM>(0);
    __shared__ FLOAT_ACCUM ltmp1[LOCAL_SIZE];
    __shared__ FLOAT_ACCUM ltmp2[LOCAL_SIZE];

    // reduce sum for mean and var
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t x_idx = o * inner_size * stride + i * stride + s;

        FLOAT_ACCUM tmp = CVT_FLOAT2ACCUM(x[x_idx]);
        pmean += tmp;
        pvar += tmp * tmp;
    }

    ltmp1[lid] = pmean;
    ltmp2[lid] = pvar;
    __syncthreads();
    for(uint32_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp1[lid] += ltmp1[lid + i];
            ltmp2[lid] += ltmp2[lid + i];
        }
        __syncthreads();
    }
    pmean             = ltmp1[0] / inner_size;
    pvar              = ltmp2[0] / inner_size - pmean * pmean;
    FLOAT_ACCUM prstd = rsqrt(pvar + FLOAT_ACCUM(eps));

    if(lid == 0)
    {
        if(mean)
            mean[gid] = CVT_ACCUM2FLOAT(pmean);
        if(rstd)
            rstd[gid] = CVT_ACCUM2FLOAT(prstd);
    }

    // forward calculation
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t idx = o * inner_size * stride + i * stride + s;

        FLOAT_ACCUM pweight;
        FLOAT_ACCUM pbias;

        pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE) ? CVT_FP32_2ACCUM(1.0f)
                                                      : CVT_FLOAT2ACCUM(weight[i]);
        pbias =
            (mode == MIOPEN_ELEMENTWISE_AFFINE) ? static_cast<FLOAT>(0) : CVT_FLOAT2ACCUM(bias[i]);

        FLOAT_ACCUM val = (CVT_FLOAT2ACCUM(x[idx]) - pmean) * prstd * pweight + pbias;
        y[idx]          = CVT_ACCUM2FLOAT(val);
    }
}

template <typename TI, typename TO>
__device__ void layernormbwdcontiguous(const TI* __restrict__ dy,
                                       const TI* __restrict__ x,
                                       const TI* __restrict__ weight,
                                       const TI* __restrict__ mean,
                                       const TI* __restrict__ rstd,
                                       TO* __restrict__ dx,
                                       uint64_t inner_size,
                                       int32_t mode)
{
    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;

    __shared__ FLOAT_ACCUM ltmp1[LOCAL_SIZE];
    __shared__ FLOAT_ACCUM ltmp2[LOCAL_SIZE];
    FLOAT_ACCUM sum_dy_weight   = 0;
    FLOAT_ACCUM sum_dy_weight_x = 0;

    // Reduce sums
    if(dy)
    {
        for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
        {
            size_t x_idx = gid * inner_size + i;

            FLOAT_ACCUM pdy_pweight =
                CVT_FLOAT2ACCUM(dy[x_idx]) * ((mode == MIOPEN_ELEMENTWISE_AFFINE)
                                                  ? CVT_FP32_2ACCUM(1.0f)
                                                  : CVT_FLOAT2ACCUM(weight[i]));

            sum_dy_weight += pdy_pweight;
            sum_dy_weight_x += pdy_pweight * CVT_FLOAT2ACCUM(x[x_idx]);
        }
    }

    ltmp1[lid] = sum_dy_weight;
    ltmp2[lid] = sum_dy_weight_x;
    __syncthreads();
    for(uint32_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp1[lid] += ltmp1[lid + i];
            ltmp2[lid] += ltmp2[lid + i];
        }
        __syncthreads();
    }

    sum_dy_weight     = ltmp1[0];
    sum_dy_weight_x   = ltmp2[0];
    FLOAT_ACCUM scale = 1.0f / inner_size;
    FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[gid]);
    FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(mean[gid]);
    FLOAT_ACCUM a     = prstd * prstd * prstd * scale * (sum_dy_weight_x - sum_dy_weight * pmean);
    FLOAT_ACCUM b     = prstd * sum_dy_weight * scale - a * pmean;

    // Backward calculation
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t idx = gid * inner_size + i;

        FLOAT_ACCUM pdy     = dy ? CVT_FLOAT2ACCUM(dy[idx]) : 0;
        FLOAT_ACCUM pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE) ? CVT_FP32_2ACCUM(1.0f)
                                                                  : CVT_FLOAT2ACCUM(weight[i]);

        FLOAT_ACCUM val = prstd * pdy * pweight - a * CVT_FLOAT2ACCUM(x[idx]) - b;
        dx[idx]         = CVT_ACCUM2FLOAT(val);
    }
}

template <typename TI, typename TO>
__device__ void layernormbwdstride(const TI* __restrict__ dy,
                                   const TI* __restrict__ x,
                                   const TI* __restrict__ weight,
                                   const TI* __restrict__ mean,
                                   const TI* __restrict__ rstd,
                                   TO* __restrict__ dx,
                                   uint64_t inner_size,
                                   uint64_t stride,
                                   int32_t mode)
{
    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;
    const uint64_t o   = gid / stride;
    const uint64_t s   = gid % stride;

    __shared__ FLOAT_ACCUM ltmp1[LOCAL_SIZE];
    __shared__ FLOAT_ACCUM ltmp2[LOCAL_SIZE];
    FLOAT_ACCUM sum_dy_weight = 0;
    FLOAT_ACCUM sum_dy_weight_x = 0;
    
    // Reduce sums
    if(dy)
    {
        for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
        {
            size_t x_idx = o * inner_size * stride + i * stride + s;

            FLOAT_ACCUM pdy_pweight = CVT_FLOAT2ACCUM(dy[x_idx]) * ((mode == MIOPEN_ELEMENTWISE_AFFINE) ? CVT_FP32_2ACCUM(1.0f)
                                                                                                        : CVT_FLOAT2ACCUM(weight[i]));
            
            sum_dy_weight += pdy_pweight;
            sum_dy_weight_x += pdy_pweight * CVT_FLOAT2ACCUM(x[x_idx]);
        }
    }

    ltmp1[lid] = sum_dy_weight;
    ltmp2[lid] = sum_dy_weight_x;
    __syncthreads();
    for(uint32_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp1[lid] += ltmp1[lid + i];
            ltmp2[lid] += ltmp2[lid + i];
        }
        __syncthreads();
    }

    sum_dy_weight = ltmp1[0];
    sum_dy_weight_x = ltmp2[0];
    FLOAT_ACCUM scale = 1.0f / inner_size;
    FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[gid]);
    FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(mean[gid]);
    FLOAT_ACCUM a = prstd * prstd * prstd * scale * (sum_dy_weight_x - sum_dy_weight * pmean);
    FLOAT_ACCUM b = prstd * sum_dy_weight * scale - a * pmean;

    // Backward calculation
    for(uint64_t i = lid; i < inner_size; i += LOCAL_SIZE)
    {
        size_t idx = o * inner_size * stride + i * stride + s;

        FLOAT_ACCUM pdy = dy ? CVT_FLOAT2ACCUM(dy[idx]) : 0;
        FLOAT_ACCUM pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE) ? CVT_FP32_2ACCUM(1.0f)
                                                                  : CVT_FLOAT2ACCUM(weight[i]);

        FLOAT_ACCUM val = prstd * pdy * pweight - a * CVT_FLOAT2ACCUM(x[idx]) - b;
        dx[idx] = CVT_ACCUM2FLOAT(val);
    }
}

template <typename TI, typename TO>
__device__ void layernormbwdweightbiascontiguous(const TI* __restrict__ dy,
                                                 const TI* __restrict__ x,
                                                 const TI* __restrict__ mean,
                                                 const TI* __restrict__ rstd,
                                                 TO* __restrict__ dw,
                                                 TO* __restrict__ db,
                                                 uint64_t outer_size,
                                                 uint64_t inner_size)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(dw || db)
    {
        FLOAT_ACCUM sum_dw = 0;
        FLOAT_ACCUM sum_db = 0;

        // Backward calculation
        for(uint64_t i = 0; i < outer_size; ++i)
        {
            uint64_t input_idx = i * inner_size + gid;

            FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[i]);
            FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(mean[i]);
            FLOAT_ACCUM pdy   = dy ? CVT_FLOAT2ACCUM(dy[input_idx]) : 0;

            sum_dw += prstd * pdy * (CVT_FLOAT2ACCUM(x[input_idx]) - pmean);
            sum_db += pdy;
        }

        if(dw)
        {
            dw[gid] = CVT_ACCUM2FLOAT(sum_dw);
        }
        if(db)
        {
            db[gid] = CVT_ACCUM2FLOAT(sum_db);
        }
    }
}

template <typename TI, typename TO>
__device__ void layernormbwdweightbiasstride(const TI* __restrict__ dy,
                                             const TI* __restrict__ x,
                                             const TI* __restrict__ mean,
                                             const TI* __restrict__ rstd,
                                             TO* __restrict__ dw,
                                             TO* __restrict__ db,
                                             uint64_t outer_size,
                                             uint64_t inner_size,
                                             uint64_t stride)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(dw || db)
    {
        FLOAT_ACCUM sum_dw = 0;
        FLOAT_ACCUM sum_db = 0;
        
        // Backward calculation
        for(uint64_t o = 0; o < outer_size; ++o)
        {
            for(uint64_t s = 0; s < stride; ++s)
            {
                uint64_t input_idx = o * inner_size * stride + gid * stride + s;

                FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[o * stride + s]);
                FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(mean[o * stride + s]);
                FLOAT_ACCUM pdy = dy ? CVT_FLOAT2ACCUM(dy[input_idx]) : 0;

                sum_dw += prstd * pdy * (CVT_FLOAT2ACCUM(x[input_idx]) - pmean);
                sum_db += pdy;
            }
        }

        if(dw)
        {
            dw[gid] = CVT_ACCUM2FLOAT(sum_dw);
        }
        if(db)
        {
            db[gid] = CVT_ACCUM2FLOAT(sum_db);
        }
    }
}

template <typename TI, typename TO>
__device__ void layernormbwdweightbiascontiguousparallel(const TI* __restrict__ dy,
                                                         const TI* __restrict__ x,
                                                         const TI* __restrict__ mean,
                                                         const TI* __restrict__ rstd,
                                                         TO* __restrict__ workspace,
                                                         uint64_t outer_size,
                                                         uint64_t inner_size,
                                                         uint64_t parallel_size)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= inner_size * parallel_size)
        return;

    uint64_t pid       = gid / inner_size;
    uint64_t input_idx = gid;

    FLOAT_ACCUM sum_dw = 0;
    FLOAT_ACCUM sum_db = 0;

    if(dy)
    {
        // Backward calculation
        for(uint64_t i = pid; i < outer_size; i += parallel_size)
        {
            FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[i]);
            FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(mean[i]);
            FLOAT_ACCUM pdy   = CVT_FLOAT2ACCUM(dy[input_idx]);

            sum_dw += pdy * prstd * (CVT_FLOAT2ACCUM(x[input_idx]) - pmean);
            sum_db += pdy;
            input_idx += inner_size * parallel_size;
        }
    }

    workspace[gid]                              = CVT_ACCUM2FLOAT(sum_dw);
    workspace[gid + parallel_size * inner_size] = CVT_ACCUM2FLOAT(sum_db);
}

template <typename TI, typename TO>
__device__ void layernormbwdweightbiasstrideparallel(const TI* __restrict__ dy,
                                                     const TI* __restrict__ x,
                                                     const TI* __restrict__ mean,
                                                     const TI* __restrict__ rstd,
                                                     TO* __restrict__ workspace,
                                                     uint64_t outer_size,
                                                     uint64_t inner_size,
                                                     uint64_t stride,
                                                     uint64_t parallel_size)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= inner_size * parallel_size)
        return;

    uint64_t pid   = gid / inner_size;
    uint64_t s_lid = (gid % inner_size) * stride;

    FLOAT_ACCUM sum_dw = 0;
    FLOAT_ACCUM sum_db = 0;

    if(dy)
    {
        // Backward calculation
        for(uint64_t i = pid; i < outer_size * stride; i += parallel_size)
        {
            uint64_t o = i / stride;
            uint64_t s = i % stride;
            uint64_t input_idx = o * inner_size * stride + s_lid + s;

            FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[i]);
            FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(mean[i]);
            FLOAT_ACCUM pdy = CVT_FLOAT2ACCUM(dy[input_idx]);

            sum_dw += pdy * prstd * (CVT_FLOAT2ACCUM(x[input_idx]) - pmean);
            sum_db += pdy;
        }
    }

    workspace[gid] = CVT_ACCUM2FLOAT(sum_dw);
    workspace[gid + parallel_size * inner_size] = CVT_ACCUM2FLOAT(sum_db);
}

template <typename TI, typename TO>
__device__ void layernormbwdreducesum(const TI* __restrict__ workspace,
                                      TO* __restrict__ dw,
                                      TO* __restrict__ db,
                                      uint64_t inner_size,
                                      uint64_t parallel_size)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= inner_size)
        return;

    if(dw || db)
    {
        FLOAT_ACCUM sum_dw = 0;
        FLOAT_ACCUM sum_db = 0;

        for(uint64_t i = 0; i < parallel_size; ++i)
        {
            uint64_t input_idx = i * inner_size + gid;
            sum_dw += CVT_FLOAT2ACCUM(workspace[input_idx]);
            sum_db += CVT_FLOAT2ACCUM(workspace[input_idx + parallel_size * inner_size]);
        }

        if(dw)
        {
            dw[gid] = CVT_ACCUM2FLOAT(sum_dw);
        }
        if(db)
        {
            db[gid] = CVT_ACCUM2FLOAT(sum_db);
        }
    }
}

extern "C" __global__ void LayernormFwdContiguous(const INPUT_TYPE* __restrict__ x,
                                                  const INPUT_TYPE* __restrict__ weight,
                                                  const INPUT_TYPE* __restrict__ bias,
                                                  OUTPUT_TYPE* __restrict__ y,
                                                  OUTPUT_TYPE* __restrict__ mean,
                                                  OUTPUT_TYPE* __restrict__ rstd,
                                                  float eps,
                                                  uint64_t inner_size,
                                                  int32_t mode)
{
    // instantiate the kernel
    layernormfwdcontiguous<INPUT_TYPE, OUTPUT_TYPE>(
        x, weight, bias, y, mean, rstd, eps, inner_size, mode);
}

extern "C" __global__ void LayernormFwdStride(const INPUT_TYPE* __restrict__ x,
                                              const INPUT_TYPE* __restrict__ weight,
                                              const INPUT_TYPE* __restrict__ bias,
                                              OUTPUT_TYPE* __restrict__ y,
                                              OUTPUT_TYPE* __restrict__ mean,
                                              OUTPUT_TYPE* __restrict__ rstd,
                                              float eps,
                                              uint64_t inner_size,
                                              uint64_t stride,
                                              int32_t mode)
{
    // instantiate the kernel
    layernormfwdstride<INPUT_TYPE, OUTPUT_TYPE>(
        x, weight, bias, y, mean, rstd, eps, inner_size, stride, mode);
}

extern "C" __global__ void LayernormBwdContiguous(const INPUT_TYPE* __restrict__ dy,
                                                  const INPUT_TYPE* __restrict__ x,
                                                  const INPUT_TYPE* __restrict__ weight,
                                                  const INPUT_TYPE* __restrict__ mean,
                                                  const INPUT_TYPE* __restrict__ rstd,
                                                  OUTPUT_TYPE* __restrict__ dx,
                                                  uint64_t inner_size,
                                                  int32_t mode)
{
    // instantiate the kernel
    layernormbwdcontiguous<INPUT_TYPE, OUTPUT_TYPE>(
        dy, x, weight, mean, rstd, dx, inner_size, mode);
}

extern "C" __global__ void LayernormBwdStride(const INPUT_TYPE* __restrict__ dy,
                                              const INPUT_TYPE* __restrict__ x,
                                              const INPUT_TYPE* __restrict__ weight,
                                              const INPUT_TYPE* __restrict__ mean,
                                              const INPUT_TYPE* __restrict__ rstd,
                                              OUTPUT_TYPE* __restrict__ dx,
                                              uint64_t inner_size,
                                              uint64_t stride,
                                              int32_t mode)
{
    // instantiate the kernel
    layernormbwdstride<INPUT_TYPE, OUTPUT_TYPE>(
        dy, x, weight, mean, rstd, dx, inner_size, stride, mode);
}

extern "C" __global__ void LayernormBwdWeightBiasContiguous(const INPUT_TYPE* __restrict__ dy,
                                                            const INPUT_TYPE* __restrict__ x,
                                                            const INPUT_TYPE* __restrict__ mean,
                                                            const INPUT_TYPE* __restrict__ rstd,
                                                            OUTPUT_TYPE* __restrict__ dw,
                                                            OUTPUT_TYPE* __restrict__ db,
                                                            uint64_t outer_size,
                                                            uint64_t inner_size)
{
    layernormbwdweightbiascontiguous<INPUT_TYPE, OUTPUT_TYPE>(
        dy, x, mean, rstd, dw, db, outer_size, inner_size);
}

extern "C" __global__ void LayernormBwdWeightBiasStride(const INPUT_TYPE* __restrict__ dy,
                                                        const INPUT_TYPE* __restrict__ x,
                                                        const INPUT_TYPE* __restrict__ mean,
                                                        const INPUT_TYPE* __restrict__ rstd,
                                                        OUTPUT_TYPE* __restrict__ dw,
                                                        OUTPUT_TYPE* __restrict__ db,
                                                        uint64_t outer_size,
                                                        uint64_t inner_size,
                                                        uint64_t stride)
{
    layernormbwdweightbiasstride<INPUT_TYPE, OUTPUT_TYPE>(
        dy, x, mean, rstd, dw, db, outer_size, inner_size, stride);
}

extern "C" __global__ void LayernormBwdWeightBiasContiguousParallel(const INPUT_TYPE* __restrict__ dy,
                                                                    const INPUT_TYPE* __restrict__ x,
                                                                    const INPUT_TYPE* __restrict__ mean,
                                                                    const INPUT_TYPE* __restrict__ rstd,
                                                                    OUTPUT_TYPE* __restrict__ workspace,
                                                                    uint64_t outer_size,
                                                                    uint64_t inner_size,
                                                                    uint64_t parallel_size)
{
    layernormbwdweightbiascontiguousparallel<INPUT_TYPE, OUTPUT_TYPE>(
        dy, x, mean, rstd, workspace, outer_size, inner_size, parallel_size);
}

extern "C" __global__ void LayernormBwdWeightBiasStrideParallel(const INPUT_TYPE* __restrict__ dy,
                                                                const INPUT_TYPE* __restrict__ x,
                                                                const INPUT_TYPE* __restrict__ mean,
                                                                const INPUT_TYPE* __restrict__ rstd,
                                                                OUTPUT_TYPE* __restrict__ workspace,
                                                                uint64_t outer_size,
                                                                uint64_t inner_size,
                                                                uint64_t stride,
                                                                uint64_t parallel_size)
{
    layernormbwdweightbiasstrideparallel<INPUT_TYPE, OUTPUT_TYPE>(
        dy, x, mean, rstd, workspace, outer_size, inner_size, stride, parallel_size);
}

extern "C" __global__ void LayernormBwdReduceSum(const INPUT_TYPE* __restrict__ workspace,
                                                 OUTPUT_TYPE* __restrict__ dw,
                                                 OUTPUT_TYPE* __restrict__ db,
                                                 uint64_t inner_size,
                                                 uint64_t parallel_size)
{
    layernormbwdreducesum<INPUT_TYPE, OUTPUT_TYPE>(
        workspace, dw, db, inner_size, parallel_size);
}
