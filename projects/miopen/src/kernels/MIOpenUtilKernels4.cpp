/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
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
 */
#include "miopen_cstdint.hpp"

#ifndef MIOPEN_USE_FP32
#define MIOPEN_USE_FP32 0
#endif

#ifndef MIOPEN_USE_FP16
#define MIOPEN_USE_FP16 0
#endif

#ifndef MIOPEN_USE_BFP16
#define MIOPEN_USE_BFP16 0
#endif

#ifndef MIOPEN_USE_INT8
#define MIOPEN_USE_INT8 0
#endif

#ifndef MIOPEN_USE_INT32
#define MIOPEN_USE_INT32 0
#endif

#ifndef MIOPEN_USE_FP8
#define MIOPEN_USE_FP8 0
#endif

#ifndef MIOPEN_USE_BFP8
#define MIOPEN_USE_BFP8 0
#endif

#if MIOPEN_USE_INT8 || MIOPEN_USE_FP8 || MIOPEN_USE_BFP8
typedef char data_t;
#elif MIOPEN_USE_INT32
typedef int data_t;
#elif MIOPEN_USE_FP16 || MIOPEN_USE_BFP16
// As the half type degrades the performance, use short instead of half in
// transpose kernels, which have no match op. May change back to half when
// compile can deliver equal performance as short
typedef short data_t;
#elif MIOPEN_USE_FP32
typedef float data_t;
#endif

typedef uint64_t arg_size_t;

// N - batch size
// C - # of maps
// H - map height
// W - map width

// RD_BLCK = ((H*W)%8==0) ? 8 : ((H*W)%4==0) ? 4 : ((H*W)%3==0)? 3 : ((H*W)%2==0)? 2 : 1;
// HW_RD = (H*W)/RD_BLCK
// MAP_RD = HW_RD*C

// lcl size0 = ((MAP_RD + 63)/64 < 8) ? ((MAP_RD + 63)/64)*64 : 512;
// local size = (lcl size0, 1, 1)
// global size = (MAP_RD, N, 1)

extern "C" __global__ void transpose_NCHW2CNHW_V1_1D_WG_float_off64(const data_t* in,
                                                                    data_t* out,
                                                                    const arg_size_t in_off,
                                                                    const arg_size_t out_off,
                                                                    const int rd_blck,
                                                                    const int hw_rd,
                                                                    const int N,
                                                                    const int C,
                                                                    const int H,
                                                                    const int W)
{
    size_t max_size       = hw_rd * C;
    unsigned int c_p_blck = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_p_blck >= max_size)
        return;
    // to reduce granularity loss
    unsigned int c_i    = c_p_blck / hw_rd;
    unsigned int p_blck = c_p_blck - (c_i * hw_rd);

    /// \ref multiply_dims_overflow_assumption
    size_t in_offset  = c_p_blck * rd_blck + in_off;
    size_t out_offset = c_i * N * H * W + p_blck * rd_blck + out_off;
    const float* cin  = (const float*)(in + in_offset);
    float* cout       = (float*)(out + out_offset);

    for(unsigned int b = 0; b < N; b++)
    {
        cout[b * hw_rd] = cin[b * C * hw_rd];
    }
}

extern "C" __global__ void transpose_NCHW2CNHW_V1_1D_WG_float2_off64(const data_t* in,
                                                                     data_t* out,
                                                                     const arg_size_t in_off,
                                                                     const arg_size_t out_off,
                                                                     const int rd_blck,
                                                                     const int hw_rd,
                                                                     const int N,
                                                                     const int C,
                                                                     const int H,
                                                                     const int W)
{
    size_t max_size       = hw_rd * C;
    unsigned int c_p_blck = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_p_blck >= max_size)
        return;
    // to reduce granularity loss
    unsigned int c_i    = c_p_blck / hw_rd;
    unsigned int p_blck = c_p_blck - (c_i * hw_rd);

    /// \ref multiply_dims_overflow_assumption
    size_t in_offset  = c_p_blck * rd_blck + in_off;
    size_t out_offset = c_i * N * H * W + p_blck * rd_blck + out_off;
    const float2* cin = (const float2*)(in + in_offset);
    float2* cout      = (float2*)(out + out_offset);

    for(unsigned int b = 0; b < N; b++)
    {
        cout[b * hw_rd] = cin[b * C * hw_rd];
    }
}

extern "C" __global__ void transpose_NCHW2CNHW_V1_1D_WG_float4_off64(const data_t* in,
                                                                     data_t* out,
                                                                     const arg_size_t in_off,
                                                                     const arg_size_t out_off,
                                                                     const int rd_blck,
                                                                     const int hw_rd,
                                                                     const int N,
                                                                     const int C,
                                                                     const int H,
                                                                     const int W)
{
    size_t max_size       = hw_rd * C;
    unsigned int c_p_blck = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_p_blck >= max_size)
        return;
    // to reduce granularity loss
    unsigned int c_i    = c_p_blck / hw_rd;
    unsigned int p_blck = c_p_blck - (c_i * hw_rd);

    /// \ref multiply_dims_overflow_assumption
    size_t in_offset  = c_p_blck * rd_blck + in_off;
    size_t out_offset = c_i * N * H * W + p_blck * rd_blck + out_off;
    const float4* cin = (const float4*)(in + in_offset);
    float4* cout      = (float4*)(out + out_offset);

    for(unsigned int b = 0; b < N; b++)
    {
        cout[b * hw_rd] = cin[b * C * hw_rd];
    }
}

extern "C" __global__ void transpose_NCHW2CNHW_V1_2D_WG_float_off64(const data_t* in,
                                                                    data_t* out,
                                                                    const arg_size_t in_off,
                                                                    const arg_size_t out_off,
                                                                    const int rd_blck,
                                                                    const int hw_rd,
                                                                    const int N,
                                                                    const int C,
                                                                    const int H,
                                                                    const int W)
{
    size_t max_size       = hw_rd * C;
    unsigned int c_p_blck = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_p_blck >= max_size)
        return;
    // to reduce granularity loss
    unsigned int c_i    = c_p_blck / hw_rd;
    unsigned int p_blck = c_p_blck - (c_i * hw_rd);

    /// \ref multiply_dims_overflow_assumption
    size_t in_offset  = c_p_blck * rd_blck + in_off;
    size_t out_offset = c_i * N * H * W + p_blck * rd_blck + out_off;
    const float* cin  = (const float*)(in + in_offset);
    float* cout       = (float*)(out + out_offset);

    unsigned int b  = blockIdx.y * blockDim.y + threadIdx.y;
    cout[b * hw_rd] = cin[b * C * hw_rd];
}

extern "C" __global__ void transpose_NCHW2CNHW_V1_2D_WG_float2_off64(const data_t* in,
                                                                     data_t* out,
                                                                     const arg_size_t in_off,
                                                                     const arg_size_t out_off,
                                                                     const int rd_blck,
                                                                     const int hw_rd,
                                                                     const int N,
                                                                     const int C,
                                                                     const int H,
                                                                     const int W)
{
    size_t max_size       = hw_rd * C;
    unsigned int c_p_blck = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_p_blck >= max_size)
        return;
    // to reduce granularity loss
    unsigned int c_i    = c_p_blck / hw_rd;
    unsigned int p_blck = c_p_blck - (c_i * hw_rd);

    /// \ref multiply_dims_overflow_assumption
    size_t in_offset  = c_p_blck * rd_blck + in_off;
    size_t out_offset = c_i * N * H * W + p_blck * rd_blck + out_off;
    const float2* cin = (const float2*)(in + in_offset);
    float2* cout      = (float2*)(out + out_offset);

    unsigned int b  = blockIdx.y * blockDim.y + threadIdx.y;
    cout[b * hw_rd] = cin[b * C * hw_rd];
}

extern "C" __global__ void transpose_NCHW2CNHW_V1_2D_WG_float4_off64(const data_t* in,
                                                                     data_t* out,
                                                                     const arg_size_t in_off,
                                                                     const arg_size_t out_off,
                                                                     const int rd_blck,
                                                                     const int hw_rd,
                                                                     const int N,
                                                                     const int C,
                                                                     const int H,
                                                                     const int W)
{
    size_t max_size       = hw_rd * C;
    unsigned int c_p_blck = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_p_blck >= max_size)
        return;
    // to reduce granularity loss
    unsigned int c_i    = c_p_blck / hw_rd;
    unsigned int p_blck = c_p_blck - (c_i * hw_rd);

    /// \ref multiply_dims_overflow_assumption
    size_t in_offset  = c_p_blck * rd_blck + in_off;
    size_t out_offset = c_i * N * H * W + p_blck * rd_blck + out_off;
    const float4* cin = (const float4*)(in + in_offset);
    float4* cout      = (float4*)(out + out_offset);

    unsigned int b  = blockIdx.y * blockDim.y + threadIdx.y;
    cout[b * hw_rd] = cin[b * C * hw_rd];
}

extern "C" __global__ void transpose_NCHW2CNHW_V2_2D_WG_off64(const data_t* in,
                                                              data_t* out,
                                                              const arg_size_t in_off,
                                                              const arg_size_t out_off,
                                                              const int w_in,
                                                              const int w_out,
                                                              const int N,
                                                              const int C,
                                                              const int h_stride,
                                                              const int w_stride,
                                                              const int hw_in,
                                                              const int hw_out)
{
    unsigned int hw_i = blockIdx.x * blockDim.x + threadIdx.x;
    if(hw_i >= hw_out)
        return;

    unsigned int c_i = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int h_i = hw_i / w_out;
    unsigned int w_i = hw_i % w_out;

    /// \ref multiply_dims_overflow_assumption
    size_t in_offset  = c_i * hw_in + h_i * h_stride * w_in + w_i * w_stride + in_off;
    size_t out_offset = c_i * N * hw_out + hw_i + out_off;
    const data_t* cin = (const data_t*)(in + in_offset);
    data_t* cout      = (data_t*)(out + out_offset);

    for(unsigned int n_i = 0; n_i < N; n_i++)
        cout[hw_out * n_i] = cin[C * hw_in * n_i];
}

extern "C" __global__ void transpose_CNHW2NCHW_V1_1D_WG_float_off64(const data_t* in,
                                                                    data_t* out,
                                                                    const arg_size_t in_off,
                                                                    const arg_size_t out_off,
                                                                    const int rd_blck,
                                                                    const int hw_rd,
                                                                    const int N,
                                                                    const int C,
                                                                    const int H,
                                                                    const int W)
{
    // to reduce granularity loss
    unsigned int c_p_blck = blockIdx.x * blockDim.x + threadIdx.x;
    size_t max_size       = hw_rd * C;
    if(c_p_blck >= max_size)
        return;
    unsigned int c_i    = c_p_blck / hw_rd;
    unsigned int p_blck = c_p_blck % hw_rd;

    /// \ref multiply_dims_overflow_assumption
    size_t in_offset  = c_i * N * H * W + p_blck * rd_blck + in_off;
    size_t out_offset = c_p_blck * rd_blck + out_off;
    const float* cin  = (const float*)(in + in_offset);
    float* cout       = (float*)(out + out_offset);

    for(unsigned int b = 0; b < N; b++)
    {
        cout[b * C * hw_rd] = cin[b * hw_rd];
    }
}

extern "C" __global__ void transpose_CNHW2NCHW_V1_1D_WG_float2_off64(const data_t* in,
                                                                     data_t* out,
                                                                     const arg_size_t in_off,
                                                                     const arg_size_t out_off,
                                                                     const int rd_blck,
                                                                     const int hw_rd,
                                                                     const int N,
                                                                     const int C,
                                                                     const int H,
                                                                     const int W)
{
    // to reduce granularity loss
    unsigned int c_p_blck = blockIdx.x * blockDim.x + threadIdx.x;
    size_t max_size       = hw_rd * C;
    if(c_p_blck >= max_size)
        return;
    unsigned int c_i    = c_p_blck / hw_rd;
    unsigned int p_blck = c_p_blck % hw_rd;

    /// \ref multiply_dims_overflow_assumption
    size_t in_offset  = c_i * N * H * W + p_blck * rd_blck + in_off;
    size_t out_offset = c_p_blck * rd_blck + out_off;
    const float2* cin = (const float2*)(in + in_offset);
    float2* cout      = (float2*)(out + out_offset);

    for(unsigned int b = 0; b < N; b++)
    {
        cout[b * C * hw_rd] = cin[b * hw_rd];
    }
}

extern "C" __global__ void transpose_CNHW2NCHW_V1_1D_WG_float4_off64(const data_t* in,
                                                                     data_t* out,
                                                                     const arg_size_t in_off,
                                                                     const arg_size_t out_off,
                                                                     const int rd_blck,
                                                                     const int hw_rd,
                                                                     const int N,
                                                                     const int C,
                                                                     const int H,
                                                                     const int W)
{
    // to reduce granularity loss
    unsigned int c_p_blck = blockIdx.x * blockDim.x + threadIdx.x;
    size_t max_size       = hw_rd * C;
    if(c_p_blck >= max_size)
        return;
    unsigned int c_i    = c_p_blck / hw_rd;
    unsigned int p_blck = c_p_blck % hw_rd;

    /// \ref multiply_dims_overflow_assumption
    size_t in_offset  = c_i * N * H * W + p_blck * rd_blck + in_off;
    size_t out_offset = c_p_blck * rd_blck + out_off;
    const float4* cin = (const float4*)(in + in_offset);
    float4* cout      = (float4*)(out + out_offset);

    for(unsigned int b = 0; b < N; b++)
    {
        cout[b * C * hw_rd] = cin[b * hw_rd];
    }
}

extern "C" __global__ void transpose_CNHW2NCHW_V1_2D_WG_float_off64(const data_t* in,
                                                                    data_t* out,
                                                                    const arg_size_t in_off,
                                                                    const arg_size_t out_off,
                                                                    const int rd_blck,
                                                                    const int hw_rd,
                                                                    const int N,
                                                                    const int C,
                                                                    const int H,
                                                                    const int W)
{
    // to reduce granularity loss
    unsigned int c_p_blck = blockIdx.x * blockDim.x + threadIdx.x;
    size_t max_size       = hw_rd * C;
    if(c_p_blck >= max_size)
        return;
    unsigned int c_i    = c_p_blck / hw_rd;
    unsigned int p_blck = c_p_blck % hw_rd;

    /// \ref multiply_dims_overflow_assumption
    size_t in_offset  = c_i * N * H * W + p_blck * rd_blck + in_off;
    size_t out_offset = c_p_blck * rd_blck + out_off;
    const float* cin  = (const float*)(in + in_offset);
    float* cout       = (float*)(out + out_offset);

    unsigned int b      = blockIdx.y * blockDim.y + threadIdx.y;
    cout[b * C * hw_rd] = cin[b * hw_rd];
}

extern "C" __global__ void transpose_CNHW2NCHW_V1_2D_WG_float2_off64(const data_t* in,
                                                                     data_t* out,
                                                                     const arg_size_t in_off,
                                                                     const arg_size_t out_off,
                                                                     const int rd_blck,
                                                                     const int hw_rd,
                                                                     const int N,
                                                                     const int C,
                                                                     const int H,
                                                                     const int W)
{
    // to reduce granularity loss
    unsigned int c_p_blck = blockIdx.x * blockDim.x + threadIdx.x;
    size_t max_size       = hw_rd * C;
    if(c_p_blck >= max_size)
        return;
    unsigned int c_i    = c_p_blck / hw_rd;
    unsigned int p_blck = c_p_blck % hw_rd;

    /// \ref multiply_dims_overflow_assumption
    size_t in_offset  = c_i * N * H * W + p_blck * rd_blck + in_off;
    size_t out_offset = c_p_blck * rd_blck + out_off;
    const float2* cin = (const float2*)(in + in_offset);
    float2* cout      = (float2*)(out + out_offset);

    unsigned int b      = blockIdx.y * blockDim.y + threadIdx.y;
    cout[b * C * hw_rd] = cin[b * hw_rd];
}

extern "C" __global__ void transpose_CNHW2NCHW_V1_2D_WG_float4_off64(const data_t* in,
                                                                     data_t* out,
                                                                     const arg_size_t in_off,
                                                                     const arg_size_t out_off,
                                                                     const int rd_blck,
                                                                     const int hw_rd,
                                                                     const int N,
                                                                     const int C,
                                                                     const int H,
                                                                     const int W)
{
    // to reduce granularity loss
    unsigned int c_p_blck = blockIdx.x * blockDim.x + threadIdx.x;
    size_t max_size       = hw_rd * C;
    if(c_p_blck >= max_size)
        return;
    unsigned int c_i    = c_p_blck / hw_rd;
    unsigned int p_blck = c_p_blck % hw_rd;

    /// \ref multiply_dims_overflow_assumption
    size_t in_offset  = c_i * N * H * W + p_blck * rd_blck + in_off;
    size_t out_offset = c_p_blck * rd_blck + out_off;
    const float4* cin = (const float4*)(in + in_offset);
    float4* cout      = (float4*)(out + out_offset);

    unsigned int b      = blockIdx.y * blockDim.y + threadIdx.y;
    cout[b * C * hw_rd] = cin[b * hw_rd];
}

extern "C" __global__ void transpose_CNHW2NCHW_V2_2D_WG_off64(const data_t* in,
                                                              data_t* out,
                                                              const arg_size_t in_off,
                                                              const arg_size_t out_off,
                                                              const int w_in,
                                                              const int w_out,
                                                              const int N,
                                                              const int C,
                                                              const int h_stride,
                                                              const int w_stride,
                                                              const int hw_in,
                                                              const int hw_out)
{
    unsigned int hw_i = blockIdx.x * blockDim.x + threadIdx.x;
    if(hw_i >= hw_out)
        return;
    unsigned int c_i = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int h_i = hw_i / w_out;
    unsigned int w_i = hw_i % w_out;

    /// \ref multiply_dims_overflow_assumption
    size_t in_offset  = c_i * N * hw_out + hw_i + in_off;
    size_t out_offset = c_i * hw_in + h_i * h_stride * w_in + w_i * w_stride + out_off;
    const data_t* cin = (const data_t*)(in + in_offset);
    data_t* cout      = (data_t*)(out + out_offset);

    for(unsigned int n_i = 0; n_i < N; n_i++)
    {
        cout[C * hw_in * n_i] = cin[hw_out * n_i];
    }
}

extern "C" __global__ void transpose_packed_MN2NM_off64(const data_t* in,
                                                        data_t* out,
                                                        const int N,
                                                        const int M,
                                                        const arg_size_t in_off,
                                                        const arg_size_t out_off)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < M * N)
    {
        const unsigned int m_i = i / N;
        const unsigned int n_i = i % N;

        /// \ref multiply_dims_overflow_assumption
        /// Computations in 32-bit domain save a bit of performance.
        /// We assume that up to 3 dimensions can me multiplied in
        /// 32-bit domain without overflow. See discussion at:
        /// https://github.com/ROCm/MIOpen/pull/2613/files#r1429241306
        const size_t in_offset  = m_i * N + n_i + in_off;
        const size_t out_offset = n_i * M + m_i + out_off;

        const data_t* cin = (const data_t*)(in + in_offset);
        data_t* cout      = (data_t*)(out + out_offset);

        *cout = *cin;
    }
}
