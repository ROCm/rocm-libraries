/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_runtime.h>
#endif

// Depthwise forward 3D convolution (groups == C, K = C * multiplier) on plain FMA instructions.
//
// y[n][k][do][ho][wo] = sum_{z, y, x} x[n][k / mult][di][hi][wi] * w[k][0][z][y][x]
//
// Each thread computes DW_TW consecutive output positions along W of one output row and
// accumulates in fp32. NCDHW: consecutive threads take consecutive W tiles of one row, so the
// input and output accesses coalesce along W. NDHWC: consecutive threads take consecutive output
// channels, so the accesses coalesce along C.
//
// Compile-time parameters (set by the solver):
//   DW_DTYPE  0 = fp32, 1 = fp16, 2 = bf16
//   DW_NDHWC  0 = NCDHW input / NKDHW output, 1 = NDHWC input / NDHWK output

#ifndef DW_DTYPE
#define DW_DTYPE 0
#endif
#ifndef DW_NDHWC
#define DW_NDHWC 0
#endif

#define DW_TW 4
#define DW_BLOCK_SIZE 256

#if DW_DTYPE == 0
typedef float data_t;
__device__ static inline float to_float(data_t v) { return v; }
__device__ static inline data_t from_float(float v) { return v; }
#elif DW_DTYPE == 1
typedef _Float16 data_t;
__device__ static inline float to_float(data_t v) { return static_cast<float>(v); }
__device__ static inline data_t from_float(float v) { return static_cast<data_t>(v); }
#else
typedef unsigned short data_t;
__device__ static inline float to_float(data_t v)
{
    return __builtin_bit_cast(float, static_cast<unsigned int>(v) << 16);
}
__device__ static inline data_t from_float(float v)
{
    // Round to nearest even; NaN keeps a quiet mantissa bit.
    const unsigned int u = __builtin_bit_cast(unsigned int, v);
    if((u & 0x7fffffffu) > 0x7f800000u)
        return static_cast<data_t>((u >> 16) | 0x40u);
    return static_cast<data_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
#endif

extern "C" __global__ void __launch_bounds__(DW_BLOCK_SIZE)
    MIOpenConvDepthwise3dFwdHip(const data_t* __restrict__ in,
                                const data_t* __restrict__ wei,
                                data_t* __restrict__ out,
                                int N,
                                int C,
                                int Di,
                                int Hi,
                                int Wi,
                                int K,
                                int Z,
                                int Y,
                                int X,
                                int Do,
                                int Ho,
                                int Wo,
                                int sd,
                                int sh,
                                int sw,
                                int pd,
                                int ph,
                                int pw,
                                int dd,
                                int dh,
                                int dw)
{
    const int mult   = K / C;
    const int wtiles = (Wo + DW_TW - 1) / DW_TW;
    long idx         = static_cast<long>(blockIdx.x) * DW_BLOCK_SIZE + threadIdx.x;
#if DW_NDHWC
    // idx = (((n * Do + do) * Ho + ho) * wtiles + wt) * K + k
    if(idx >= static_cast<long>(N) * Do * Ho * wtiles * K)
        return;
    const int k = static_cast<int>(idx % K);
    idx /= K;
    const int wt = static_cast<int>(idx % wtiles);
    idx /= wtiles;
    const int ho = static_cast<int>(idx % Ho);
    idx /= Ho;
    const int d_o = static_cast<int>(idx % Do);
    const int n   = static_cast<int>(idx / Do);
#else
    // idx = (((n * K + k) * Do + do) * Ho + ho) * wtiles + wt
    if(idx >= static_cast<long>(N) * K * Do * Ho * wtiles)
        return;
    const int wt = static_cast<int>(idx % wtiles);
    idx /= wtiles;
    const int ho = static_cast<int>(idx % Ho);
    idx /= Ho;
    const int d_o = static_cast<int>(idx % Do);
    idx /= Do;
    const int k = static_cast<int>(idx % K);
    const int n = static_cast<int>(idx / K);
#endif
    const int c   = k / mult;
    const int wo0 = wt * DW_TW;

    float acc[DW_TW];
    for(int t = 0; t < DW_TW; ++t)
        acc[t] = 0.f;

    const data_t* w_k = wei + static_cast<long>(k) * Z * Y * X;
    for(int z = 0; z < Z; ++z)
    {
        const int di = d_o * sd - pd + z * dd;
        if(di < 0 || di >= Di)
            continue;
        for(int y = 0; y < Y; ++y)
        {
            const int hi = ho * sh - ph + y * dh;
            if(hi < 0 || hi >= Hi)
                continue;
#if DW_NDHWC
            const data_t* row =
                in + ((static_cast<long>(n) * Di + di) * Hi + hi) * static_cast<long>(Wi) * C + c;
            const long w_stride = C;
#else
            const data_t* row = in + ((static_cast<long>(n) * C + c) * Di + di) * Hi * Wi +
                                static_cast<long>(hi) * Wi;
            const long w_stride = 1;
#endif
            const data_t* w_zy = w_k + (z * Y + y) * X;
            for(int x = 0; x < X; ++x)
            {
                const float wv = to_float(w_zy[x]);
                for(int t = 0; t < DW_TW; ++t)
                {
                    const int wi  = (wo0 + t) * sw - pw + x * dw;
                    const float v = (wi >= 0 && wi < Wi) ? to_float(row[wi * w_stride]) : 0.f;
                    acc[t]        = __builtin_fmaf(v, wv, acc[t]);
                }
            }
        }
    }

    for(int t = 0; t < DW_TW; ++t)
    {
        const int wo = wo0 + t;
        if(wo >= Wo)
            break;
#if DW_NDHWC
        out[(((static_cast<long>(n) * Do + d_o) * Ho + ho) * Wo + wo) * K + k] = from_float(acc[t]);
#else
        out[(((static_cast<long>(n) * K + k) * Do + d_o) * Ho + ho) * Wo + wo] = from_float(acc[t]);
#endif
    }
}
