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
// Each thread computes a DW_TD x DW_TH x DW_TW tile of output positions (depths x rows x columns)
// and accumulates in fp32. It walks the DW_PLANES input planes the tile needs; in each it reads the
// DW_ROWS input rows, each as one DW_SPAN-wide window held in registers, and applies every
// (output depth, output row, filter row) combination that meets at it, so each input element is
// read once per plane instead of once per output. NCDHW: consecutive threads take consecutive W
// tiles, so the accesses coalesce along W. NDHWC: consecutive threads take consecutive output
// channels, so they coalesce along C.
//
// Compile-time parameters (set by the solver):
//   DW_DTYPE            0 = fp32, 1 = fp16, 2 = bf16
//   DW_NDHWC            0 = NCDHW input / NKDHW output, 1 = NDHWC input / NDHWK output
//   DW_KZ, DW_KY, DW_KX filter depth, height, width
//   DW_SH, DW_SW        stride along H, W
//   DW_DH, DW_DW        dilation along H, W
//   DW_TD, DW_TH, DW_TW output tile per thread; DW_TD > 1 requires depth stride and dilation 1

#ifndef DW_DTYPE
#define DW_DTYPE 0
#endif
#ifndef DW_NDHWC
#define DW_NDHWC 0
#endif
#ifndef DW_KZ
#define DW_KZ 3
#endif
#ifndef DW_KY
#define DW_KY 3
#endif
#ifndef DW_KX
#define DW_KX 3
#endif
#ifndef DW_SH
#define DW_SH 1
#endif
#ifndef DW_SW
#define DW_SW 1
#endif
#ifndef DW_DH
#define DW_DH 1
#endif
#ifndef DW_DW
#define DW_DW 1
#endif
#ifndef DW_TD
#define DW_TD 1
#endif
#ifndef DW_TH
#define DW_TH 4
#endif
#ifndef DW_TW
#define DW_TW 4
#endif

#define DW_BLOCK_SIZE 256
#define DW_SPAN ((DW_TW - 1) * DW_SW + (DW_KX - 1) * DW_DW + 1)
#define DW_ROWS ((DW_TH - 1) * DW_SH + (DW_KY - 1) * DW_DH + 1)
#define DW_PLANES (DW_TD + DW_KZ - 1)

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
                                int Do,
                                int Ho,
                                int Wo,
                                int sd,
                                int pd,
                                int ph,
                                int pw,
                                int dd)
{
    const int mult   = K / C;
    const int wtiles = (Wo + DW_TW - 1) / DW_TW;
    const int htiles = (Ho + DW_TH - 1) / DW_TH;
    const int dtiles = (Do + DW_TD - 1) / DW_TD;
    long idx         = static_cast<long>(blockIdx.x) * DW_BLOCK_SIZE + threadIdx.x;
#if DW_NDHWC
    // idx = (((n * dtiles + dt) * htiles + ht) * wtiles + wt) * K + k
    if(idx >= static_cast<long>(N) * dtiles * htiles * wtiles * K)
        return;
    const int k = static_cast<int>(idx % K);
    idx /= K;
    const int wt = static_cast<int>(idx % wtiles);
    idx /= wtiles;
    const int ht = static_cast<int>(idx % htiles);
    idx /= htiles;
    const int dt        = static_cast<int>(idx % dtiles);
    const int n         = static_cast<int>(idx / dtiles);
    const long w_stride = C;
#else
    // idx = (((n * K + k) * dtiles + dt) * htiles + ht) * wtiles + wt
    if(idx >= static_cast<long>(N) * K * dtiles * htiles * wtiles)
        return;
    const int wt = static_cast<int>(idx % wtiles);
    idx /= wtiles;
    const int ht = static_cast<int>(idx % htiles);
    idx /= htiles;
    const int dt = static_cast<int>(idx % dtiles);
    idx /= dtiles;
    const int k         = static_cast<int>(idx % K);
    const int n         = static_cast<int>(idx / K);
    const long w_stride = 1;
#endif
    const int c       = k / mult;
    const int wo0     = wt * DW_TW;
    const int ho0     = ht * DW_TH;
    const int do0     = dt * DW_TD;
    const int wi0     = wo0 * DW_SW - pw;
    const int hi0     = ho0 * DW_SH - ph;
    const bool inside = wi0 >= 0 && wi0 + DW_SPAN <= Wi;

    float acc[DW_TD][DW_TH][DW_TW];
    for(int o = 0; o < DW_TD; ++o)
        for(int a = 0; a < DW_TH; ++a)
            for(int t = 0; t < DW_TW; ++t)
                acc[o][a][t] = 0.f;

    const data_t* w_k = wei + static_cast<long>(k) * (DW_KZ * DW_KY * DW_KX);
    // Input plane p feeds output depth o through filter depth z = p - o. With DW_TD == 1 this is
    // one plane per filter tap along D (any depth stride / dilation); DW_TD > 1 needs both to be 1.
#pragma unroll
    for(int p = 0; p < DW_PLANES; ++p)
    {
        const int di = do0 * sd - pd + p * dd;
        if(di < 0 || di >= Di)
            continue;
        float wv[DW_TD][DW_KY][DW_KX];
#pragma unroll
        for(int o = 0; o < DW_TD; ++o)
        {
            const int z = p - o;
            if(z < 0 || z >= DW_KZ)
                continue;
            for(int y = 0; y < DW_KY; ++y)
                for(int x = 0; x < DW_KX; ++x)
                    wv[o][y][x] = to_float(w_k[(z * DW_KY + y) * DW_KX + x]);
        }
#if DW_NDHWC
        const data_t* plane =
            in + (static_cast<long>(n) * Di + di) * Hi * static_cast<long>(Wi) * C + c;
#else
        const data_t* plane =
            in + ((static_cast<long>(n) * C + c) * Di + di) * Hi * static_cast<long>(Wi);
#endif
#pragma unroll
        for(int r = 0; r < DW_ROWS; ++r)
        {
            const int hi = hi0 + r;
            if(hi < 0 || hi >= Hi)
                continue;
            const data_t* row = plane + static_cast<long>(hi) * Wi * w_stride;
            float v[DW_SPAN];
            if(inside)
            {
#pragma unroll
                for(int j = 0; j < DW_SPAN; ++j)
                    v[j] = to_float(row[(wi0 + j) * w_stride]);
            }
            else
            {
#pragma unroll
                for(int j = 0; j < DW_SPAN; ++j)
                {
                    const int wi = wi0 + j;
                    v[j]         = (wi >= 0 && wi < Wi) ? to_float(row[wi * w_stride]) : 0.f;
                }
            }
            // Input row r feeds output row a through filter row y when r == a * SH + y * DH.
#pragma unroll
            for(int o = 0; o < DW_TD; ++o)
            {
                const int z = p - o;
                if(z < 0 || z >= DW_KZ)
                    continue;
#pragma unroll
                for(int a = 0; a < DW_TH; ++a)
                {
#pragma unroll
                    for(int y = 0; y < DW_KY; ++y)
                    {
                        if(a * DW_SH + y * DW_DH != r)
                            continue;
#pragma unroll
                        for(int t = 0; t < DW_TW; ++t)
#pragma unroll
                            for(int x = 0; x < DW_KX; ++x)
                                acc[o][a][t] = __builtin_fmaf(
                                    v[t * DW_SW + x * DW_DW], wv[o][y][x], acc[o][a][t]);
                    }
                }
            }
        }
    }

    for(int o = 0; o < DW_TD; ++o)
    {
        const int d_o = do0 + o;
        if(d_o >= Do)
            break;
        for(int a = 0; a < DW_TH; ++a)
        {
            const int ho = ho0 + a;
            if(ho >= Ho)
                break;
            for(int t = 0; t < DW_TW; ++t)
            {
                const int wo = wo0 + t;
                if(wo >= Wo)
                    break;
#if DW_NDHWC
                out[(((static_cast<long>(n) * Do + d_o) * Ho + ho) * Wo + wo) * K + k] =
                    from_float(acc[o][a][t]);
#else
                out[(((static_cast<long>(n) * K + k) * Do + d_o) * Ho + ho) * Wo + wo] =
                    from_float(acc[o][a][t]);
#endif
            }
        }
    }
}
