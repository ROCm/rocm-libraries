// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Backward data of a depthwise conv1d (groups == channels, one channel per group), which is
// also the forward of a depthwise ConvTranspose1d. NCL layout, stored as NCHW with H == 1.
//
// dx[o] = sum over (i, t) with i * S + t - P == o of dy[i] * w[t]. Writing o + P = S * q + r
// (0 <= r < S), the taps that reach output phase r are t = r + S * m, and they read dy[q - m].
// Each lane owns TW consecutive q, i.e. S * TW consecutive outputs, computed from one register
// window of dy, so dy is read without expanding it and dx is written once.
//
// Grid: x over blocks of 256 lanes along the output, y over channels, z over the batch, so the
// filter is wave-uniform and read through the scalar unit.

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_runtime.h>
#endif

#ifndef DW_DTYPE
#define DW_DTYPE 0 // 0 fp32, 1 fp16, 2 bf16
#endif
#ifndef DW_K
#define DW_K 12 // filter width
#endif
#ifndef DW_S
#define DW_S 2 // stride
#endif
#ifndef DW_TW
#define DW_TW 1 // q per lane
#endif

#define DW_BLOCK_SIZE 256
#define DW_TAPS_PER_PHASE ((DW_K + DW_S - 1) / DW_S)
#define DW_WINDOW (DW_TW + DW_TAPS_PER_PHASE - 1)
#define DW_ELEM_BYTES (DW_DTYPE == 0 ? 4 : 2)
#define DW_LANE_BYTES (DW_S * DW_TW * DW_ELEM_BYTES)

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
// Round to nearest even; NaN stays NaN.
__device__ static inline data_t from_float(float v)
{
    const unsigned int u = __builtin_bit_cast(unsigned int, v);
    if((u & 0x7fffffffu) > 0x7f800000u)
        return static_cast<data_t>((u >> 16) | 0x40u);
    return static_cast<data_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
#endif

extern "C" __global__ void __launch_bounds__(DW_BLOCK_SIZE)
    MIOpenConvDepthwiseBwd1dHip(const data_t* __restrict__ dy,
                                const data_t* __restrict__ wei,
                                data_t* __restrict__ dx,
                                int C,
                                int Li, // dy length
                                int Lo, // dx length
                                int P)  // padding
{
    const int c    = static_cast<int>(blockIdx.y);
    const long row = static_cast<long>(blockIdx.z) * C + c;
    const int q0 =
        (static_cast<int>(blockIdx.x) * DW_BLOCK_SIZE + static_cast<int>(threadIdx.x)) * DW_TW;
    const int out0 = DW_S * q0 - P;
    if(out0 >= Lo)
        return;
    const data_t* in = dy + row * Li;
    data_t* out      = dx + row * Lo;

    float w[DW_K];
#pragma unroll
    for(int t = 0; t < DW_K; ++t)
        w[t] = to_float(wei[c * DW_K + t]);

    // dy[q0 - (DW_TAPS_PER_PHASE - 1)] .. dy[q0 + DW_TW - 1]
    float v[DW_WINDOW];
    const int in0 = q0 - (DW_TAPS_PER_PHASE - 1);
    if(in0 >= 0 && in0 + DW_WINDOW <= Li)
    {
#pragma unroll
        for(int j = 0; j < DW_WINDOW; ++j)
            v[j] = to_float(in[in0 + j]);
    }
    else
    {
#pragma unroll
        for(int j = 0; j < DW_WINDOW; ++j)
        {
            const int i = in0 + j;
            v[j]        = (i >= 0 && i < Li) ? to_float(in[i]) : 0.0f;
        }
    }

    data_t res[DW_S * DW_TW];
#pragma unroll
    for(int a = 0; a < DW_TW; ++a)
    {
#pragma unroll
        for(int r = 0; r < DW_S; ++r)
        {
            float acc = 0.0f;
#pragma unroll
            for(int m = 0; m < DW_TAPS_PER_PHASE; ++m)
            {
                const int t = r + DW_S * m;
                if(t < DW_K)
                    acc = __builtin_fmaf(v[a + DW_TAPS_PER_PHASE - 1 - m], w[t], acc);
            }
            res[DW_S * a + r] = from_float(acc);
        }
    }

    const bool inside = out0 >= 0 && out0 + DW_S * DW_TW <= Lo;
#if DW_LANE_BYTES == 4 || DW_LANE_BYTES == 8 || DW_LANE_BYTES == 16
    // One vector store per lane when its outputs are aligned: 16-bit types gain from it.
    if(inside && reinterpret_cast<unsigned long>(out + out0) % DW_LANE_BYTES == 0)
    {
#if DW_LANE_BYTES == 16
        typedef unsigned int vec_t __attribute__((ext_vector_type(4)));
#elif DW_LANE_BYTES == 8
        typedef unsigned int vec_t __attribute__((ext_vector_type(2)));
#else
        typedef unsigned int vec_t;
#endif
        *reinterpret_cast<vec_t*>(out + out0) = __builtin_bit_cast(vec_t, res);
        return;
    }
#endif
#pragma unroll
    for(int e = 0; e < DW_S * DW_TW; ++e)
    {
        const int o = out0 + e;
        if(inside || (o >= 0 && o < Lo))
            out[o] = res[e];
    }
}
