// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Backward data of a depthwise convolution (groups == C, K a multiple of C), which is also the
// forward of a depthwise transposed convolution. Two kernels, selected by DW_GENERIC:
//
// - ConvDgradDepthwise1d (DW_GENERIC 0): the kernel of MIOpen's ConvHipDirectDepthwiseBwd1d
//   (MIOpenConvDepthwiseBwd1dHip.cpp) with its entry point renamed. NCL (NCHW with H == 1),
//   K == C, dilation 1; filter width and stride are compile-time constants, lengths, padding,
//   channels and batch are arguments.
// - ConvDgradDepthwise (DW_GENERIC 1): one dx element per lane, gathering every dy element and
//   tap that reaches it. 1-D to 3-D (a lower rank passes unit outer extents), channels-first or
//   channels-last, any stride, dilation and padding, K = M x C.

#include <hip/hip_runtime.h>

#ifndef DW_GENERIC
#define DW_GENERIC 0
#endif
#ifndef DW_DTYPE
#define DW_DTYPE 0 // 0 fp32, 1 fp16, 2 bf16
#endif

#define DW_BLOCK_SIZE 256

#if DW_DTYPE == 0
typedef float data_t;
__device__ static inline float to_float(data_t v)
{
    return v;
}
__device__ static inline data_t from_float(float v)
{
    return v;
}
#elif DW_DTYPE == 1
typedef _Float16 data_t;
__device__ static inline float to_float(data_t v)
{
    return static_cast<float>(v);
}
__device__ static inline data_t from_float(float v)
{
    return static_cast<data_t>(v);
}
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

#if !DW_GENERIC

// dx[o] = sum over (i, t) with i * S + t - P == o of dy[i] * w[t]. Writing o + P = S * q + r
// (0 <= r < S), the taps that reach output phase r are t = r + S * m, and they read dy[q - m].
// Each lane owns TW consecutive q, i.e. S * TW consecutive outputs, computed from one register
// window of dy, so dy is read without expanding it and dx is written once.
//
// Grid: x over blocks of 256 lanes along the output, y over channels, z over the batch, so the
// filter is wave-uniform and read through the scalar unit.

#ifndef DW_K
#define DW_K 12 // filter width
#endif
#ifndef DW_S
#define DW_S 2 // stride
#endif
#ifndef DW_TW
#define DW_TW 1 // q per lane
#endif

#define DW_TAPS_PER_PHASE ((DW_K + DW_S - 1) / DW_S)
#define DW_WINDOW (DW_TW + DW_TAPS_PER_PHASE - 1)
#define DW_ELEM_BYTES (DW_DTYPE == 0 ? 4 : 2)
#define DW_LANE_BYTES (DW_S * DW_TW * DW_ELEM_BYTES)

extern "C" __global__ void __launch_bounds__(DW_BLOCK_SIZE)
    ConvDgradDepthwise1d(const data_t* __restrict__ dy,
                         const data_t* __restrict__ wei,
                         data_t* __restrict__ dx,
                         int C,
                         int Li, // dy length
                         int Lo, // dx length
                         int P) // padding
{
    const int c = static_cast<int>(blockIdx.y);
    const long row = static_cast<long>(blockIdx.z) * C + c;
    const int q0
        = (static_cast<int>(blockIdx.x) * DW_BLOCK_SIZE + static_cast<int>(threadIdx.x)) * DW_TW;
    const int out0 = DW_S * q0 - P;
    if(out0 >= Lo)
        return;
    const data_t* in = dy + row * Li;
    data_t* out = dx + row * Lo;

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
            v[j] = (i >= 0 && i < Li) ? to_float(in[i]) : 0.0f;
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

#else // DW_GENERIC

#ifndef DW_CHANNELS_LAST
#define DW_CHANNELS_LAST 0
#endif

// dx (N, C, Di, Hi, Wi), w (K, 1, Z, Y, X), dy (N, K, Do, Ho, Wo) with K = M x C: dx channel c
// receives dy channels c x M .. c x M + M - 1. Channels-last stores C (K) innermost. Along
// each axis, dx coordinate x receives dy coordinate o through tap t when o * s + t * d - p == x.
extern "C" __global__ void __launch_bounds__(DW_BLOCK_SIZE)
    ConvDgradDepthwise(const data_t* __restrict__ dy,
                       const data_t* __restrict__ wei,
                       data_t* __restrict__ dx,
                       int N,
                       int C,
                       int Di,
                       int Hi,
                       int Wi,
                       int K,
                       int Do,
                       int Ho,
                       int Wo,
                       int Z,
                       int Y,
                       int X,
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
    const int M = K / C;
    const long long dhw = static_cast<long long>(Di) * Hi * Wi;
    const long long odhw = static_cast<long long>(Do) * Ho * Wo;
    const long long total = static_cast<long long>(N) * C * dhw;
    const long long step = static_cast<long long>(gridDim.x) * DW_BLOCK_SIZE;

    for(long long e = static_cast<long long>(blockIdx.x) * DW_BLOCK_SIZE + threadIdx.x; e < total;
        e += step)
    {
#if DW_CHANNELS_LAST
        const int c = static_cast<int>(e % C);
        const long long rest = e / C;
        const int xw = static_cast<int>(rest % Wi);
        const int xh = static_cast<int>((rest / Wi) % Hi);
        const int xd = static_cast<int>((rest / Wi / Hi) % Di);
        const int n = static_cast<int>(rest / dhw);
#else
        const int xw = static_cast<int>(e % Wi);
        const int xh = static_cast<int>((e / Wi) % Hi);
        const int xd = static_cast<int>((e / Wi / Hi) % Di);
        const int c = static_cast<int>((e / dhw) % C);
        const int n = static_cast<int>(e / dhw / C);
#endif
        float acc = 0.0f;
        for(int z = 0; z < Z; ++z)
        {
            const int od_s = xd + pd - z * dd;
            if(od_s < 0 || od_s % sd != 0 || od_s / sd >= Do)
                continue;
            const int od = od_s / sd;
            for(int y = 0; y < Y; ++y)
            {
                const int oh_s = xh + ph - y * dh;
                if(oh_s < 0 || oh_s % sh != 0 || oh_s / sh >= Ho)
                    continue;
                const int oh = oh_s / sh;
                for(int x = 0; x < X; ++x)
                {
                    const int ow_s = xw + pw - x * dw;
                    if(ow_s < 0 || ow_s % sw != 0 || ow_s / sw >= Wo)
                        continue;
                    const int ow = ow_s / sw;
                    const long long spatial_out = (static_cast<long long>(od) * Ho + oh) * Wo + ow;
                    const int tap = (z * Y + y) * X + x;
                    for(int j = 0; j < M; ++j)
                    {
                        const int k = c * M + j;
#if DW_CHANNELS_LAST
                        const long long dy_index
                            = (static_cast<long long>(n) * odhw + spatial_out) * K + k;
#else
                        const long long dy_index
                            = (static_cast<long long>(n) * K + k) * odhw + spatial_out;
#endif
                        acc = __builtin_fmaf(
                            to_float(dy[dy_index]),
                            to_float(wei[static_cast<long long>(k) * Z * Y * X + tap]),
                            acc);
                    }
                }
            }
        }
        dx[e] = from_float(acc);
    }
}

#endif // DW_GENERIC
