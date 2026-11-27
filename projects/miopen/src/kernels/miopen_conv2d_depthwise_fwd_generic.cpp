// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// General-shape 2D depthwise forward convolution for packed NCHW FP16/BF16 tensors.
//
// Compile-time parameters (set by ConvGenericDepthwiseFwd2D, see
// solver/conv/conv_generic_depthwise_fwd_2d.cpp). They
// depend only on the data type and the filter geometry, so one binary serves every spatial size:
//   IO_DTYPE        __half or hip_bfloat16
//   DW_FY, DW_FX    filter height and width
//   DW_SH, DW_SW    stride
//   DW_DH, DW_DW    dilation
//   DW_VO           consecutive output columns computed by one thread (2, 4 or 8)
//   DW_USE_LDS      0: read the input through the cache (small problems, 3x3 filters with stride)
//                   1: stage an input tile in LDS first; tile sizes are kernel arguments
//   DW_LDS_FLOATS   LDS buffer size in floats (only for DW_USE_LDS=1)
//
// Every output is accumulated in FP32 with fused multiply-adds in filter order (row, then
// column) and rounded once to IO_DTYPE, so for finite data both variants produce bitwise
// identical results.

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#endif

#ifndef IO_DTYPE
#define IO_DTYPE __half
#endif

#define DW_BLOCK_SIZE 256

typedef unsigned long long dw_u64;

__device__ inline float dw_to_float(__half v) { return __half2float(v); }
__device__ inline float dw_to_float(hip_bfloat16 v) { return static_cast<float>(v); }

__device__ inline unsigned short dw_to_bits(float v, __half)
{
    return __half_as_ushort(__float2half(v));
}
__device__ inline unsigned short dw_to_bits(float v, hip_bfloat16) { return hip_bfloat16(v).data; }

constexpr int dw_span  = (DW_VO - 1) * DW_SW + (DW_FX - 1) * DW_DW + 1;
constexpr int dw_span4 = (dw_span + 3) / 4 * 4;

// Writes DW_VO consecutive outputs starting at column ow of one output row. Uses one vector store
// when the address allows it; the partial last strip of a row is written element by element.
__device__ inline void dw_store(unsigned short* op, int ow, int wo, const float (&acc)[DW_VO])
{
    unsigned short bits[DW_VO];
#pragma unroll
    for(int v = 0; v < DW_VO; ++v)
        bits[v] = dw_to_bits(acc[v], IO_DTYPE{});
    const dw_u64 addr = reinterpret_cast<dw_u64>(op);
    if(ow + DW_VO <= wo && addr % (2 * DW_VO) == 0)
    {
        typedef unsigned short vec_t __attribute__((ext_vector_type(DW_VO)));
        // cppcheck does not model ext_vector_type: it treats t as a scalar and misses that every
        // lane is written before the store.
        vec_t t;
#pragma unroll
        for(int v = 0; v < DW_VO; ++v)
            // cppcheck-suppress legacyUninitvar
            t[v] = bits[v];
        // cppcheck-suppress uninitvar
        *reinterpret_cast<vec_t*>(op) = t;
    }
    else if(DW_VO % 2 == 0 && ow + DW_VO <= wo && addr % 4 == 0)
    {
        typedef unsigned short vec2_t __attribute__((ext_vector_type(2)));
#pragma unroll
        for(int v = 0; v + 1 < DW_VO; v += 2)
        {
            vec2_t t;
            // cppcheck-suppress uninitvar
            t[0]                               = bits[v];
            t[1]                               = bits[v + 1];
            *reinterpret_cast<vec2_t*>(op + v) = t;
        }
    }
    else
    {
#pragma unroll
        for(int v = 0; v < DW_VO; ++v)
            if(ow + v < wo)
                op[v] = bits[v];
    }
}

#if DW_USE_LDS

// One workgroup computes tile_h output rows x tile_w output columns for tile_planes consecutive
// planes (plane = one (n, c) pair). The input rows and columns the tile needs, including the zero
// halo for padding, are converted to FP32 in LDS once and then reused by every filter tap. The
// filter of each plane follows the input tiles in the same buffer. The host guarantees
// tile_planes * (in_rows * lds_pitch + DW_FY * DW_FX) <= DW_LDS_FLOATS, tile_w % DW_VO == 0,
// lds_pitch % 4 == 0 and lds_pitch >= (tile_w / DW_VO - 1) * DW_VO * DW_SW + dw_span4.
extern "C" __global__ void __launch_bounds__(DW_BLOCK_SIZE)
    miopen_conv2d_depthwise_fwd_generic(const IO_DTYPE* __restrict__ input,
                                        const IO_DTYPE* __restrict__ weights,
                                        IO_DTYPE* __restrict__ output,
                                        int channels,
                                        int planes,
                                        int in_h,
                                        int in_w,
                                        int out_h,
                                        int out_w,
                                        int pad_h,
                                        int pad_w,
                                        int tile_h,
                                        int tile_w,
                                        int tile_planes,
                                        int lds_pitch,
                                        int tiles_w,
                                        int tiles_h)
{
    __shared__ __attribute__((aligned(16))) float lds[DW_LDS_FLOATS];

    const int in_rows    = (tile_h - 1) * DW_SH + (DW_FY - 1) * DW_DH + 1;
    const int plane_lds  = in_rows * lds_pitch;
    const int total_in   = tile_planes * plane_lds;
    float* const lds_wei = lds + total_in;

    int b                 = blockIdx.x;
    const int tw          = b % tiles_w;
    b                     = b / tiles_w;
    const int th          = b % tiles_h;
    const int nc0         = (b / tiles_h) * tile_planes;
    const int oh0         = th * tile_h;
    const int ow0         = tw * tile_w;
    const int ih0         = oh0 * DW_SH - pad_h;
    const int iw0         = ow0 * DW_SW - pad_w;
    const int tid         = threadIdx.x;
    const dw_u64 plane_in = static_cast<dw_u64>(in_h) * in_w;

    for(int i = tid; i < tile_planes * (DW_FY * DW_FX); i += DW_BLOCK_SIZE)
    {
        const int p  = i / (DW_FY * DW_FX);
        const int k  = i - p * (DW_FY * DW_FX);
        const int nc = nc0 + p;
        // The channel count can reach INT32_MAX, so the filter base is formed in 64 bits.
        lds_wei[i] =
            nc < planes
                ? dw_to_float(weights[static_cast<dw_u64>(nc % channels) * (DW_FY * DW_FX) + k])
                : 0.0f;
    }

    // Eight independent loads per thread are issued before their LDS stores so the global
    // latencies overlap.
    for(int i0 = tid; i0 < total_in; i0 += DW_BLOCK_SIZE * 8)
    {
        float v[8];
#pragma unroll
        for(int k = 0; k < 8; ++k)
        {
            const int i = i0 + k * DW_BLOCK_SIZE;
            v[k]        = 0.0f;
            if(i < total_in)
            {
                const int row = i / lds_pitch;
                const int col = i - row * lds_pitch;
                const int p   = row / in_rows;
                const int r   = row - p * in_rows;
                const int ih  = ih0 + r;
                const int iw  = iw0 + col;
                const int nc  = nc0 + p;
                if(nc < planes && static_cast<unsigned>(ih) < static_cast<unsigned>(in_h) &&
                   static_cast<unsigned>(iw) < static_cast<unsigned>(in_w))
                    v[k] = dw_to_float(input[nc * plane_in + ih * in_w + iw]);
            }
        }
#pragma unroll
        for(int k = 0; k < 8; ++k)
        {
            const int i = i0 + k * DW_BLOCK_SIZE;
            if(i < total_in)
                lds[i] = v[k];
        }
    }
    __syncthreads();

    const int strips = tile_w / DW_VO;
    const int per_p  = tile_h * strips;
    for(int it = tid; it < tile_planes * per_p; it += DW_BLOCK_SIZE)
    {
        const int p   = it / per_p;
        const int rem = it - p * per_p;
        const int r   = rem / strips;
        const int s   = rem - r * strips;
        const int nc  = nc0 + p;
        const int oh  = oh0 + r;
        const int ow  = ow0 + s * DW_VO;
        if(nc >= planes || oh >= out_h || ow >= out_w)
            continue;

        const float* wp   = lds_wei + p * (DW_FY * DW_FX);
        const float* base = lds + p * plane_lds + r * DW_SH * lds_pitch + s * (DW_VO * DW_SW);
        float acc[DW_VO];
#pragma unroll
        for(int v = 0; v < DW_VO; ++v)
            acc[v] = 0.0f;
#pragma unroll
        for(int fy = 0; fy < DW_FY; ++fy)
        {
            const float* rp = base + fy * DW_DH * lds_pitch;
            float x[dw_span4];
            if constexpr((DW_VO * DW_SW) % 4 == 0)
            {
                // 16-byte aligned: lds_pitch % 4 == 0 and the strip offset is a multiple of 4.
#pragma unroll
                for(int q = 0; q < dw_span4 / 4; ++q)
                {
                    const float4 t = reinterpret_cast<const float4*>(rp)[q];
                    x[4 * q]       = t.x;
                    x[4 * q + 1]   = t.y;
                    x[4 * q + 2]   = t.z;
                    x[4 * q + 3]   = t.w;
                }
            }
            else
            {
#pragma unroll
                for(int j = 0; j < dw_span; ++j)
                    x[j] = rp[j];
            }
#pragma unroll
            for(int fx = 0; fx < DW_FX; ++fx)
            {
                const float wv = wp[fy * DW_FX + fx];
#pragma unroll
                for(int v = 0; v < DW_VO; ++v)
                    acc[v] = __builtin_fmaf(x[v * DW_SW + fx * DW_DW], wv, acc[v]);
            }
        }
        dw_store(reinterpret_cast<unsigned short*>(output) +
                     nc * static_cast<dw_u64>(out_h) * out_w + oh * out_w + ow,
                 ow,
                 out_w,
                 acc);
    }
}

#else

// One thread computes DW_VO consecutive outputs of one output row, reading the input through the
// cache. Out-of-range taps are skipped, which is equivalent to adding 0 * w.
extern "C" __global__ void __launch_bounds__(DW_BLOCK_SIZE)
    miopen_conv2d_depthwise_fwd_generic(const IO_DTYPE* __restrict__ input,
                                        const IO_DTYPE* __restrict__ weights,
                                        IO_DTYPE* __restrict__ output,
                                        int channels,
                                        int planes,
                                        int in_h,
                                        int in_w,
                                        int out_h,
                                        int out_w,
                                        int pad_h,
                                        int pad_w,
                                        int strips)
{
    const dw_u64 gid   = static_cast<dw_u64>(blockIdx.x) * DW_BLOCK_SIZE + threadIdx.x;
    const dw_u64 total = static_cast<dw_u64>(planes) * out_h * strips;
    if(gid >= total)
        return;
    const int s    = static_cast<int>(gid % strips);
    const dw_u64 t = gid / strips;
    const int oh   = static_cast<int>(t % out_h);
    const int nc   = static_cast<int>(t / out_h);

    const IO_DTYPE* ip = input + nc * static_cast<dw_u64>(in_h) * in_w;
    // The channel count can reach INT32_MAX, so the filter base is formed in 64 bits.
    const IO_DTYPE* wp = weights + static_cast<dw_u64>(nc % channels) * (DW_FY * DW_FX);
    float wr[DW_FY * DW_FX];
#pragma unroll
    for(int i = 0; i < DW_FY * DW_FX; ++i)
        wr[i] = dw_to_float(wp[i]);

    float acc[DW_VO];
#pragma unroll
    for(int v = 0; v < DW_VO; ++v)
        acc[v] = 0.0f;
    const int ow  = s * DW_VO;
    const int iw0 = ow * DW_SW - pad_w;
    const int ih0 = oh * DW_SH - pad_h;
#pragma unroll
    for(int fy = 0; fy < DW_FY; ++fy)
    {
        const int ih = ih0 + fy * DW_DH;
        if(static_cast<unsigned>(ih) >= static_cast<unsigned>(in_h))
            continue;
        const IO_DTYPE* row = ip + ih * in_w;
        float x[dw_span];
#pragma unroll
        for(int j = 0; j < dw_span; ++j)
        {
            const int iw = iw0 + j;
            x[j] = static_cast<unsigned>(iw) < static_cast<unsigned>(in_w) ? dw_to_float(row[iw])
                                                                           : 0.0f;
        }
#pragma unroll
        for(int fx = 0; fx < DW_FX; ++fx)
        {
#pragma unroll
            for(int v = 0; v < DW_VO; ++v)
                acc[v] = __builtin_fmaf(x[v * DW_SW + fx * DW_DW], wr[fy * DW_FX + fx], acc[v]);
        }
    }
    dw_store(reinterpret_cast<unsigned short*>(output) + nc * static_cast<dw_u64>(out_h) * out_w +
                 oh * out_w + ow,
             ow,
             out_w,
             acc);
}

#endif
