// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The VALU depthwise backward-weights kernels: a partial kernel and a fold kernel.
//
// The schedule of hipconv's Hankel depthwise wgrad, reimplemented on plain vector ALU: the
// reduction over images and output rows is cut into partitions; each block of the partial
// kernel sums one partition for a block of channels, in fp32, and writes the sums to a
// workspace; the fold kernel adds the partitions in a fixed order and rounds once into dw. No
// atomics, so the result is the same on every run. FMA in fp32 with no wave-size intrinsics, so
// the kernels run on wave32 and wave64 devices alike.
//
// One HIPRTC compile per kernel config; every shape is a runtime argument
// (ConvDepthwiseValuWrwArgs.h).
//   CONV_DW_TYPE     element type of x, dy and dw: float, _Float16 or __bf16
//   CONV_DW_LAYOUT   0 = channels-last (NHWC / NDHWC), 1 = channels-first (NCHW / NCDHW)
//   CONV_DW_VARIANT  0 = floor: any filter, stride, dilation, padding, 1-D to 3-D
//                    1 = register tile: compile-time CONV_DW_KH x CONV_DW_KW (2-D or 1-D),
//                        every tap's sum in registers, dy read once per position
//   CONV_DW_FLIP     1 reflects the filter (true convolution)

#include "ConvDepthwiseValuWrwArgs.h"

#ifndef CONV_DW_TYPE
#error "CONV_DW_TYPE must name the element type"
#endif
#ifndef CONV_DW_LAYOUT
#define CONV_DW_LAYOUT 0
#endif
#ifndef CONV_DW_VARIANT
#define CONV_DW_VARIANT 0
#endif
#ifndef CONV_DW_FLIP
#define CONV_DW_FLIP 0
#endif
#ifndef CONV_DW_KH
#define CONV_DW_KH 3
#endif
#ifndef CONV_DW_KW
#define CONV_DW_KW 3
#endif

using ElementType = CONV_DW_TYPE;

namespace
{

constexpr int BLOCK = 256;
#if CONV_DW_LAYOUT == 0
/// Channels-last: channels per block (consecutive lanes, so loads coalesce) and lanes that
/// split the partition's rows (floor: the taps) between them.
constexpr int CHANNELS = 64;
constexpr int LANES = BLOCK / CHANNELS;
#endif

__device__ inline float toFloat(ElementType v)
{
    return static_cast<float>(v);
}

/// Element offset of (n, c, z, y, x) in a packed tensor of c channels and extents (d, h, w).
__device__ inline long
    offsetOf(int n, int c, int z, int y, int x, int channels, int d, int h, int w)
{
#if CONV_DW_LAYOUT == 0
    return ((((long)n * d + z) * h + y) * w + x) * channels + c;
#else
    return ((((long)n * channels + c) * d + z) * h + y) * w + x;
#endif
}

/// The partition's rows [first, last) of n * do * ho.
__device__ inline void
    partitionRows(const ConvDepthwiseValuWrwArgs& a, int partition, int& first, int& last)
{
    first = partition * a.rowsPerPartition;
    last = min(a.rows, first + a.rowsPerPartition);
}

#if CONV_DW_LAYOUT == 1
/// Sum over 256 threads through LDS; thread 0 gets the total.
__device__ inline float blockSum(float value, float* scratch)
{
    const int tid = static_cast<int>(threadIdx.x);
    scratch[tid] = value;
    __syncthreads();
    for(int half = BLOCK / 2; half > 0; half /= 2)
    {
        if(tid < half)
        {
            scratch[tid] += scratch[tid + half];
        }
        __syncthreads();
    }
    const float total = scratch[0];
    __syncthreads();
    return total;
}
#endif

} // namespace

extern "C" __global__ void __launch_bounds__(BLOCK)
    ConvDepthwiseValuWrwPartial(ConvDepthwiseValuWrwArgs a)
{
    const auto* x = static_cast<const ElementType*>(a.x);
    const auto* dy = static_cast<const ElementType*>(a.dy);
    const int tid = static_cast<int>(threadIdx.x);

#if CONV_DW_LAYOUT == 0
    // Grid: partitions x channel blocks. Lane `lane` of a channel takes taps (floor) or rows
    // (tile) lane, lane + LANES, ...
    const int chunks = (a.c + CHANNELS - 1) / CHANNELS;
    const int partition = static_cast<int>(blockIdx.x) / chunks;
    const int c = (static_cast<int>(blockIdx.x) % chunks) * CHANNELS + tid % CHANNELS;
    const int lane = tid / CHANNELS;
    int first = 0;
    int last = 0;
    partitionRows(a, partition, first, last);
#if CONV_DW_VARIANT == 0
    for(int tap = lane; tap < a.taps; tap += LANES)
    {
        const int kx = tap % a.kw;
        const int ky = (tap / a.kw) % a.kh;
        const int kz = tap / (a.kw * a.kh);
        float sum = 0.0f;
        if(c < a.c)
        {
            for(int row = first; row < last; ++row)
            {
                const int oy = row % a.ho;
                const int oz = (row / a.ho) % a.dOut;
                const int n = row / (a.ho * a.dOut);
                const int iz = oz * a.sd - a.pd + kz * a.dd;
                const int iy = oy * a.sh - a.ph + ky * a.dh;
                if(iz < 0 || iz >= a.di || iy < 0 || iy >= a.hi)
                {
                    continue;
                }
                for(int ox = 0; ox < a.wo; ++ox)
                {
                    const int ix = ox * a.sw - a.pw + kx * a.dw_;
                    if(ix < 0 || ix >= a.wi)
                    {
                        continue;
                    }
                    sum += toFloat(dy[offsetOf(n, c, oz, oy, ox, a.c, a.dOut, a.ho, a.wo)])
                           * toFloat(x[offsetOf(n, c, iz, iy, ix, a.c, a.di, a.hi, a.wi)]);
                }
            }
            a.partial[((long)partition * a.c + c) * a.taps + tap] = sum;
        }
    }
#else
    // Register tile: every tap's sum in registers; the lanes split the rows, then add up in LDS.
    constexpr int TAPS = CONV_DW_KH * CONV_DW_KW;
    float sum[TAPS];
#pragma unroll
    for(int t = 0; t < TAPS; ++t)
    {
        sum[t] = 0.0f;
    }
    if(c < a.c)
    {
        for(int row = first + lane; row < last; row += LANES)
        {
            const int oy = row % a.ho;
            const int n = row / a.ho;
            for(int ox = 0; ox < a.wo; ++ox)
            {
                const float g = toFloat(dy[offsetOf(n, c, 0, oy, ox, a.c, 1, a.ho, a.wo)]);
#pragma unroll
                for(int ky = 0; ky < CONV_DW_KH; ++ky)
                {
                    const int iy = oy * a.sh - a.ph + ky * a.dh;
                    const bool rowInside = iy >= 0 && iy < a.hi;
#pragma unroll
                    for(int kx = 0; kx < CONV_DW_KW; ++kx)
                    {
                        const int ix = ox * a.sw - a.pw + kx * a.dw_;
                        if(rowInside && ix >= 0 && ix < a.wi)
                        {
                            sum[ky * CONV_DW_KW + kx]
                                += g * toFloat(x[offsetOf(n, c, 0, iy, ix, a.c, 1, a.hi, a.wi)]);
                        }
                    }
                }
            }
        }
    }
    __shared__ float lanes[LANES][CHANNELS];
#pragma unroll
    for(int t = 0; t < TAPS; ++t)
    {
        lanes[lane][tid % CHANNELS] = sum[t];
        __syncthreads();
        if(lane == 0 && c < a.c)
        {
            float total = 0.0f;
            for(int l = 0; l < LANES; ++l)
            {
                total += lanes[l][tid % CHANNELS];
            }
            a.partial[((long)partition * a.c + c) * TAPS + t] = total;
        }
        __syncthreads();
    }
#endif
#else
    // Channels-first. Grid: partitions x channels; the block's threads stride over the
    // partition's positions (width fastest, so loads coalesce) and add up per tap in LDS.
    __shared__ float scratch[BLOCK];
    const int partition = static_cast<int>(blockIdx.x) / a.c;
    const int c = static_cast<int>(blockIdx.x) % a.c;
    int first = 0;
    int last = 0;
    partitionRows(a, partition, first, last);
    const int positions = (last - first) * a.wo;
#if CONV_DW_VARIANT == 0
    for(int tap = 0; tap < a.taps; ++tap)
    {
        const int kx = tap % a.kw;
        const int ky = (tap / a.kw) % a.kh;
        const int kz = tap / (a.kw * a.kh);
        float sum = 0.0f;
        for(int p = tid; p < positions; p += BLOCK)
        {
            const int row = first + p / a.wo;
            const int ox = p % a.wo;
            const int oy = row % a.ho;
            const int oz = (row / a.ho) % a.dOut;
            const int n = row / (a.ho * a.dOut);
            const int iz = oz * a.sd - a.pd + kz * a.dd;
            const int iy = oy * a.sh - a.ph + ky * a.dh;
            const int ix = ox * a.sw - a.pw + kx * a.dw_;
            if(iz >= 0 && iz < a.di && iy >= 0 && iy < a.hi && ix >= 0 && ix < a.wi)
            {
                sum += toFloat(dy[offsetOf(n, c, oz, oy, ox, a.c, a.dOut, a.ho, a.wo)])
                       * toFloat(x[offsetOf(n, c, iz, iy, ix, a.c, a.di, a.hi, a.wi)]);
            }
        }
        const float total = blockSum(sum, scratch);
        if(tid == 0)
        {
            a.partial[((long)partition * a.c + c) * a.taps + tap] = total;
        }
    }
#else
    constexpr int TAPS = CONV_DW_KH * CONV_DW_KW;
    float sum[TAPS];
#pragma unroll
    for(int t = 0; t < TAPS; ++t)
    {
        sum[t] = 0.0f;
    }
    for(int p = tid; p < positions; p += BLOCK)
    {
        const int row = first + p / a.wo;
        const int ox = p % a.wo;
        const int oy = row % a.ho;
        const int n = row / a.ho;
        const float g = toFloat(dy[offsetOf(n, c, 0, oy, ox, a.c, 1, a.ho, a.wo)]);
#pragma unroll
        for(int ky = 0; ky < CONV_DW_KH; ++ky)
        {
            const int iy = oy * a.sh - a.ph + ky * a.dh;
            const bool rowInside = iy >= 0 && iy < a.hi;
#pragma unroll
            for(int kx = 0; kx < CONV_DW_KW; ++kx)
            {
                const int ix = ox * a.sw - a.pw + kx * a.dw_;
                if(rowInside && ix >= 0 && ix < a.wi)
                {
                    sum[ky * CONV_DW_KW + kx]
                        += g * toFloat(x[offsetOf(n, c, 0, iy, ix, a.c, 1, a.hi, a.wi)]);
                }
            }
        }
    }
#pragma unroll
    for(int t = 0; t < TAPS; ++t)
    {
        const float total = blockSum(sum[t], scratch);
        if(tid == 0)
        {
            a.partial[((long)partition * a.c + c) * TAPS + t] = total;
        }
    }
#endif
#endif
}

/// dw[c, tap] = the partitions' sums in partition order, rounded once; reflected for true
/// convolution, whose filter gradient is cross-correlation's reversed.
extern "C" __global__ void __launch_bounds__(BLOCK)
    ConvDepthwiseValuWrwFold(ConvDepthwiseValuWrwArgs a)
{
    const long index = (long)blockIdx.x * BLOCK + threadIdx.x;
    const long count = (long)a.c * a.taps;
    if(index >= count)
    {
        return;
    }
    float total = 0.0f;
    for(int p = 0; p < a.partitions; ++p)
    {
        total += a.partial[(long)p * count + index];
    }
    const int tap = static_cast<int>(index % a.taps);
    const long c = index / a.taps;
    const int target = CONV_DW_FLIP != 0 ? a.taps - 1 - tap : tap;
    static_cast<ElementType*>(a.dw)[c * a.taps + target] = static_cast<ElementType>(total);
}
