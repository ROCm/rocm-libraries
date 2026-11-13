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
//                    1 = register tile: compile-time CONV_DW_KD x CONV_DW_KH x CONV_DW_KW (KD 1
//                        for 1-D and 2-D), every tap's sum in registers, dy read once per
//                        position
//                    2 = matrix core (ConvDepthwiseHankelWrw.h): 16-bit channels-last, unit
//                        stride and dilation, compile-time CONV_DW_KH x CONV_DW_KW, on 4x4x4
//                        MFMA; CONV_DW_Q columns a tile, CONV_DW_G rows a step, CONV_DW_NWG
//                        16-channel groups a wave
//   CONV_DW_FLIP     1 reflects the filter (true convolution)
//   CONV_DW_MULT     depth multiplier (default 1): mult output channels per input channel, output
//                    channel k reading input channel k / mult (variants 0 and 1)
//
// The kernels walk output channels c (a.c x mult of them): dy and dw by c, x by c / mult.

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
#ifndef CONV_DW_KD
#define CONV_DW_KD 1
#endif
#ifndef CONV_DW_KH
#define CONV_DW_KH 3
#endif
#ifndef CONV_DW_KW
#define CONV_DW_KW 3
#endif
#ifndef CONV_DW_MULT
#define CONV_DW_MULT 1
#endif

using ElementType = CONV_DW_TYPE;

namespace
{

constexpr int BLOCK = 256;
constexpr int MULT = CONV_DW_MULT;
static_assert(MULT >= 1 && (MULT == 1 || CONV_DW_VARIANT != 2), "depth multiplier");

/// Output channels (dy's and dw's): the input channels times the multiplier.
__device__ inline int outChannels(const ConvDepthwiseValuWrwArgs& a)
{
    return a.c * MULT;
}
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

#if CONV_DW_VARIANT == 1
constexpr int TAPS = CONV_DW_KD * CONV_DW_KH * CONV_DW_KW;

/// Adds dy at output position (oz, oy, ox) times the x element under every tap of the
/// compile-time filter to that tap's sum.
__device__ inline void accumulateTaps(float (&sum)[TAPS],
                                      const ElementType* x,
                                      float g,
                                      int n,
                                      int c,
                                      int oz,
                                      int oy,
                                      int ox,
                                      const ConvDepthwiseValuWrwArgs& a)
{
#pragma unroll
    for(int kz = 0; kz < CONV_DW_KD; ++kz)
    {
        const int iz = oz * a.sd - a.pd + kz * a.dd;
        const bool planeInside = iz >= 0 && iz < a.di;
#pragma unroll
        for(int ky = 0; ky < CONV_DW_KH; ++ky)
        {
            const int iy = oy * a.sh - a.ph + ky * a.dh;
            const bool rowInside = planeInside && iy >= 0 && iy < a.hi;
#pragma unroll
            for(int kx = 0; kx < CONV_DW_KW; ++kx)
            {
                const int ix = ox * a.sw - a.pw + kx * a.dw_;
                if(rowInside && ix >= 0 && ix < a.wi)
                {
                    sum[(kz * CONV_DW_KH + ky) * CONV_DW_KW + kx]
                        += g * toFloat(x[offsetOf(n, c / MULT, iz, iy, ix, a.c, a.di, a.hi, a.wi)]);
                }
            }
        }
    }
}

/// Output columns per register strip of the unit-stride, undilated width path.
#ifndef CONV_DW_STRIP
#define CONV_DW_STRIP 8
#endif
constexpr int STRIP = CONV_DW_STRIP;

/// Output row `row` (n, oz, oy) of channel c at width stride 1 and dilation 1, columns from
/// oxBegin to the row's end in strips of STRIP columns, oxStep apart: the strip's dy values
/// and, per filter row, the STRIP + KW - 1 x values under it sit in registers, so each x
/// element is read once per filter row (not once per tap) and offsets are formed per row, not
/// per tap.
__device__ inline void accumulateStrips(float (&sum)[TAPS],
                                        const ElementType* x,
                                        const ElementType* dy,
                                        int row,
                                        int c,
                                        int oxBegin,
                                        int oxStep,
                                        const ConvDepthwiseValuWrwArgs& a)
{
    constexpr int SPAN = STRIP + CONV_DW_KW - 1;
    const int oy = row % a.ho;
    const int oz = (row / a.ho) % a.dOut;
    const int n = row / (a.ho * a.dOut);
    // Element strides of a column of x and of dy.
    const long cs = CONV_DW_LAYOUT == 0 ? a.c : 1;
    const long csy = CONV_DW_LAYOUT == 0 ? outChannels(a) : 1;
    const ElementType* dyRow = dy + offsetOf(n, c, oz, oy, 0, outChannels(a), a.dOut, a.ho, a.wo);
    // Filter rows (kz, ky) outside x read a clamped row and count zero: no branches, so every
    // load of a strip issues before its first FMA.
    constexpr int FROWS = CONV_DW_KD * CONV_DW_KH;
    const ElementType* xRows[FROWS];
    bool rowInside[FROWS];
#pragma unroll
    for(int kz = 0; kz < CONV_DW_KD; ++kz)
    {
#pragma unroll
        for(int ky = 0; ky < CONV_DW_KH; ++ky)
        {
            const int iz = oz * a.sd - a.pd + kz * a.dd;
            const int iy = oy * a.sh - a.ph + ky * a.dh;
            rowInside[kz * CONV_DW_KH + ky] = iz >= 0 && iz < a.di && iy >= 0 && iy < a.hi;
            xRows[kz * CONV_DW_KH + ky] = x
                                          + offsetOf(n,
                                                     c / MULT,
                                                     min(max(iz, 0), a.di - 1),
                                                     min(max(iy, 0), a.hi - 1),
                                                     0,
                                                     a.c,
                                                     a.di,
                                                     a.hi,
                                                     a.wi);
        }
    }
    for(int ox0 = oxBegin; ox0 < a.wo; ox0 += oxStep)
    {
        float g[STRIP];
#pragma unroll
        for(int t = 0; t < STRIP; ++t)
        {
            const int ox = min(ox0 + t, a.wo - 1);
            const float value = toFloat(dyRow[ox * csy]);
            g[t] = ox0 + t < a.wo ? value : 0.0f;
        }
        const int ix0 = ox0 - a.pw;
        float v[FROWS][SPAN];
#pragma unroll
        for(int r = 0; r < FROWS; ++r)
        {
#pragma unroll
            for(int j = 0; j < SPAN; ++j)
            {
                const int ix = ix0 + j;
                const float value = toFloat(xRows[r][min(max(ix, 0), a.wi - 1) * cs]);
                v[r][j] = rowInside[r] && ix >= 0 && ix < a.wi ? value : 0.0f;
            }
        }
#pragma unroll
        for(int r = 0; r < FROWS; ++r)
        {
#pragma unroll
            for(int kx = 0; kx < CONV_DW_KW; ++kx)
            {
#pragma unroll
                for(int t = 0; t < STRIP; ++t)
                {
                    sum[r * CONV_DW_KW + kx] += g[t] * v[r][t + kx];
                }
            }
        }
    }
}
#endif

} // namespace

#if CONV_DW_VARIANT == 2
#include "ConvDepthwiseHankelWrw.h"
#endif

extern "C" __global__ void __launch_bounds__(BLOCK)
    ConvDepthwiseValuWrwPartial(ConvDepthwiseValuWrwArgs a)
{
#if CONV_DW_VARIANT == 2
    static_assert(hankel::THREADS == BLOCK, "one block size");
    hankel::partial<__is_same(ElementType, __bf16)>(a);
#else
    const auto* x = static_cast<const ElementType*>(a.x);
    const auto* dy = static_cast<const ElementType*>(a.dy);
    const int tid = static_cast<int>(threadIdx.x);

#if CONV_DW_LAYOUT == 0
    // Grid: partitions x channel blocks. Lane `lane` of a channel takes taps (floor) or rows
    // (tile) lane, lane + LANES, ...
    const int channels = outChannels(a);
    const int chunks = (channels + CHANNELS - 1) / CHANNELS;
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
        if(c < channels)
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
                    sum += toFloat(
                               dy[offsetOf(n, c, oz, oy, ox, outChannels(a), a.dOut, a.ho, a.wo)])
                           * toFloat(x[offsetOf(n, c / MULT, iz, iy, ix, a.c, a.di, a.hi, a.wi)]);
                }
            }
            a.partial[((long)partition * channels + c) * a.taps + tap] = sum;
        }
    }
#else
    // Register tile: every tap's sum in registers; the lanes split the rows, then add up in LDS.
    float sum[TAPS];
#pragma unroll
    for(int t = 0; t < TAPS; ++t)
    {
        sum[t] = 0.0f;
    }
    if(c < channels && a.sw == 1 && a.dw_ == 1)
    {
        for(int row = first + lane; row < last; row += LANES)
        {
            accumulateStrips(sum, x, dy, row, c, 0, STRIP, a);
        }
    }
    else if(c < channels)
    {
        for(int row = first + lane; row < last; row += LANES)
        {
            const int oy = row % a.ho;
            const int oz = (row / a.ho) % a.dOut;
            const int n = row / (a.ho * a.dOut);
            for(int ox = 0; ox < a.wo; ++ox)
            {
                const float g
                    = toFloat(dy[offsetOf(n, c, oz, oy, ox, outChannels(a), a.dOut, a.ho, a.wo)]);
                accumulateTaps(sum, x, g, n, c, oz, oy, ox, a);
            }
        }
    }
    __shared__ float lanes[LANES][CHANNELS];
#pragma unroll
    for(int t = 0; t < TAPS; ++t)
    {
        lanes[lane][tid % CHANNELS] = sum[t];
        __syncthreads();
        if(lane == 0 && c < channels)
        {
            float total = 0.0f;
            for(int l = 0; l < LANES; ++l)
            {
                total += lanes[l][tid % CHANNELS];
            }
            a.partial[((long)partition * channels + c) * TAPS + t] = total;
        }
        __syncthreads();
    }
#endif
#else
    // Channels-first. Grid: partitions x channels; the block's threads stride over the
    // partition's positions (width fastest, so loads coalesce) and add up per tap in LDS.
    __shared__ float scratch[BLOCK];
    const int channels = outChannels(a);
    const int partition = static_cast<int>(blockIdx.x) / channels;
    const int c = static_cast<int>(blockIdx.x) % channels;
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
                sum += toFloat(dy[offsetOf(n, c, oz, oy, ox, outChannels(a), a.dOut, a.ho, a.wo)])
                       * toFloat(x[offsetOf(n, c / MULT, iz, iy, ix, a.c, a.di, a.hi, a.wi)]);
            }
        }
        const float total = blockSum(sum, scratch);
        if(tid == 0)
        {
            a.partial[((long)partition * channels + c) * a.taps + tap] = total;
        }
    }
#else
    float sum[TAPS];
#pragma unroll
    for(int t = 0; t < TAPS; ++t)
    {
        sum[t] = 0.0f;
    }
    if(a.sw == 1 && a.dw_ == 1)
    {
        // A thread takes strips of STRIP columns (accumulateStrips), consecutive threads
        // consecutive strips.
        const int strips = (a.wo + STRIP - 1) / STRIP;
        for(int p = tid; p < (last - first) * strips; p += BLOCK)
        {
            accumulateStrips(sum, x, dy, first + p / strips, c, (p % strips) * STRIP, a.wo, a);
        }
    }
    else
    {
        for(int p = tid; p < positions; p += BLOCK)
        {
            const int row = first + p / a.wo;
            const int ox = p % a.wo;
            const int oy = row % a.ho;
            const int oz = (row / a.ho) % a.dOut;
            const int n = row / (a.ho * a.dOut);
            const float g
                = toFloat(dy[offsetOf(n, c, oz, oy, ox, outChannels(a), a.dOut, a.ho, a.wo)]);
            accumulateTaps(sum, x, g, n, c, oz, oy, ox, a);
        }
    }
#pragma unroll
    for(int t = 0; t < TAPS; ++t)
    {
        const float total = blockSum(sum[t], scratch);
        if(tid == 0)
        {
            a.partial[((long)partition * channels + c) * TAPS + t] = total;
        }
    }
#endif
#endif
#endif // CONV_DW_VARIANT == 2
}

/// dw[c, tap] = the partitions' sums in partition order, rounded once; reflected for true
/// convolution, whose filter gradient is cross-correlation's reversed.
extern "C" __global__ void __launch_bounds__(BLOCK)
    ConvDepthwiseValuWrwFold(ConvDepthwiseValuWrwArgs a)
{
    const long index = (long)blockIdx.x * BLOCK + threadIdx.x;
    const long count = (long)a.c * MULT * a.taps; // output channels x taps
    if(index >= count)
    {
        return;
    }
    // Partition order; 16 loads in flight a thread (one at a time left a fold of a few
    // hundred partitions waiting on memory latency per partition).
    float total = 0.0f;
    int p = 0;
    for(; p + 16 <= a.partitions; p += 16)
    {
        float v[16];
#pragma unroll
        for(int j = 0; j < 16; ++j)
        {
            v[j] = a.partial[(long)(p + j) * count + index];
        }
#pragma unroll
        for(int j = 0; j < 16; ++j)
        {
            total += v[j];
        }
    }
    for(; p < a.partitions; ++p)
    {
        total += a.partial[(long)p * count + index];
    }
    const int tap = static_cast<int>(index % a.taps);
    const long c = index / a.taps;
    const int target = CONV_DW_FLIP != 0 ? a.taps - 1 - tap : tap;
    static_cast<ElementType*>(a.dw)[c * a.taps + target] = static_cast<ElementType>(total);
}
