// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Depthwise (channels-per-group == 1) forward convolution over NCHW tensors.
//
// The generic grouped implicit-GEMM path degenerates for this problem class:
// with one channel per group GemmN == 1 and GemmK == Fy*Fx, so nearly the whole
// MFMA/WMMA tile is padding and the cost tracks the group count instead of the
// FLOPs. This kernel keeps the per-channel taps in registers and slides a
// register window down the input rows, so an input row shared by consecutive
// output rows is read once rather than FILTER_H times. When STRIDE_H is not a
// whole number of dilation steps no row is shared, and the window is reloaded
// per output row (see kRowsReusable).
//
// Compile-time parameters are supplied by the solver: IO_DTYPE, FILTER_H/W,
// PAD_H/W, STRIDE_H/W, DILATION_H/W, OUT_PER_LANE, ROW_CHUNK, WORKGROUP_SIZE.

#ifndef IO_DTYPE
#define IO_DTYPE __half
#endif
#ifndef FILTER_H
#define FILTER_H 3
#endif
#ifndef FILTER_W
#define FILTER_W 3
#endif
#ifndef PAD_H
#define PAD_H 1
#endif
#ifndef PAD_W
#define PAD_W 1
#endif
#ifndef STRIDE_H
#define STRIDE_H 1
#endif
#ifndef STRIDE_W
#define STRIDE_W 1
#endif
#ifndef DILATION_H
#define DILATION_H 1
#endif
#ifndef DILATION_W
#define DILATION_W 1
#endif
#ifndef OUT_PER_LANE
#define OUT_PER_LANE 2
#endif
#ifndef ROW_CHUNK
#define ROW_CHUNK 8
#endif
#ifndef WORKGROUP_SIZE
#define WORKGROUP_SIZE 256
#endif

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#endif

namespace {

constexpr int kFilterH    = FILTER_H;
constexpr int kFilterW    = FILTER_W;
constexpr int kPadH       = PAD_H;
constexpr int kPadW       = PAD_W;
constexpr int kStrideH    = STRIDE_H;
constexpr int kStrideW    = STRIDE_W;
constexpr int kDilationH  = DILATION_H;
constexpr int kDilationW  = DILATION_W;
constexpr int kOutPerLane = OUT_PER_LANE;
constexpr int kRowChunk   = ROW_CHUNK;
constexpr int kWeightSize = kFilterH * kFilterW;

// Input columns a lane touches: its kOutPerLane outputs start kStrideW apart and
// each spans (kFilterW - 1) * kDilationW + 1 columns.
constexpr int kInSpanW = (kOutPerLane - 1) * kStrideW + (kFilterW - 1) * kDilationW + 1;

// Window slots, keyed by absolute input row modulo the row footprint of one
// output row. Rows within a footprint are kDilationH apart and span less than
// kWinSlots, so that mapping is collision-free.
constexpr int kWinSlots = (kFilterH - 1) * kDilationH + 1;

// A row of output row oh+1 is already resident iff it also belonged to output
// row oh, which needs kStrideH to be a whole number of dilation steps.
constexpr bool kRowsReusable = (kStrideH % kDilationH) == 0;
constexpr int kRowShift      = kRowsReusable ? kStrideH / kDilationH : kFilterH;

__device__ inline float to_float(__half v) { return __half2float(v); }
__device__ inline float to_float(__hip_bfloat16 v) { return __bfloat162float(v); }
__device__ inline float to_float(float v) { return v; }

__device__ inline void from_float(float v, __half& d) { d = __float2half(v); }
__device__ inline void from_float(float v, __hip_bfloat16& d) { d = __float2bfloat16(v); }
__device__ inline void from_float(float v, float& d) { d = v; }

} // namespace

extern "C" __global__ void __launch_bounds__(WORKGROUP_SIZE)
    miopen_conv2d_depthwise_fwd(const IO_DTYPE* __restrict__ input,   // [N, C, iH, iW]
                                const IO_DTYPE* __restrict__ weights, // [C, 1, FILTER_H, FILTER_W]
                                IO_DTYPE* __restrict__ output,        // [N, C, oH, oW]
                                int N,
                                int C,
                                int iH,
                                int iW,
                                int oH,
                                int oW)
{
    // Flat tile index over (n, c, row-block, column-block). Flattening instead
    // of mapping (n, c) onto gridDim.y keeps workgroups full for the small
    // spatial sizes typical of the deeper layers of a detection network.
    const int tiles_w   = (oW + kOutPerLane - 1) / kOutPerLane;
    const int row_tiles = (oH + kRowChunk - 1) / kRowChunk;
    const int per_image = tiles_w * row_tiles;
    // The tile count is a product of four tensor dimensions, and the solver's
    // AllTensorsLengthsFitIntoInt() check only bounds them individually, so the
    // product is carried in 64 bits. The derived indices below stay 32-bit.
    const long long total = static_cast<long long>(per_image) * N * C;
    const long long step  = static_cast<long long>(gridDim.x) * WORKGROUP_SIZE;

    for(long long tile = static_cast<long long>(blockIdx.x) * WORKGROUP_SIZE + threadIdx.x;
        tile < total;
        tile += step)
    {
        const int nc  = static_cast<int>(tile / per_image);
        const int t   = static_cast<int>(tile - static_cast<long long>(nc) * per_image);
        const int c   = nc % C;
        const int rt  = t / tiles_w;
        const int ow0 = (t - rt * tiles_w) * kOutPerLane;
        const int oh0 = rt * kRowChunk;
        const int iwb = ow0 * kStrideW - kPadW;

        float taps[kFilterH][kFilterW];
        const IO_DTYPE* wptr = weights + static_cast<size_t>(c) * kWeightSize;
#pragma unroll
        for(int fh = 0; fh < kFilterH; ++fh)
        {
#pragma unroll
            for(int fw = 0; fw < kFilterW; ++fw)
                taps[fh][fw] = to_float(wptr[fh * kFilterW + fw]);
        }

        const size_t image_in  = static_cast<size_t>(nc) * iH * iW;
        const size_t image_out = static_cast<size_t>(nc) * oH * oW;

        // The whole column span is interior, so the row loads need no
        // per-element bounds test. This says nothing about the output columns:
        // the stores below are guarded individually, so a partial output tile
        // still takes the fast load path.
        const bool wide = (iwb >= 0) && (iwb + kInSpanW <= iW);

        float win[kWinSlots][kInSpanW];

        auto load_row = [&](int slot, int ih) {
#pragma unroll
            for(int s = 0; s < kInSpanW; ++s)
                win[slot][s] = 0.0f;
            if(ih < 0 || ih >= iH)
                return;
            const IO_DTYPE* row = input + image_in + static_cast<size_t>(ih) * iW;
            if(wide)
            {
#pragma unroll
                for(int s = 0; s < kInSpanW; ++s)
                    win[slot][s] = to_float(row[iwb + s]);
            }
            else
            {
#pragma unroll
                for(int s = 0; s < kInSpanW; ++s)
                {
                    const int iw = iwb + s;
                    if(iw >= 0 && iw < iW)
                        win[slot][s] = to_float(row[iw]);
                }
            }
        };

        // Slots are keyed by row offset relative to the row-block base, so both
        // the producer and the consumer index the window with constants.
        const int ih0 = oh0 * kStrideH - kPadH;
#pragma unroll
        for(int fh = 0; fh < kFilterH; ++fh)
            load_row((fh * kDilationH) % kWinSlots, ih0 + fh * kDilationH);

#pragma unroll
        for(int r = 0; r < kRowChunk; ++r)
        {
            const int oh = oh0 + r;
            if(oh >= oH)
                break;

            float acc[kOutPerLane];
#pragma unroll
            for(int o = 0; o < kOutPerLane; ++o)
                acc[o] = 0.0f;

            const int base = r * kStrideH;
#pragma unroll
            for(int fh = 0; fh < kFilterH; ++fh)
            {
                const int slot = (base + fh * kDilationH) % kWinSlots;
#pragma unroll
                for(int o = 0; o < kOutPerLane; ++o)
                {
#pragma unroll
                    for(int fw = 0; fw < kFilterW; ++fw)
                        acc[o] =
                            fmaf(taps[fh][fw], win[slot][o * kStrideW + fw * kDilationW], acc[o]);
                }
            }

            IO_DTYPE* out = output + image_out + static_cast<size_t>(oh) * oW + ow0;
#pragma unroll
            for(int o = 0; o < kOutPerLane; ++o)
            {
                if(ow0 + o < oW)
                    from_float(acc[o], out[o]);
            }

            // Fetch the rows the next output row gains. Row fh of that row is
            // already resident as row fh + kRowShift of this one when kStrideH
            // is a whole number of dilation steps; otherwise nothing survives
            // and the window is fully reloaded. Skip it on the last row of the
            // chunk and on the last output row, where no iteration follows to
            // read what was fetched.
            if(r + 1 < kRowChunk && oh + 1 < oH)
            {
                const int next = base + kStrideH;
#pragma unroll
                for(int fh = 0; fh < kFilterH; ++fh)
                {
                    if(kRowsReusable && fh < kFilterH - kRowShift)
                        continue;
                    load_row((next + fh * kDilationH) % kWinSlots, ih0 + next + fh * kDilationH);
                }
            }
        }
    }
}
