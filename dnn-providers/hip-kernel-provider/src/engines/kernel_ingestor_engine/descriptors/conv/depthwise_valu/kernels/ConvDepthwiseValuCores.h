// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// VALU depthwise convolution cores for the CONV_DEPTHWISE_VALU engines.
//
// The hand-written hipconv VALU depthwise kernels, as harvested into MIOpen's
// ConvDepthwiseDirect solver (rocm-libraries PR #10945, branch
// users/brpepers/conv-depthwise-direct). The compute is that validated code; the changes for
// this home are the removed hip_runtime.h include (HIPRTC provides the builtins) and the filter
// reads going through dwTap2 / dwTap3, so CONV_DW_FLIP can reflect the filter: true
// convolution, and backward data computed as the forward convolution of dy.
//
// Plain vector-ALU FMA with an fp32 accumulator: no wave-size intrinsics, so the cores run on
// wave32 and wave64 devices alike; each config's speed varies by device and is measured, not
// assumed. K == C (depthwise), weights W[C,(KD,)KH,KW] contiguous.
//
// Code generation (from the HIP / rocKE / FlyDSL kernel comparison):
//   - Per-lane load predicates are branchless: an out-of-range tap or halo element loads an
//     in-range address and its value is selected away, so the loads of a group stay in flight
//     together; a branch per load waits for each one alone. The LDS staging loops run a
//     compile-time trip count (the block size is the config's), fully unrolled, so a thread's
//     halo loads issue back to back.
//   - Index math is 32-bit when the tensors (the one-thread-per-output floors) or the images
//     (the block-staged cores, with a 64-bit image base) have fewer than 2^31 elements, with
//     the thread decode by unsigned magic division; larger tensors take the same code in 64-bit.
//   - The staging barrier fences only LDS on gfx12 (dwLdsBarrier).
//
//   v2_wstrip_core_nhwc     generic W-strip floor; any stride/dilation/kernel size.
//   v3a_microtile_core_nhwc register 2D halo reuse (no LDS); s=d=1, compile-time KH/KW.
//   v4_fused_core_nhwc      LDS block stage + register micro-tile (two-level blocking).
//   v3b_lds_core_nhwc       LDS block stage + tap loop from LDS (register-light; big KH).
//   v2_wstrip_core_ndhwc    3D channel-last floor;   v3b_lds_core_ndhwc 3D LDS stage.
//   v2_core_nchw / v2_core_ncdhw  channel-first floors, coalesced on the contiguous width.
//   v3b_lds_core_nchw       channel-first LDS plane-tile stage + register micro-tile (2D).
//   v5_gather_bwd           backward data as a gather over the taps; any stride, both layouts.
//   v6_phase_block_nhwc/nchw  backward data by stride phase blocks: each thread computes the
//                           S x S dx pixels of one dy position from the dy window they share
//                           (compile-time filter and stride, unit dilation), both layouts.
#pragma once

#include "ConvArchFamily.h"

namespace conv_depthwise_valu
{

// dtype <-> fp32 conversion. __half and __hip_bfloat16 (MIOpen's -DIO_DTYPE
// choices) and float all convert to/from float natively in HIP via static_cast.
template <typename T>
__device__ inline float valu_to_f32(T v)
{
    return static_cast<float>(v);
}
template <typename T>
__device__ inline void valu_from_f32(float s, T& d)
{
    d = static_cast<T>(s);
}

#ifndef CONV_DW_FLIP
#define CONV_DW_FLIP 0
#endif

// The filter element filter tap (ky, kx) multiplies: itself, or its reflection when the filter
// is flipped (true convolution, or backward data run as a forward convolution of dy).
__device__ inline int dwTap2(int ky, int kx, int kh, int kw)
{
    return CONV_DW_FLIP != 0 ? (kh - 1 - ky) * kw + (kw - 1 - kx) : ky * kw + kx;
}

__device__ inline int dwTap3(int kz, int ky, int kx, int kd, int kh, int kw)
{
    return CONV_DW_FLIP != 0 ? ((kd - 1 - kz) * kh + (kh - 1 - ky)) * kw + (kw - 1 - kx)
                             : (kz * kh + ky) * kw + kx;
}

// Runtime geometry passed to every core. K == C for depthwise.
// 2D cores (NHWC) read only the first two rows; the depth fields are populated
// and read only by the 3D floor (v2_wstrip_core_ndhwc, NDHWC). Depth defaults of
// Di=Do=1, kd=1, pd=0, sd=1, dd=1 make the 3D core degenerate to the 2D result.
struct ValuParams
{
    int N, C, Hi, Wi, Ho, Wo;
    int kh, kw, ph, pw, sh, sw, dh, dw;
    int Di, Do, kd, pd, sd, dd; // depth axis (3D only)
};

// The barrier between the LDS stage and its reads. The waves exchange only LDS, so on gfx12
// the fences cover the local address space alone: __syncthreads() fences every address space at
// workgroup scope, which adds a global_inv (L0 invalidate) after the barrier there. Other
// families keep __syncthreads().
__device__ inline void dwLdsBarrier()
{
#if HKP_ARCH_GFX12
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    __syncthreads();
#endif
}

/// a < bound for a signed a and a non-negative bound, as one unsigned compare.
__device__ inline bool dwIn(int a, int bound)
{
    return static_cast<unsigned>(a) < static_cast<unsigned>(bound);
}

/// Division by a uniform runtime divisor. Index type int: unsigned magic division (multiplier
/// and shift from the divisor, once per thread; exact for a dividend and divisor in [0, 2^31)).
/// Index type long: the plain 64-bit division.
template <typename I>
struct DwDivider
{
    I d;
    __device__ explicit DwDivider(int divisor)
        : d(divisor)
    {
    }
    __device__ I div(I n) const
    {
        return n / d;
    }
};
template <>
struct DwDivider<int>
{
    int d;
    unsigned mul, sh;
    __device__ explicit DwDivider(int divisor)
        : d(divisor)
    {
        const unsigned ud = static_cast<unsigned>(divisor);
        sh = ud > 1 ? 32u - static_cast<unsigned>(__builtin_clz(ud - 1)) : 0u;
        mul = static_cast<unsigned>(((((1ull << sh) - ud) << 32) / ud) + 1);
    }
    __device__ int div(int n) const
    {
        const unsigned un = static_cast<unsigned>(n);
        const unsigned hi
            = static_cast<unsigned>((static_cast<unsigned long long>(un) * mul) >> 32);
        return static_cast<int>((hi + un) >> sh);
    }
};

/// Whole-tensor 32-bit indexing: x, y and the launched threads below 2^31.
__device__ inline bool dwTensorsFit32(const ValuParams& p)
{
    const long lim = 0x7fffffffL;
    return (long)p.N * p.C * p.Di * p.Hi * p.Wi < lim && (long)p.N * p.C * p.Do * p.Ho * p.Wo < lim
           && (long)gridDim.x * blockDim.x < lim;
}

/// Per-image 32-bit indexing (64-bit image base): one image of x and of y below 2^31 elements.
__device__ inline bool dwImagesFit32(const ValuParams& p)
{
    const long lim = 0x7fffffffL;
    return (long)p.C * p.Di * p.Hi * p.Wi < lim && (long)p.C * p.Do * p.Ho * p.Wo < lim;
}

// ---- v2 — register W-strip: WSTRIP output columns per thread; k innermost ----
// Handles any stride / dilation / padding / kernel size. The universal floor.
template <typename T, int WSTRIP, typename I>
__device__ inline void v2_wstrip_nhwc_body(const T* __restrict__ A,
                                           const T* __restrict__ Wt,
                                           T* __restrict__ D,
                                           const ValuParams& p)
{
    const int Wtiles = (p.Wo + WSTRIP - 1) / WSTRIP;
    const I tid = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x;

    const DwDivider<I> byC(p.C), byTiles(Wtiles), byHo(p.Ho);
    I rest = byC.div(tid);
    const int k = static_cast<int>(tid - rest * p.C); // channel — innermost/coalesced
    I next = byTiles.div(rest);
    const int wt = static_cast<int>(rest - next * Wtiles); // which W-strip
    rest = byHo.div(next);
    const int ho = static_cast<int>(next - rest * p.Ho);
    const int n = static_cast<int>(rest);
    if(n >= p.N)
        return;

    const int wo_base = wt * WSTRIP;

    float acc[WSTRIP];
#pragma unroll
    for(int j = 0; j < WSTRIP; ++j)
        acc[j] = 0.0f;

    const T* wbase = Wt + (long)k * p.kh * p.kw;

    for(int ky = 0; ky < p.kh; ++ky)
    {
        const int ih = ho * p.sh - p.ph + ky * p.dh;
        const bool vy = dwIn(ih, p.Hi);
        const I a_row = (((I)n * p.Hi + (vy ? ih : 0)) * p.Wi) * p.C + k;
        for(int kx = 0; kx < p.kw; ++kx)
        {
            const float w = valu_to_f32(wbase[dwTap2(ky, kx, p.kh, p.kw)]); // loaded once per tap
#pragma unroll
            for(int j = 0; j < WSTRIP; ++j)
            {
                const int wo = wo_base + j;
                const int iw = wo * p.sw - p.pw + kx * p.dw;
                const bool v = vy & (wo < p.Wo) & dwIn(iw, p.Wi);
                const float a = valu_to_f32(A[a_row + (I)(v ? iw : 0) * p.C]);
                const float t = acc[j] + w * a;
                acc[j] = v ? t : acc[j];
            }
        }
    }

#pragma unroll
    for(int j = 0; j < WSTRIP; ++j)
    {
        const int wo = wo_base + j;
        if(wo >= p.Wo)
            continue;
        const I o = (((I)n * p.Ho + ho) * p.Wo + wo) * p.C + k;
        valu_from_f32(acc[j], D[o]);
    }
}

template <typename T, int WSTRIP>
__device__ inline void v2_wstrip_core_nhwc(const T* __restrict__ A,
                                           const T* __restrict__ Wt,
                                           T* __restrict__ D,
                                           ValuParams p)
{
    if(dwTensorsFit32(p))
        v2_wstrip_nhwc_body<T, WSTRIP, int>(A, Wt, D, p);
    else
        v2_wstrip_nhwc_body<T, WSTRIP, long>(A, Wt, D, p);
}

// ---- v2 (NCHW) — channel-outer 2D floor; W contiguous, coalesced on W ----
// The channel-first mirror of v2_wstrip_core_nhwc. In NCHW the width axis is
// contiguous, so consecutive threads map to consecutive output columns (wo) for
// coalesced loads/stores, and each thread computes one output element (no W-strip
// — a per-thread strip along the contiguous axis would break coalescing; the
// tiny KH*KW weights are reused from cache instead). Channel is the outer plane.
// Any stride/dilation/padding/kernel size. Weights W[C,KH,KW] contiguous — the
// depthwise filter [C,1,KH,KW] in NCHW is byte-identical to the NHWC [C,KH,KW,1]
// case, so the weight addressing is shared with v2_wstrip_core_nhwc.
template <typename T, typename I>
__device__ inline void v2_nchw_body(const T* __restrict__ A,
                                    const T* __restrict__ Wt,
                                    T* __restrict__ D,
                                    const ValuParams& p)
{
    const I tid = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x;
    const I total = (I)p.N * p.C * p.Ho * p.Wo;
    if(tid >= total)
        return;

    const DwDivider<I> byWo(p.Wo), byHo(p.Ho), byC(p.C);
    I rest = byWo.div(tid);
    const int wo = static_cast<int>(tid - rest * p.Wo); // contiguous — coalesced across lanes
    I next = byHo.div(rest);
    const int ho = static_cast<int>(rest - next * p.Ho);
    rest = byC.div(next);
    const int k = static_cast<int>(next - rest * p.C); // channel (outer plane)
    const int n = static_cast<int>(rest);

    float acc = 0.0f;
    const T* wbase = Wt + (long)k * p.kh * p.kw;
    const I a_chanrow = ((I)n * p.C + k) * p.Hi; // ((n*C+k)*Hi)

    for(int ky = 0; ky < p.kh; ++ky)
    {
        const int ih = ho * p.sh - p.ph + ky * p.dh;
        const bool vy = dwIn(ih, p.Hi);
        const I a_row = (a_chanrow + (vy ? ih : 0)) * p.Wi;
        for(int kx = 0; kx < p.kw; ++kx)
        {
            const int iw = wo * p.sw - p.pw + kx * p.dw;
            const bool v = vy & dwIn(iw, p.Wi);
            const float a = valu_to_f32(A[a_row + (v ? iw : 0)]);
            const float t = acc + valu_to_f32(wbase[dwTap2(ky, kx, p.kh, p.kw)]) * a;
            acc = v ? t : acc;
        }
    }

    const I o = (((I)n * p.C + k) * p.Ho + ho) * p.Wo + wo;
    valu_from_f32(acc, D[o]);
}

template <typename T>
__device__ inline void
    v2_core_nchw(const T* __restrict__ A, const T* __restrict__ Wt, T* __restrict__ D, ValuParams p)
{
    if(dwTensorsFit32(p))
        v2_nchw_body<T, int>(A, Wt, D, p);
    else
        v2_nchw_body<T, long>(A, Wt, D, p);
}

// ---- v2 (3D) — register W-strip over NDHWC; the universal 3D floor ----
// The mechanical depth extension of v2_wstrip_core_nhwc: one extra output-depth (do_)
// level in the thread-index decompose and one extra tap loop (kz -> id) wrapping
// the 2D tap nest, with depth folded into the NDHWC address math. Register
// footprint (acc[WSTRIP]) is identical to the 2D core. Any stride/dilation/pad/
// kernel size on all three spatial axes. Weights W[C, kd, kh, kw].
template <typename T, int WSTRIP, typename I>
__device__ inline void v2_wstrip_ndhwc_body(const T* __restrict__ A,
                                            const T* __restrict__ Wt,
                                            T* __restrict__ D,
                                            const ValuParams& p)
{
    const int Wtiles = (p.Wo + WSTRIP - 1) / WSTRIP;
    const I tid = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x;

    const DwDivider<I> byC(p.C), byTiles(Wtiles), byHo(p.Ho), byDo(p.Do);
    I rest = byC.div(tid);
    const int k = static_cast<int>(tid - rest * p.C); // channel — innermost/coalesced
    I next = byTiles.div(rest);
    const int wt = static_cast<int>(rest - next * Wtiles); // which W-strip
    rest = byHo.div(next);
    const int ho = static_cast<int>(next - rest * p.Ho);
    next = byDo.div(rest);
    const int do_ = static_cast<int>(rest - next * p.Do); // output depth
    const int n = static_cast<int>(next);
    if(n >= p.N)
        return;

    const int wo_base = wt * WSTRIP;

    float acc[WSTRIP];
#pragma unroll
    for(int j = 0; j < WSTRIP; ++j)
        acc[j] = 0.0f;

    const T* wbase = Wt + (long)k * p.kd * p.kh * p.kw;

    for(int kz = 0; kz < p.kd; ++kz)
    {
        const int id = do_ * p.sd - p.pd + kz * p.dd;
        const bool vz = dwIn(id, p.Di);
        for(int ky = 0; ky < p.kh; ++ky)
        {
            const int ih = ho * p.sh - p.ph + ky * p.dh;
            const bool vy = vz & dwIn(ih, p.Hi);
            const I a_row
                = ((((I)n * p.Di + (vz ? id : 0)) * p.Hi + (vy ? ih : 0)) * p.Wi) * p.C + k;
            for(int kx = 0; kx < p.kw; ++kx)
            {
                const float w = valu_to_f32(wbase[dwTap3(kz, ky, kx, p.kd, p.kh, p.kw)]);
#pragma unroll
                for(int j = 0; j < WSTRIP; ++j)
                {
                    const int wo = wo_base + j;
                    const int iw = wo * p.sw - p.pw + kx * p.dw;
                    const bool v = vy & (wo < p.Wo) & dwIn(iw, p.Wi);
                    const float a = valu_to_f32(A[a_row + (I)(v ? iw : 0) * p.C]);
                    const float t = acc[j] + w * a;
                    acc[j] = v ? t : acc[j];
                }
            }
        }
    }

#pragma unroll
    for(int j = 0; j < WSTRIP; ++j)
    {
        const int wo = wo_base + j;
        if(wo >= p.Wo)
            continue;
        const I o = ((((I)n * p.Do + do_) * p.Ho + ho) * p.Wo + wo) * p.C + k;
        valu_from_f32(acc[j], D[o]);
    }
}

template <typename T, int WSTRIP>
__device__ inline void v2_wstrip_core_ndhwc(const T* __restrict__ A,
                                            const T* __restrict__ Wt,
                                            T* __restrict__ D,
                                            ValuParams p)
{
    if(dwTensorsFit32(p))
        v2_wstrip_ndhwc_body<T, WSTRIP, int>(A, Wt, D, p);
    else
        v2_wstrip_ndhwc_body<T, WSTRIP, long>(A, Wt, D, p);
}

// ---- v2 (NCDHW) — channel-outer 3D floor; W contiguous, coalesced on W ----
// The channel-first mirror of v2_wstrip_core_ndhwc (and the depth extension of
// v2_core_nchw): one output element per thread, wo innermost for coalescing, an
// extra output-depth (do_) decode level and a kz->id tap loop wrapping the 2D
// nest. NCDHW address math. Any stride/dilation/pad/kernel size on all three
// axes. Weights W[C,kd,kh,kw] contiguous.
template <typename T, typename I>
__device__ inline void v2_ncdhw_body(const T* __restrict__ A,
                                     const T* __restrict__ Wt,
                                     T* __restrict__ D,
                                     const ValuParams& p)
{
    const I tid = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x;
    const I total = (I)p.N * p.C * p.Do * p.Ho * p.Wo;
    if(tid >= total)
        return;

    const DwDivider<I> byWo(p.Wo), byHo(p.Ho), byDo(p.Do), byC(p.C);
    I rest = byWo.div(tid);
    const int wo = static_cast<int>(tid - rest * p.Wo); // contiguous — coalesced across lanes
    I next = byHo.div(rest);
    const int ho = static_cast<int>(rest - next * p.Ho);
    rest = byDo.div(next);
    const int do_ = static_cast<int>(next - rest * p.Do); // output depth
    next = byC.div(rest);
    const int k = static_cast<int>(rest - next * p.C); // channel (outer plane)
    const int n = static_cast<int>(next);

    float acc = 0.0f;
    const T* wbase = Wt + (long)k * p.kd * p.kh * p.kw;
    const I a_chan = (I)n * p.C + k; // (n*C+k) — channel plane base

    for(int kz = 0; kz < p.kd; ++kz)
    {
        const int id = do_ * p.sd - p.pd + kz * p.dd;
        const bool vz = dwIn(id, p.Di);
        for(int ky = 0; ky < p.kh; ++ky)
        {
            const int ih = ho * p.sh - p.ph + ky * p.dh;
            const bool vy = vz & dwIn(ih, p.Hi);
            const I a_row = (((a_chan * p.Di + (vz ? id : 0)) * p.Hi) + (vy ? ih : 0)) * p.Wi;
            for(int kx = 0; kx < p.kw; ++kx)
            {
                const int iw = wo * p.sw - p.pw + kx * p.dw;
                const bool v = vy & dwIn(iw, p.Wi);
                const float a = valu_to_f32(A[a_row + (v ? iw : 0)]);
                const float t = acc + valu_to_f32(wbase[dwTap3(kz, ky, kx, p.kd, p.kh, p.kw)]) * a;
                acc = v ? t : acc;
            }
        }
    }

    const I o = ((((I)n * p.C + k) * p.Do + do_) * p.Ho + ho) * p.Wo + wo;
    valu_from_f32(acc, D[o]);
}

template <typename T>
__device__ inline void v2_core_ncdhw(const T* __restrict__ A,
                                     const T* __restrict__ Wt,
                                     T* __restrict__ D,
                                     ValuParams p)
{
    if(dwTensorsFit32(p))
        v2_ncdhw_body<T, int>(A, Wt, D, p);
    else
        v2_ncdhw_body<T, long>(A, Wt, D, p);
}

// ---- v3a microtile: register 2D halo reuse (no LDS, no barrier) ----
// Each thread emits a TH×TW output tile and caches the (TH+KH-1)×(TW+KW-1) input
// patch in registers. Bakes s=d=1 and compile-time KH/KW so the tap loops and the
// register patch/accumulators fully unroll into named registers.
template <typename T, int KH, int KW, int TH, int TW, typename I>
__device__ inline void v3a_microtile_nhwc_body(const T* __restrict__ A,
                                               const T* __restrict__ Wt,
                                               T* __restrict__ D,
                                               const ValuParams& p)
{
    const int Htiles = (p.Ho + TH - 1) / TH;
    const int Wtiles = (p.Wo + TW - 1) / TW;
    const I tid = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x;

    const DwDivider<I> byC(p.C), byWt(Wtiles), byHt(Htiles);
    I rest = byC.div(tid);
    const int k = static_cast<int>(tid - rest * p.C); // channel innermost
    I next = byWt.div(rest);
    const int wt = static_cast<int>(rest - next * Wtiles);
    rest = byHt.div(next);
    const int ht = static_cast<int>(next - rest * Htiles);
    const int n = static_cast<int>(rest);
    if(n >= p.N)
        return;

    const int ho0 = ht * TH, wo0 = wt * TW;
    const int ih0 = ho0 - p.ph, iw0 = wo0 - p.pw; // s=d=1
    const int PH = TH + KH - 1, PW = TW + KW - 1;

    // register input patch, loaded once per thread per channel; every load issued (an element
    // outside the image reads the image's first one) and its value masked
    const I image = (I)n * p.Hi * p.Wi;
    float patch[PH][PW];
#pragma unroll
    for(int py = 0; py < PH; ++py)
    {
        const int ih = ih0 + py;
        const bool ihv = dwIn(ih, p.Hi);
#pragma unroll
        for(int px = 0; px < PW; ++px)
        {
            const int iw = iw0 + px;
            const bool v = ihv & dwIn(iw, p.Wi);
            const float a = valu_to_f32(A[(image + (v ? ih * p.Wi + iw : 0)) * p.C + k]);
            patch[py][px] = v ? a : 0.0f;
        }
    }

    // weights once per thread
    float w[KH][KW];
    const T* wb = Wt + (long)k * KH * KW;
#pragma unroll
    for(int ky = 0; ky < KH; ++ky)
#pragma unroll
        for(int kx = 0; kx < KW; ++kx)
            w[ky][kx] = valu_to_f32(wb[dwTap2(ky, kx, KH, KW)]);

    float acc[TH][TW];
#pragma unroll
    for(int ty = 0; ty < TH; ++ty)
#pragma unroll
        for(int tx = 0; tx < TW; ++tx)
            acc[ty][tx] = 0.0f;

#pragma unroll
    for(int ky = 0; ky < KH; ++ky)
#pragma unroll
        for(int kx = 0; kx < KW; ++kx)
#pragma unroll
            for(int ty = 0; ty < TH; ++ty)
#pragma unroll
                for(int tx = 0; tx < TW; ++tx)
                    acc[ty][tx] += w[ky][kx] * patch[ty + ky][tx + kx];

#pragma unroll
    for(int ty = 0; ty < TH; ++ty)
    {
        const int ho = ho0 + ty;
        if(ho >= p.Ho)
            continue;
#pragma unroll
        for(int tx = 0; tx < TW; ++tx)
        {
            const int wo = wo0 + tx;
            if(wo >= p.Wo)
                continue;
            const I o = (((I)n * p.Ho + ho) * p.Wo + wo) * p.C + k;
            valu_from_f32(acc[ty][tx], D[o]);
        }
    }
}

template <typename T, int KH, int KW, int TH, int TW>
__device__ inline void v3a_microtile_core_nhwc(const T* __restrict__ A,
                                               const T* __restrict__ Wt,
                                               T* __restrict__ D,
                                               ValuParams p)
{
    if(dwTensorsFit32(p))
        v3a_microtile_nhwc_body<T, KH, KW, TH, TW, int>(A, Wt, D, p);
    else
        v3a_microtile_nhwc_body<T, KH, KW, TH, TW, long>(A, Wt, D, p);
}

// Stages the halo'd [LD][LH][LW][BK] channels-last block (LD = 1 in 2-D) of image An into LDS,
// padding and the channel tail as zeros: NT threads, a compile-time trip count, fully unrolled,
// every load issued (an element outside the image reads its first one) and its value selected.
template <typename T, int LD, int LH, int LW, int BK, int NT, typename I>
__device__ inline void dwStageChannelsLast(T* __restrict__ tile,
                                           const T* __restrict__ An,
                                           const ValuParams& p,
                                           int id0,
                                           int ih0,
                                           int iw0,
                                           int k_base)
{
    constexpr int TOTAL = LD * LH * LW * BK;
    constexpr int ITERS = (TOTAL + NT - 1) / NT;
    T zero;
    valu_from_f32(0.0f, zero);
    const int tid = static_cast<int>(threadIdx.x);
    // All of the thread's loads first, then the LDS stores: the loads stay in flight together.
    T v[ITERS];
    bool in[ITERS];
#pragma unroll
    for(int it = 0; it < ITERS; ++it)
    {
        const int idx = tid + it * NT;
        const int kc = idx % BK;
        int t = idx / BK;
        const int px = t % LW;
        t /= LW;
        const int py = t % LH;
        const int pz = t / LH;
        const int id = id0 + pz, ih = ih0 + py, iw = iw0 + px, kk = k_base + kc;
        in[it] = dwIn(id, p.Di) & dwIn(ih, p.Hi) & dwIn(iw, p.Wi) & (kk < p.C);
        v[it] = An[in[it] ? (((I)id * p.Hi + ih) * p.Wi + iw) * p.C + kk : 0];
    }
#pragma unroll
    for(int it = 0; it < ITERS; ++it)
    {
        const int idx = tid + it * NT;
        if(TOTAL % NT == 0 || it < ITERS - 1 || idx < TOTAL)
            tile[idx] = in[it] ? v[it] : zero;
    }
}

// ---- v4 fused: LDS block stage + register micro-tile ----
// The workgroup stages a [BH+KH-1, BW+KW-1, BK] halo'd block into LDS once, then
// each thread reads an (RH+KH-1)×(RW+KW-1) patch from LDS into registers and
// accumulates RH×RW outputs. threads/block = (BH/RH)*(BW/RW)*BK.
template <typename T, int KH, int KW, int BH, int BW, int BK, int RH, int RW, typename I>
__device__ inline void v4_fused_nhwc_body(const T* __restrict__ A,
                                          const T* __restrict__ Wt,
                                          T* __restrict__ D,
                                          const ValuParams& p,
                                          T* __restrict__ tile)
{
    const int LH = BH + KH - 1, LW = BW + KW - 1;

    const int Hblocks = (p.Ho + BH - 1) / BH;
    const int Wblocks = (p.Wo + BW - 1) / BW;
    const int Kblocks = (p.C + BK - 1) / BK;

    unsigned blk = blockIdx.x;
    const int kb = blk % Kblocks;
    blk /= Kblocks;
    const int wb = blk % Wblocks;
    blk /= Wblocks;
    const int hb = blk % Hblocks;
    const int n = blk / Hblocks; // grid sized so n<N

    const int ho_base = hb * BH, wo_base = wb * BW, k_base = kb * BK;
    const int ih0 = ho_base - p.ph, iw0 = wo_base - p.pw; // s=d=1

    // cooperative halo stage into LDS
    const T* An = A + (long)n * p.Hi * p.Wi * p.C;
    dwStageChannelsLast<T, 1, LH, LW, BK, (BH / RH) * (BW / RW) * BK, I>(
        tile, An, p, 0, ih0, iw0, k_base);
    dwLdsBarrier();

    const int tid = threadIdx.x;
    const int klocal = tid % BK;
    int rest = tid / BK;
    const int wtile = rest % (BW / RW);
    const int htile = rest / (BW / RW);
    const int oy = htile * RH, ox = wtile * RW;
    const int k = k_base + klocal;

    const int PHr = RH + KH - 1, PWr = RW + KW - 1;
    float patch[PHr][PWr];
#pragma unroll
    for(int py = 0; py < PHr; ++py)
#pragma unroll
        for(int px = 0; px < PWr; ++px)
            patch[py][px] = valu_to_f32(tile[(((oy + py) * LW) + (ox + px)) * BK + klocal]);

    // A channel past C reads the last channel's filter; its outputs are not stored.
    float w[KH][KW];
    const T* wbp = Wt + (long)(k < p.C ? k : p.C - 1) * KH * KW;
#pragma unroll
    for(int ky = 0; ky < KH; ++ky)
#pragma unroll
        for(int kx = 0; kx < KW; ++kx)
            w[ky][kx] = valu_to_f32(wbp[dwTap2(ky, kx, KH, KW)]);

    float acc[RH][RW];
#pragma unroll
    for(int ty = 0; ty < RH; ++ty)
#pragma unroll
        for(int tx = 0; tx < RW; ++tx)
            acc[ty][tx] = 0.0f;

#pragma unroll
    for(int ky = 0; ky < KH; ++ky)
#pragma unroll
        for(int kx = 0; kx < KW; ++kx)
#pragma unroll
            for(int ty = 0; ty < RH; ++ty)
#pragma unroll
                for(int tx = 0; tx < RW; ++tx)
                    acc[ty][tx] += w[ky][kx] * patch[ty + ky][tx + kx];

    if(k >= p.C)
        return;
    T* Dn = D + (long)n * p.Ho * p.Wo * p.C;
#pragma unroll
    for(int ty = 0; ty < RH; ++ty)
    {
        const int ho = ho_base + oy + ty;
        if(ho >= p.Ho)
            continue;
#pragma unroll
        for(int tx = 0; tx < RW; ++tx)
        {
            const int wo = wo_base + ox + tx;
            if(wo >= p.Wo)
                continue;
            const I o = ((I)ho * p.Wo + wo) * p.C + k;
            valu_from_f32(acc[ty][tx], Dn[o]);
        }
    }
}

template <typename T, int KH, int KW, int BH, int BW, int BK, int RH, int RW>
__device__ inline void v4_fused_core_nhwc(const T* __restrict__ A,
                                          const T* __restrict__ Wt,
                                          T* __restrict__ D,
                                          ValuParams p)
{
    // Tiling + LDS-budget invariants: a bad config would silently under-launch
    // (partial-tile threads never spawned -> wrong output) or overflow LDS.
    static_assert(BH % RH == 0, "v4_fused: BH must be divisible by RH");
    static_assert(BW % RW == 0, "v4_fused: BW must be divisible by RW");
    static_assert((BH + KH - 1) * (BW + KW - 1) * BK * static_cast<int>(sizeof(T)) <= 65536,
                  "v4_fused: LDS tile exceeds 64 KB");
    // One LDS tile for both index widths (a __shared__ array per instantiation would be two).
    __shared__ T tile[(BH + KH - 1) * (BW + KW - 1) * BK];
    if(dwImagesFit32(p))
        v4_fused_nhwc_body<T, KH, KW, BH, BW, BK, RH, RW, int>(A, Wt, D, p, tile);
    else
        v4_fused_nhwc_body<T, KH, KW, BH, BW, BK, RH, RW, long>(A, Wt, D, p, tile);
}

// ---- v3b lds-direct: LDS block stage, tap loop reads from LDS ----
// Register-light (no materialised patch) so it survives large KH/KW where v4's
// register patch would spill. LDS tap traffic is KH*KW loads/output, but LDS
// bandwidth >> DRAM, so this is the right tool for 9x9 depthwise.
template <typename T, int KH, int KW, int BH, int BW, int BK, int RH, int RW, typename I>
__device__ inline void v3b_lds_nhwc_body(const T* __restrict__ A,
                                         const T* __restrict__ Wt,
                                         T* __restrict__ D,
                                         const ValuParams& p,
                                         T* __restrict__ tile)
{
    const int LH = BH + KH - 1, LW = BW + KW - 1;

    const int Hblocks = (p.Ho + BH - 1) / BH;
    const int Wblocks = (p.Wo + BW - 1) / BW;
    const int Kblocks = (p.C + BK - 1) / BK;

    unsigned blk = blockIdx.x;
    const int kb = blk % Kblocks;
    blk /= Kblocks;
    const int wb = blk % Wblocks;
    blk /= Wblocks;
    const int hb = blk % Hblocks;
    const int n = blk / Hblocks;

    const int ho_base = hb * BH, wo_base = wb * BW, k_base = kb * BK;
    const int ih0 = ho_base - p.ph, iw0 = wo_base - p.pw;

    const T* An = A + (long)n * p.Hi * p.Wi * p.C;
    dwStageChannelsLast<T, 1, LH, LW, BK, (BH / RH) * (BW / RW) * BK, I>(
        tile, An, p, 0, ih0, iw0, k_base);
    dwLdsBarrier();

    const int tid = threadIdx.x;
    const int klocal = tid % BK;
    int rest = tid / BK;
    const int wtile = rest % (BW / RW);
    const int htile = rest / (BW / RW);
    const int oy = htile * RH, ox = wtile * RW;
    const int k = k_base + klocal;

    float acc[RH][RW];
#pragma unroll
    for(int ty = 0; ty < RH; ++ty)
#pragma unroll
        for(int tx = 0; tx < RW; ++tx)
            acc[ty][tx] = 0.0f;

    // A channel past C reads the last channel's filter; its outputs are not stored.
    const T* wbp = Wt + (long)(k < p.C ? k : p.C - 1) * KH * KW;
#pragma unroll
    for(int ky = 0; ky < KH; ++ky)
#pragma unroll
        for(int kx = 0; kx < KW; ++kx)
        {
            const float wv = valu_to_f32(wbp[dwTap2(ky, kx, KH, KW)]);
#pragma unroll
            for(int ty = 0; ty < RH; ++ty)
#pragma unroll
                for(int tx = 0; tx < RW; ++tx)
                    acc[ty][tx]
                        += wv
                           * valu_to_f32(
                               tile[(((oy + ty + ky) * LW) + (ox + tx + kx)) * BK + klocal]);
        }

    if(k >= p.C)
        return;
    T* Dn = D + (long)n * p.Ho * p.Wo * p.C;
#pragma unroll
    for(int ty = 0; ty < RH; ++ty)
    {
        const int ho = ho_base + oy + ty;
        if(ho >= p.Ho)
            continue;
#pragma unroll
        for(int tx = 0; tx < RW; ++tx)
        {
            const int wo = wo_base + ox + tx;
            if(wo >= p.Wo)
                continue;
            const I o = ((I)ho * p.Wo + wo) * p.C + k;
            valu_from_f32(acc[ty][tx], Dn[o]);
        }
    }
}

template <typename T, int KH, int KW, int BH, int BW, int BK, int RH, int RW>
__device__ inline void v3b_lds_core_nhwc(const T* __restrict__ A,
                                         const T* __restrict__ Wt,
                                         T* __restrict__ D,
                                         ValuParams p)
{
    static_assert(BH % RH == 0, "v3b_lds: BH must be divisible by RH");
    static_assert(BW % RW == 0, "v3b_lds: BW must be divisible by RW");
    static_assert((BH + KH - 1) * (BW + KW - 1) * BK * static_cast<int>(sizeof(T)) <= 65536,
                  "v3b_lds: LDS tile exceeds 64 KB");
    __shared__ T tile[(BH + KH - 1) * (BW + KW - 1) * BK];
    if(dwImagesFit32(p))
        v3b_lds_nhwc_body<T, KH, KW, BH, BW, BK, RH, RW, int>(A, Wt, D, p, tile);
    else
        v3b_lds_nhwc_body<T, KH, KW, BH, BW, BK, RH, RW, long>(A, Wt, D, p, tile);
}

// ---- v3b (3D) lds-direct over NDHWC: LDS block stage, tap loop reads from LDS ----
// The depth extension of v3b_lds_core_nhwc: the workgroup stages a halo'd
// [BD+KD-1, BH+KH-1, BW+KW-1, BK] block into LDS once, then each thread loops the
// full KD*KH*KW tap nest reading from LDS and accumulates an RD*RH*RW output
// micro-tile. This is the 3D depthwise workhorse: a 3x3x3 stencil reloads each
// input element ~27x from *LDS* instead of from DRAM (the v2 3D floor's cost),
// cutting global traffic to one load per staged element. Register-light (only
// acc[RD][RH][RW]; weights are scalar-per-tap) so the 27-tap nest never spills.
// s=d=1 on all three axes; compile-time KD/KH/KW. NDHWC, channel (k) innermost.
template <typename T,
          int KD,
          int KH,
          int KW,
          int BD,
          int BH,
          int BW,
          int BK,
          int RD,
          int RH,
          int RW,
          typename I>
__device__ inline void v3b_lds_ndhwc_body(const T* __restrict__ A,
                                          const T* __restrict__ Wt,
                                          T* __restrict__ D,
                                          const ValuParams& p,
                                          T* __restrict__ tile) // [pz][py][px][kc], kc innermost
{
    const int LD = BD + KD - 1, LH = BH + KH - 1, LW = BW + KW - 1;

    const int Dblocks = (p.Do + BD - 1) / BD;
    const int Hblocks = (p.Ho + BH - 1) / BH;
    const int Wblocks = (p.Wo + BW - 1) / BW;
    const int Kblocks = (p.C + BK - 1) / BK;

    unsigned blk = blockIdx.x;
    const int kb = blk % Kblocks;
    blk /= Kblocks;
    const int wb = blk % Wblocks;
    blk /= Wblocks;
    const int hb = blk % Hblocks;
    blk /= Hblocks;
    const int db = blk % Dblocks;
    const int n = blk / Dblocks; // grid sized so n < N

    const int do_base = db * BD, ho_base = hb * BH, wo_base = wb * BW, k_base = kb * BK;
    const int id0 = do_base - p.pd, ih0 = ho_base - p.ph, iw0 = wo_base - p.pw; // s=d=1

    const T* An = A + (long)n * p.Di * p.Hi * p.Wi * p.C;
    dwStageChannelsLast<T, LD, LH, LW, BK, (BD / RD) * (BH / RH) * (BW / RW) * BK, I>(
        tile, An, p, id0, ih0, iw0, k_base);
    dwLdsBarrier();

    const int tid = threadIdx.x;
    const int klocal = tid % BK;
    int rest = tid / BK;
    const int wtile = rest % (BW / RW);
    rest /= (BW / RW);
    const int htile = rest % (BH / RH);
    const int dtile = rest / (BH / RH);
    const int oz = dtile * RD, oy = htile * RH, ox = wtile * RW;
    const int k = k_base + klocal;

    float acc[RD][RH][RW];
#pragma unroll
    for(int tz = 0; tz < RD; ++tz)
#pragma unroll
        for(int ty = 0; ty < RH; ++ty)
#pragma unroll
            for(int tx = 0; tx < RW; ++tx)
                acc[tz][ty][tx] = 0.0f;

    // A channel past C reads the last channel's filter; its outputs are not stored.
    const T* wbp = Wt + (long)(k < p.C ? k : p.C - 1) * KD * KH * KW;
#pragma unroll
    for(int kz = 0; kz < KD; ++kz)
#pragma unroll
        for(int ky = 0; ky < KH; ++ky)
#pragma unroll
            for(int kx = 0; kx < KW; ++kx)
            {
                const float wv = valu_to_f32(wbp[dwTap3(kz, ky, kx, KD, KH, KW)]);
#pragma unroll
                for(int tz = 0; tz < RD; ++tz)
#pragma unroll
                    for(int ty = 0; ty < RH; ++ty)
#pragma unroll
                        for(int tx = 0; tx < RW; ++tx)
                            acc[tz][ty][tx]
                                += wv
                                   * valu_to_f32(tile[((((oz + tz + kz) * LH) + (oy + ty + ky)) * LW
                                                       + (ox + tx + kx))
                                                          * BK
                                                      + klocal]);
            }

    if(k >= p.C)
        return;
    T* Dn = D + (long)n * p.Do * p.Ho * p.Wo * p.C;
#pragma unroll
    for(int tz = 0; tz < RD; ++tz)
    {
        const int do_ = do_base + oz + tz;
        if(do_ >= p.Do)
            continue;
#pragma unroll
        for(int ty = 0; ty < RH; ++ty)
        {
            const int ho = ho_base + oy + ty;
            if(ho >= p.Ho)
                continue;
#pragma unroll
            for(int tx = 0; tx < RW; ++tx)
            {
                const int wo = wo_base + ox + tx;
                if(wo >= p.Wo)
                    continue;
                const I o = (((I)do_ * p.Ho + ho) * p.Wo + wo) * p.C + k;
                valu_from_f32(acc[tz][ty][tx], Dn[o]);
            }
        }
    }
}

template <typename T,
          int KD,
          int KH,
          int KW,
          int BD,
          int BH,
          int BW,
          int BK,
          int RD,
          int RH,
          int RW>
__device__ inline void v3b_lds_core_ndhwc(const T* __restrict__ A,
                                          const T* __restrict__ Wt,
                                          T* __restrict__ D,
                                          ValuParams p)
{
    static_assert(BD % RD == 0, "v3b_lds_3d: BD must be divisible by RD");
    static_assert(BH % RH == 0, "v3b_lds_3d: BH must be divisible by RH");
    static_assert(BW % RW == 0, "v3b_lds_3d: BW must be divisible by RW");
    static_assert((BD + KD - 1) * (BH + KH - 1) * (BW + KW - 1) * BK * static_cast<int>(sizeof(T))
                      <= 65536,
                  "v3b_lds_3d: LDS tile exceeds 64 KB");
    __shared__ T tile[(BD + KD - 1) * (BH + KH - 1) * (BW + KW - 1) * BK];
    if(dwImagesFit32(p))
        v3b_lds_ndhwc_body<T, KD, KH, KW, BD, BH, BW, BK, RD, RH, RW, int>(A, Wt, D, p, tile);
    else
        v3b_lds_ndhwc_body<T, KD, KH, KW, BD, BH, BW, BK, RD, RH, RW, long>(A, Wt, D, p, tile);
}

// ---- v3b (NCHW) — channel-first LDS plane tile + register micro-tile ----
// Not from PR #10945: the channel-first counterpart of the halo variants, for the shapes where
// the NCHW floor reads every input KH*KW times from global memory. One block per (image,
// BH x BW output tile, BK channels): the tile's (BH+KH-1) x (BW+KW-1) input halo of each channel
// plane is staged in LDS as fp32, padding resolved as zeros at load, with consecutive threads on
// consecutive columns so the loads coalesce along the contiguous width. Each thread then
// computes RH x RW outputs of one plane, reading each LDS row segment once per filter row and
// reusing it across the filter's width. s=d=1, compile-time KH/KW; weights W[C,KH,KW].
template <typename T, int KH, int KW, int BH, int BW, int BK, int RH, int RW, typename I>
__device__ inline void v3b_lds_nchw_body(const T* __restrict__ A,
                                         const T* __restrict__ Wt,
                                         T* __restrict__ D,
                                         const ValuParams& p,
                                         float (*__restrict__ tile)[BH + KH - 1][BW + KW]) // [BK]
{
    constexpr int HH = BH + KH - 1;
    constexpr int HW = BW + KW - 1;
    constexpr int TX = BW / RW;
    constexpr int TY = BH / RH;
    constexpr int THREADS = TX * TY * BK;
    constexpr int TOTAL = BK * HH * HW;
    constexpr int ITERS = (TOTAL + THREADS - 1) / THREADS;

    const int tilesW = (p.Wo + BW - 1) / BW;
    const int tilesH = (p.Ho + BH - 1) / BH;
    const int chunks = (p.C + BK - 1) / BK;
    int block = static_cast<int>(blockIdx.x);
    const int cb = block % chunks;
    block /= chunks;
    const int tw = block % tilesW;
    block /= tilesW;
    const int th = block % tilesH;
    const int n = block / tilesH;

    const int oy0 = th * BH;
    const int ox0 = tw * BW;
    const int iy0 = oy0 - p.ph;
    const int ix0 = ox0 - p.pw;
    const int c0 = cb * BK;
    const int tid = static_cast<int>(threadIdx.x);

    // Compile-time trip count, fully unrolled; every load issued (outside the image or past
    // the last channel: the image's first element) and its value selected. All of the thread's
    // loads first, then the LDS stores: the loads stay in flight together.
    const T* An = A + (long)n * p.C * p.Hi * p.Wi;
    T v[ITERS];
    bool in[ITERS];
#pragma unroll
    for(int it = 0; it < ITERS; ++it)
    {
        const int i = tid + it * THREADS;
        const int col = i % HW;
        const int rest = i / HW;
        const int row = rest % HH;
        const int c = c0 + rest / HH;
        const int iy = iy0 + row;
        const int ix = ix0 + col;
        in[it] = (c < p.C) & dwIn(iy, p.Hi) & dwIn(ix, p.Wi);
        v[it] = An[in[it] ? ((I)c * p.Hi + iy) * p.Wi + ix : 0];
    }
#pragma unroll
    for(int it = 0; it < ITERS; ++it)
    {
        const int i = tid + it * THREADS;
        if(TOTAL % THREADS == 0 || it < ITERS - 1 || i < TOTAL)
        {
            const int col = i % HW;
            const int rest = i / HW;
            tile[rest / HH][rest % HH][col] = in[it] ? valu_to_f32(v[it]) : 0.0f;
        }
    }
    dwLdsBarrier();

    const int tx = tid % TX;
    const int ty = (tid / TX) % TY;
    const int kc = tid / (TX * TY);
    const int c = c0 + kc;
    if(c >= p.C)
    {
        return;
    }
    float w[KH][KW];
#pragma unroll
    for(int ky = 0; ky < KH; ++ky)
    {
#pragma unroll
        for(int kx = 0; kx < KW; ++kx)
        {
            w[ky][kx] = valu_to_f32(Wt[static_cast<long>(c) * KH * KW + dwTap2(ky, kx, KH, KW)]);
        }
    }
    float acc[RH][RW];
#pragma unroll
    for(int r = 0; r < RH; ++r)
    {
#pragma unroll
        for(int j = 0; j < RW; ++j)
        {
            acc[r][j] = 0.0f;
        }
    }
#pragma unroll
    for(int r = 0; r < RH; ++r)
    {
#pragma unroll
        for(int ky = 0; ky < KH; ++ky)
        {
            float segment[RW + KW - 1];
            const float* row = &tile[kc][ty * RH + r + ky][tx * RW];
#pragma unroll
            for(int j = 0; j < RW + KW - 1; ++j)
            {
                segment[j] = row[j];
            }
#pragma unroll
            for(int kx = 0; kx < KW; ++kx)
            {
#pragma unroll
                for(int j = 0; j < RW; ++j)
                {
                    acc[r][j] += w[ky][kx] * segment[j + kx];
                }
            }
        }
    }
    T* Dn = D + (long)n * p.C * p.Ho * p.Wo;
#pragma unroll
    for(int r = 0; r < RH; ++r)
    {
        const int oy = oy0 + ty * RH + r;
        if(oy >= p.Ho)
        {
            continue;
        }
        T* out = Dn + ((I)c * p.Ho + oy) * p.Wo;
#pragma unroll
        for(int j = 0; j < RW; ++j)
        {
            const int ox = ox0 + tx * RW + j;
            if(ox < p.Wo)
            {
                valu_from_f32(acc[r][j], out[ox]);
            }
        }
    }
}

template <typename T, int KH, int KW, int BH, int BW, int BK, int RH, int RW>
__device__ inline void v3b_lds_core_nchw(const T* __restrict__ A,
                                         const T* __restrict__ Wt,
                                         T* __restrict__ D,
                                         ValuParams p)
{
    static_assert(BH % RH == 0 && BW % RW == 0, "the micro-tile must divide the block tile");
    __shared__ float tile[BK][BH + KH - 1][BW + KW]; // a pad column: HW + 1
    if(dwImagesFit32(p))
        v3b_lds_nchw_body<T, KH, KW, BH, BW, BK, RH, RW, int>(A, Wt, D, p, tile);
    else
        v3b_lds_nchw_body<T, KH, KW, BH, BW, BK, RH, RW, long>(A, Wt, D, p, tile);
}

// ---- v5 — backward data as a gather; any stride, dilation and padding ----
// Not from PR #10945. The other cores run backward data as the forward convolution of dy,
// which holds for unit stride only. Here each thread owns one dx element and adds, over the
// filter taps, the dy element the forward convolution read it through: the output position o
// with o * stride = i + pad - tap * dilation, when that lands on the stride grid. No zero
// upsampling, so a strided problem spends no multiply on a skipped tap, and the sum needs no
// atomics. Geometry in gather terms: A = dy with extents (Di, Hi, Wi), D = dx with extents
// (Do, Ho, Wo), padding/stride/dilation as the forward convolution has them. Coalesced on width
// (channels-first) or channels (channels-last). Weights W[C, KD, KH, KW] contiguous.
template <typename T, int LAYOUT>
__device__ inline void v5_gather_bwd(const T* __restrict__ A,
                                     const T* __restrict__ Wt,
                                     T* __restrict__ D,
                                     ValuParams p)
{
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)p.N * p.C * p.Do * p.Ho * p.Wo;
    if(tid >= total)
        return;
    int k;
    int ix;
    int iy;
    int iz;
    int n;
    long rest = tid;
    if(LAYOUT == 0)
    {
        k = rest % p.C;
        rest /= p.C;
        ix = rest % p.Wo;
        rest /= p.Wo;
        iy = rest % p.Ho;
        rest /= p.Ho;
        iz = rest % p.Do;
        n = rest / p.Do;
    }
    else
    {
        ix = rest % p.Wo;
        rest /= p.Wo;
        iy = rest % p.Ho;
        rest /= p.Ho;
        iz = rest % p.Do;
        rest /= p.Do;
        k = rest % p.C;
        n = rest / p.C;
    }

    float acc = 0.0f;
    const T* wbase = Wt + (long)k * p.kd * p.kh * p.kw;
    for(int kz = 0; kz < p.kd; ++kz)
    {
        const int sz = iz + p.pd - kz * p.dd;
        if(sz < 0 || sz % p.sd != 0 || sz / p.sd >= p.Di)
            continue;
        const int oz = sz / p.sd;
        for(int ky = 0; ky < p.kh; ++ky)
        {
            const int sy = iy + p.ph - ky * p.dh;
            if(sy < 0 || sy % p.sh != 0 || sy / p.sh >= p.Hi)
                continue;
            const int oy = sy / p.sh;
            for(int kx = 0; kx < p.kw; ++kx)
            {
                const int sx = ix + p.pw - kx * p.dw;
                if(sx < 0 || sx % p.sw != 0 || sx / p.sw >= p.Wi)
                    continue;
                const int ox = sx / p.sw;
                const long a = LAYOUT == 0
                                   ? ((((long)n * p.Di + oz) * p.Hi + oy) * p.Wi + ox) * p.C + k
                                   : ((((long)n * p.C + k) * p.Di + oz) * p.Hi + oy) * p.Wi + ox;
                acc += valu_to_f32(wbase[dwTap3(kz, ky, kx, p.kd, p.kh, p.kw)]) * valu_to_f32(A[a]);
            }
        }
    }
    const long o = LAYOUT == 0 ? ((((long)n * p.Do + iz) * p.Ho + iy) * p.Wo + ix) * p.C + k
                               : ((((long)n * p.C + k) * p.Do + iz) * p.Ho + iy) * p.Wo + ix;
    valu_from_f32(acc, D[o]);
}

/// A compile-time flag for generic lambdas (HIPRTC has no standard headers).
template <bool B>
struct DwFlag
{
    static constexpr bool value = B;
};

// ---- v6 — backward data by stride phase blocks; stride S compile-time, unit dilation ----
// A thread owns the S x S dx pixels sharing
// (qy, qx) = ((y + ph) / S, (x + pw) / S), one per phase (ry, rx). They all read the dy window
// rows qy - m, columns qx - m' (m < ceil(KH / S)), which the thread loads once; phase (ry, rx)
// takes taps (ry + S m, rx + S m'). Channels-last: VEC channels per thread, CVB channel vectors per
// block, the block's filter slice in LDS as fp32.
template <typename T, int KH, int KW, int S, int VEC, int CVB>
__device__ inline void v6_phase_block_nhwc(const T* __restrict__ A,
                                           const T* __restrict__ Wt,
                                           T* __restrict__ D,
                                           ValuParams p)
{
    constexpr int PIX = 256 / CVB;
    constexpr int BC = CVB * VEC;
    constexpr int MY = (KH + S - 1) / S, MX = (KW + S - 1) / S;
    __shared__ float sW[KH * KW][BC];
    const int tid = threadIdx.x;
    // One-dimensional grid: channel block fastest, then pixel block.
    const int cblocks = (p.C + BC - 1) / BC;
    const int pblock = blockIdx.x / cblocks;
    const int cb = (blockIdx.x - pblock * cblocks) * BC;
    for(int i = tid; i < KH * KW * BC; i += 256)
    {
        const int tap = i / BC, cc = i - tap * BC, ch = cb + cc;
        const int ky = tap / KW, kx = tap - ky * KW;
        sW[tap][cc]
            = ch < p.C ? valu_to_f32(Wt[(long)ch * KH * KW + dwTap2(ky, kx, KH, KW)]) : 0.0f;
    }
    dwLdsBarrier();
    const int qh = (p.Ho - 1 + p.ph) / S + 1, qw = (p.Wo - 1 + p.pw) / S + 1;
    const int cv = tid % CVB, tp = tid / CVB;
    const int c0 = cb + cv * VEC;
    const int pix = pblock * PIX + tp;
    if(pix >= p.N * qh * qw || c0 >= p.C)
        return;
    const int qx = pix % qw;
    const int rest = pix / qw;
    const int qy = rest % qh;
    const int n = rest / qh;
    // Two forms: every channel of the vector real (the rule: whole-vector loads and stores), or
    // the channel tail (element-wise). One branch per thread, none per load.
    auto run = [&](auto wholeFlag) {
        constexpr bool whole = decltype(wholeFlag)::value;
        typedef unsigned int __attribute__((ext_vector_type(VEC * sizeof(T) / 4))) Raw;
        // The dy window, fp32.
        float win[MY][MX][VEC];
#pragma unroll
        for(int m = 0; m < MY; ++m)
#pragma unroll
            for(int mm = 0; mm < MX; ++mm)
            {
                // Every load is issued (a position outside dy reads the image's first one) and its
                // value masked, so the window's loads stay in flight together: a branch per load would
                // wait for each one alone.
                const int oy = qy - m, ox = qx - mm;
                const bool in = oy >= 0 && oy < p.Hi && ox >= 0 && ox < p.Wi;
                const T* a
                    = A + ((long)(n * p.Hi + (in ? oy : 0)) * p.Wi + (in ? ox : 0)) * p.C + c0;
                T v[VEC];
                if constexpr(whole)
                    *reinterpret_cast<Raw*>(v) = *reinterpret_cast<const Raw*>(a);
                else
#pragma unroll
                    for(int e = 0; e < VEC; ++e)
                        v[e] = a[c0 + e < p.C ? e : 0];
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                    win[m][mm][e] = in && c0 + e < p.C ? valu_to_f32(v[e]) : 0.0f;
            }
#pragma unroll
        for(int ry = 0; ry < S; ++ry)
        {
            const int y = S * qy + ry - p.ph;
            if(y < 0 || y >= p.Ho)
                continue;
#pragma unroll
            for(int rx = 0; rx < S; ++rx)
            {
                const int x = S * qx + rx - p.pw;
                if(x < 0 || x >= p.Wo)
                    continue;
                float acc[VEC];
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                    acc[e] = 0.0f;
#pragma unroll
                for(int m = 0; m < MY; ++m)
                {
                    if(ry + S * m >= KH)
                        continue;
#pragma unroll
                    for(int mm = 0; mm < MX; ++mm)
                    {
                        if(rx + S * mm >= KW)
                            continue;
                        const float* w = &sW[(ry + S * m) * KW + rx + S * mm][cv * VEC];
#pragma unroll
                        for(int e = 0; e < VEC; ++e)
                            acc[e] += w[e] * win[m][mm][e];
                    }
                }
                T* d = D + ((long)(n * p.Ho + y) * p.Wo + x) * p.C + c0;
                T o[VEC];
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                    valu_from_f32(acc[e], o[e]);
                if constexpr(whole)
                    *reinterpret_cast<Raw*>(d) = *reinterpret_cast<const Raw*>(o);
                else
#pragma unroll
                    for(int e = 0; e < VEC; ++e)
                        if(c0 + e < p.C)
                            d[e] = o[e];
            }
        }
    };
    if(c0 + VEC <= p.C && p.C % VEC == 0)
        run(DwFlag<true>{});
    else
        run(DwFlag<false>{});
}

// Channels-first phase blocks: lanes along qx of one dx row band; a thread computes S rows x S
// columns of dx (one (qy, qx) block) from its MY x MX dy window, filter of its channel in registers.
template <typename T, int KH, int KW, int S>
__device__ inline void v6_phase_block_nchw(const T* __restrict__ A,
                                           const T* __restrict__ Wt,
                                           T* __restrict__ D,
                                           ValuParams p)
{
    constexpr int MY = (KH + S - 1) / S, MX = (KW + S - 1) / S;
    const int qh = (p.Ho - 1 + p.ph) / S + 1, qw = (p.Wo - 1 + p.pw) / S + 1;
    const int t = blockIdx.x * 256 + threadIdx.x;
    if(t >= p.N * p.C * qh * qw)
        return;
    const int qx = t % qw;
    int rest = t / qw;
    const int qy = rest % qh;
    rest /= qh;
    const int c = rest % p.C;
    const int n = rest / p.C;
    float w[KH * KW];
    const T* wc = Wt + (long)c * KH * KW;
#pragma unroll
    for(int ky = 0; ky < KH; ++ky)
#pragma unroll
        for(int kx = 0; kx < KW; ++kx)
            w[ky * KW + kx] = valu_to_f32(wc[dwTap2(ky, kx, KH, KW)]);
    const T* a0 = A + (long)(n * p.C + c) * p.Hi * p.Wi;
    float win[MY][MX];
#pragma unroll
    for(int m = 0; m < MY; ++m)
#pragma unroll
        for(int mm = 0; mm < MX; ++mm)
        {
            // Every load issued (outside dy: the plane's first element), its value masked: the
            // window's loads stay in flight together.
            const int oy = qy - m, ox = qx - mm;
            const bool in = oy >= 0 && oy < p.Hi && ox >= 0 && ox < p.Wi;
            const float v = valu_to_f32(a0[in ? (long)oy * p.Wi + ox : 0]);
            win[m][mm] = in ? v : 0.0f;
        }
    T* d0 = D + (long)(n * p.C + c) * p.Ho * p.Wo;
#pragma unroll
    for(int ry = 0; ry < S; ++ry)
    {
        const int y = S * qy + ry - p.ph;
        if(y < 0 || y >= p.Ho)
            continue;
#pragma unroll
        for(int rx = 0; rx < S; ++rx)
        {
            const int x = S * qx + rx - p.pw;
            if(x < 0 || x >= p.Wo)
                continue;
            float acc = 0.0f;
#pragma unroll
            for(int m = 0; m < MY; ++m)
#pragma unroll
                for(int mm = 0; mm < MX; ++mm)
                    if(ry + S * m < KH && rx + S * mm < KW)
                        acc += w[(ry + S * m) * KW + rx + S * mm] * win[m][mm];
            valu_from_f32(acc, d0[(long)y * p.Wo + x]);
        }
    }
}
} // namespace conv_depthwise_valu
