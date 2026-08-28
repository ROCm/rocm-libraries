// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// VALU depthwise convolution kernel cores — RDNA (wave32) track.
//
// Device-side compute cores for the ConvDepthwiseDirect solver. These are the
// four hipconv VALU kernels re-homed into MIOpen as __device__ template cores
// (the original hipconv kernels were __global__; a __global__ cannot be launched
// from the extern "C" __global__ wrapper in miopen_depthwise_valu.cpp, so the
// only change from the validated source is the qualifier and the launch_bounds
// moving onto the wrapper). The compute is byte-for-byte the validated code.
//
// Non-matrix-core depthwise conv. For one channel per group (cpg == 1) a WMMA
// lane would waste a matrix-core slot on a 1-wide dot, so these use plain
// vector-ALU FMA with an fp32 accumulator. NHWC, channel (k) innermost so the
// global loads/stores stay coalesced. K == C (depthwise), weights W[C,KH,KW].
//
// ISA-generic across every RDNA level (RDNA3 gfx110x, RDNA3.5 gfx115x,
// RDNA4 gfx120x): baseline wave32 only — global ld/st, integer address math,
// fp32 FMA, fp16/bf16<->fp32 convert. No WMMA, no tr16, no async-LDS.
//
// The ladder (fastest core for a shape is selected host-side in
// conv_depthwise_direct.cpp):
//   v2_wstrip_core_nhwc     generic W-strip floor; any stride/dilation/kernel size.
//   v3a_microtile_core_nhwc register 2D halo reuse (no LDS); s=d=1, compile-time KH/KW.
//   v4_fused_core_nhwc      LDS block stage + register micro-tile (two-level blocking).
//   v3b_lds_core_nhwc       LDS block stage + tap loop from LDS (register-light; big KH).
//
// The above are channel-last (NHWC / NDHWC) native. The channel-first (NCHW /
// NCDHW) floors mirror the WStrip floor but coalesce on the contiguous width axis
// instead of on channel (the halo/LDS variants stay channel-last):
//   v2_core_nchw       generic channel-first 2D floor; any stride/dilation/size.
//   v2_core_ncdhw      generic channel-first 3D floor.
#ifndef MIOPEN_DEPTHWISE_VALU_KERNELS_HPP
#define MIOPEN_DEPTHWISE_VALU_KERNELS_HPP

#include <hip/hip_runtime.h>

namespace miopen {
namespace conv_depthwise_direct {

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

// ---- v2 — register W-strip: WSTRIP output columns per thread; k innermost ----
// Handles any stride / dilation / padding / kernel size. The universal floor.
template <typename T, int WSTRIP>
__device__ inline void v2_wstrip_core_nhwc(const T* __restrict__ A,
                                           const T* __restrict__ Wt,
                                           T* __restrict__ D,
                                           ValuParams p)
{
    const int Wtiles = (p.Wo + WSTRIP - 1) / WSTRIP;
    long tid         = (long)blockIdx.x * blockDim.x + threadIdx.x;

    const int k  = tid % p.C; // channel — innermost/coalesced
    long rest    = tid / p.C;
    const int wt = rest % Wtiles; // which W-strip
    rest /= Wtiles;
    const int ho = rest % p.Ho;
    const int n  = rest / p.Ho;
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
        if(ih < 0 || ih >= p.Hi)
            continue;
        const long a_row = (((long)n * p.Hi + ih) * p.Wi) * p.C + k;
        for(int kx = 0; kx < p.kw; ++kx)
        {
            const float w = valu_to_f32(wbase[ky * p.kw + kx]); // loaded once per tap
#pragma unroll
            for(int j = 0; j < WSTRIP; ++j)
            {
                const int wo = wo_base + j;
                if(wo >= p.Wo)
                    continue;
                const int iw = wo * p.sw - p.pw + kx * p.dw;
                if(iw < 0 || iw >= p.Wi)
                    continue;
                acc[j] += w * valu_to_f32(A[a_row + (long)iw * p.C]);
            }
        }
    }

#pragma unroll
    for(int j = 0; j < WSTRIP; ++j)
    {
        const int wo = wo_base + j;
        if(wo >= p.Wo)
            continue;
        const long o = (((long)n * p.Ho + ho) * p.Wo + wo) * p.C + k;
        valu_from_f32(acc[j], D[o]);
    }
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
template <typename T>
__device__ inline void
v2_core_nchw(const T* __restrict__ A, const T* __restrict__ Wt, T* __restrict__ D, ValuParams p)
{
    const long tid   = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)p.N * p.C * p.Ho * p.Wo;
    if(tid >= total)
        return;

    const int wo = tid % p.Wo; // contiguous — coalesced across lanes
    long rest    = tid / p.Wo;
    const int ho = rest % p.Ho;
    rest /= p.Ho;
    const int k = rest % p.C; // channel (outer plane)
    const int n = rest / p.C;

    float acc            = 0.0f;
    const T* wbase       = Wt + (long)k * p.kh * p.kw;
    const long a_chanrow = ((long)n * p.C + k) * p.Hi; // ((n*C+k)*Hi)

    for(int ky = 0; ky < p.kh; ++ky)
    {
        const int ih = ho * p.sh - p.ph + ky * p.dh;
        if(ih < 0 || ih >= p.Hi)
            continue;
        const long a_row = (a_chanrow + ih) * p.Wi;
        for(int kx = 0; kx < p.kw; ++kx)
        {
            const int iw = wo * p.sw - p.pw + kx * p.dw;
            if(iw < 0 || iw >= p.Wi)
                continue;
            acc += valu_to_f32(wbase[ky * p.kw + kx]) * valu_to_f32(A[a_row + iw]);
        }
    }

    const long o = (((long)n * p.C + k) * p.Ho + ho) * p.Wo + wo;
    valu_from_f32(acc, D[o]);
}

// ---- v2 (3D) — register W-strip over NDHWC; the universal 3D floor ----
// The mechanical depth extension of v2_wstrip_core_nhwc: one extra output-depth (do_)
// level in the thread-index decompose and one extra tap loop (kz -> id) wrapping
// the 2D tap nest, with depth folded into the NDHWC address math. Register
// footprint (acc[WSTRIP]) is identical to the 2D core. Any stride/dilation/pad/
// kernel size on all three spatial axes. Weights W[C, kd, kh, kw].
template <typename T, int WSTRIP>
__device__ inline void v2_wstrip_core_ndhwc(const T* __restrict__ A,
                                            const T* __restrict__ Wt,
                                            T* __restrict__ D,
                                            ValuParams p)
{
    const int Wtiles = (p.Wo + WSTRIP - 1) / WSTRIP;
    long tid         = (long)blockIdx.x * blockDim.x + threadIdx.x;

    const int k  = tid % p.C; // channel — innermost/coalesced
    long rest    = tid / p.C;
    const int wt = rest % Wtiles; // which W-strip
    rest /= Wtiles;
    const int ho = rest % p.Ho;
    rest /= p.Ho;
    const int do_ = rest % p.Do; // output depth
    const int n   = rest / p.Do;
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
        if(id < 0 || id >= p.Di)
            continue;
        for(int ky = 0; ky < p.kh; ++ky)
        {
            const int ih = ho * p.sh - p.ph + ky * p.dh;
            if(ih < 0 || ih >= p.Hi)
                continue;
            const long a_row = ((((long)n * p.Di + id) * p.Hi + ih) * p.Wi) * p.C + k;
            for(int kx = 0; kx < p.kw; ++kx)
            {
                const float w = valu_to_f32(wbase[(kz * p.kh + ky) * p.kw + kx]);
#pragma unroll
                for(int j = 0; j < WSTRIP; ++j)
                {
                    const int wo = wo_base + j;
                    if(wo >= p.Wo)
                        continue;
                    const int iw = wo * p.sw - p.pw + kx * p.dw;
                    if(iw < 0 || iw >= p.Wi)
                        continue;
                    acc[j] += w * valu_to_f32(A[a_row + (long)iw * p.C]);
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
        const long o = ((((long)n * p.Do + do_) * p.Ho + ho) * p.Wo + wo) * p.C + k;
        valu_from_f32(acc[j], D[o]);
    }
}

// ---- v2 (NCDHW) — channel-outer 3D floor; W contiguous, coalesced on W ----
// The channel-first mirror of v2_wstrip_core_ndhwc (and the depth extension of
// v2_core_nchw): one output element per thread, wo innermost for coalescing, an
// extra output-depth (do_) decode level and a kz->id tap loop wrapping the 2D
// nest. NCDHW address math. Any stride/dilation/pad/kernel size on all three
// axes. Weights W[C,kd,kh,kw] contiguous.
template <typename T>
__device__ inline void
v2_core_ncdhw(const T* __restrict__ A, const T* __restrict__ Wt, T* __restrict__ D, ValuParams p)
{
    const long tid   = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)p.N * p.C * p.Do * p.Ho * p.Wo;
    if(tid >= total)
        return;

    const int wo = tid % p.Wo; // contiguous — coalesced across lanes
    long rest    = tid / p.Wo;
    const int ho = rest % p.Ho;
    rest /= p.Ho;
    const int do_ = rest % p.Do; // output depth
    rest /= p.Do;
    const int k = rest % p.C; // channel (outer plane)
    const int n = rest / p.C;

    float acc         = 0.0f;
    const T* wbase    = Wt + (long)k * p.kd * p.kh * p.kw;
    const long a_chan = (long)n * p.C + k; // (n*C+k) — channel plane base

    for(int kz = 0; kz < p.kd; ++kz)
    {
        const int id = do_ * p.sd - p.pd + kz * p.dd;
        if(id < 0 || id >= p.Di)
            continue;
        for(int ky = 0; ky < p.kh; ++ky)
        {
            const int ih = ho * p.sh - p.ph + ky * p.dh;
            if(ih < 0 || ih >= p.Hi)
                continue;
            const long a_row = (((a_chan * p.Di + id) * p.Hi) + ih) * p.Wi;
            for(int kx = 0; kx < p.kw; ++kx)
            {
                const int iw = wo * p.sw - p.pw + kx * p.dw;
                if(iw < 0 || iw >= p.Wi)
                    continue;
                acc +=
                    valu_to_f32(wbase[(kz * p.kh + ky) * p.kw + kx]) * valu_to_f32(A[a_row + iw]);
            }
        }
    }

    const long o = ((((long)n * p.C + k) * p.Do + do_) * p.Ho + ho) * p.Wo + wo;
    valu_from_f32(acc, D[o]);
}

// ---- v3a microtile: register 2D halo reuse (no LDS, no barrier) ----
// Each thread emits a TH×TW output tile and caches the (TH+KH-1)×(TW+KW-1) input
// patch in registers. Bakes s=d=1 and compile-time KH/KW so the tap loops and the
// register patch/accumulators fully unroll into named registers.
template <typename T, int KH, int KW, int TH, int TW>
__device__ inline void v3a_microtile_core_nhwc(const T* __restrict__ A,
                                               const T* __restrict__ Wt,
                                               T* __restrict__ D,
                                               ValuParams p)
{
    const int Htiles = (p.Ho + TH - 1) / TH;
    const int Wtiles = (p.Wo + TW - 1) / TW;
    long tid         = (long)blockIdx.x * blockDim.x + threadIdx.x;

    const int k  = tid % p.C; // channel innermost
    long rest    = tid / p.C;
    const int wt = rest % Wtiles;
    rest /= Wtiles;
    const int ht = rest % Htiles;
    const int n  = rest / Htiles;
    if(n >= p.N)
        return;

    const int ho0 = ht * TH, wo0 = wt * TW;
    const int ih0 = ho0 - p.ph, iw0 = wo0 - p.pw; // s=d=1
    const int PH = TH + KH - 1, PW = TW + KW - 1;

    // register input patch, loaded once per thread per channel
    float patch[PH][PW];
#pragma unroll
    for(int py = 0; py < PH; ++py)
    {
        const int ih    = ih0 + py;
        const bool ihv  = (ih >= 0 && ih < p.Hi);
        const long arow = ihv ? ((((long)n * p.Hi + ih) * p.Wi) * p.C + k) : 0;
#pragma unroll
        for(int px = 0; px < PW; ++px)
        {
            const int iw = iw0 + px;
            patch[py][px] =
                (ihv && iw >= 0 && iw < p.Wi) ? valu_to_f32(A[arow + (long)iw * p.C]) : 0.0f;
        }
    }

    // weights once per thread
    float w[KH][KW];
    const T* wb = Wt + (long)k * KH * KW;
#pragma unroll
    for(int ky = 0; ky < KH; ++ky)
#pragma unroll
        for(int kx = 0; kx < KW; ++kx)
            w[ky][kx] = valu_to_f32(wb[ky * KW + kx]);

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
            const long o = (((long)n * p.Ho + ho) * p.Wo + wo) * p.C + k;
            valu_from_f32(acc[ty][tx], D[o]);
        }
    }
}

// ---- v4 fused: LDS block stage + register micro-tile ----
// The workgroup stages a [BH+KH-1, BW+KW-1, BK] halo'd block into LDS once, then
// each thread reads an (RH+KH-1)×(RW+KW-1) patch from LDS into registers and
// accumulates RH×RW outputs. threads/block = (BH/RH)*(BW/RW)*BK.
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

    const int LH = BH + KH - 1, LW = BW + KW - 1;
    __shared__ T tile[LH * LW * BK];

    const int Hblocks = (p.Ho + BH - 1) / BH;
    const int Wblocks = (p.Wo + BW - 1) / BW;
    const int Kblocks = (p.C + BK - 1) / BK;

    long blk     = blockIdx.x;
    const int kb = blk % Kblocks;
    blk /= Kblocks;
    const int wb = blk % Wblocks;
    blk /= Wblocks;
    const int hb = blk % Hblocks;
    const int n  = blk / Hblocks; // grid sized so n<N

    const int ho_base = hb * BH, wo_base = wb * BW, k_base = kb * BK;
    const int ih0 = ho_base - p.ph, iw0 = wo_base - p.pw; // s=d=1

    T zero;
    valu_from_f32(0.0f, zero);

    // cooperative halo stage into LDS
    const int nthreads = blockDim.x;
    for(int idx = threadIdx.x; idx < LH * LW * BK; idx += nthreads)
    {
        const int kc = idx % BK;
        const int t  = idx / BK;
        const int px = t % LW;
        const int py = t / LW;
        const int ih = ih0 + py, iw = iw0 + px, kk = k_base + kc;
        const bool in = ih >= 0 && ih < p.Hi && iw >= 0 && iw < p.Wi && kk < p.C;
        tile[idx]     = in ? A[(((long)n * p.Hi + ih) * p.Wi + iw) * p.C + kk] : zero;
    }
    __syncthreads();

    const int tid    = threadIdx.x;
    const int klocal = tid % BK;
    int rest         = tid / BK;
    const int wtile  = rest % (BW / RW);
    const int htile  = rest / (BW / RW);
    const int oy = htile * RH, ox = wtile * RW;
    const int k = k_base + klocal;

    const int PHr = RH + KH - 1, PWr = RW + KW - 1;
    float patch[PHr][PWr];
#pragma unroll
    for(int py = 0; py < PHr; ++py)
#pragma unroll
        for(int px = 0; px < PWr; ++px)
            patch[py][px] = valu_to_f32(tile[(((oy + py) * LW) + (ox + px)) * BK + klocal]);

    float w[KH][KW];
    if(k < p.C)
    {
        const T* wbp = Wt + (long)k * KH * KW;
#pragma unroll
        for(int ky = 0; ky < KH; ++ky)
#pragma unroll
            for(int kx = 0; kx < KW; ++kx)
                w[ky][kx] = valu_to_f32(wbp[ky * KW + kx]);
    }
    else
    {
#pragma unroll
        for(int ky = 0; ky < KH; ++ky)
#pragma unroll
            for(int kx = 0; kx < KW; ++kx)
                w[ky][kx] = 0.0f;
    }

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
            const long o = (((long)n * p.Ho + ho) * p.Wo + wo) * p.C + k;
            valu_from_f32(acc[ty][tx], D[o]);
        }
    }
}

// ---- v3b lds-direct: LDS block stage, tap loop reads from LDS ----
// Register-light (no materialised patch) so it survives large KH/KW where v4's
// register patch would spill. LDS tap traffic is KH*KW loads/output, but LDS
// bandwidth >> DRAM, so this is the right tool for 9x9 depthwise.
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

    const int LH = BH + KH - 1, LW = BW + KW - 1;
    __shared__ T tile[LH * LW * BK];

    const int Hblocks = (p.Ho + BH - 1) / BH;
    const int Wblocks = (p.Wo + BW - 1) / BW;
    const int Kblocks = (p.C + BK - 1) / BK;

    long blk     = blockIdx.x;
    const int kb = blk % Kblocks;
    blk /= Kblocks;
    const int wb = blk % Wblocks;
    blk /= Wblocks;
    const int hb = blk % Hblocks;
    const int n  = blk / Hblocks;

    const int ho_base = hb * BH, wo_base = wb * BW, k_base = kb * BK;
    const int ih0 = ho_base - p.ph, iw0 = wo_base - p.pw;

    T zero;
    valu_from_f32(0.0f, zero);
    const int nthreads = blockDim.x;
    for(int idx = threadIdx.x; idx < LH * LW * BK; idx += nthreads)
    {
        const int kc = idx % BK;
        const int t  = idx / BK;
        const int px = t % LW;
        const int py = t / LW;
        const int ih = ih0 + py, iw = iw0 + px, kk = k_base + kc;
        const bool in = ih >= 0 && ih < p.Hi && iw >= 0 && iw < p.Wi && kk < p.C;
        tile[idx]     = in ? A[(((long)n * p.Hi + ih) * p.Wi + iw) * p.C + kk] : zero;
    }
    __syncthreads();

    const int tid    = threadIdx.x;
    const int klocal = tid % BK;
    int rest         = tid / BK;
    const int wtile  = rest % (BW / RW);
    const int htile  = rest / (BW / RW);
    const int oy = htile * RH, ox = wtile * RW;
    const int k = k_base + klocal;

    float acc[RH][RW];
#pragma unroll
    for(int ty = 0; ty < RH; ++ty)
#pragma unroll
        for(int tx = 0; tx < RW; ++tx)
            acc[ty][tx] = 0.0f;

    const T* wbp = (k < p.C) ? Wt + (long)k * KH * KW : nullptr;
#pragma unroll
    for(int ky = 0; ky < KH; ++ky)
#pragma unroll
        for(int kx = 0; kx < KW; ++kx)
        {
            const float wv = wbp ? valu_to_f32(wbp[ky * KW + kx]) : 0.0f;
#pragma unroll
            for(int ty = 0; ty < RH; ++ty)
#pragma unroll
                for(int tx = 0; tx < RW; ++tx)
                    acc[ty][tx] +=
                        wv *
                        valu_to_f32(tile[(((oy + ty + ky) * LW) + (ox + tx + kx)) * BK + klocal]);
        }

    if(k >= p.C)
        return;
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
            const long o = (((long)n * p.Ho + ho) * p.Wo + wo) * p.C + k;
            valu_from_f32(acc[ty][tx], D[o]);
        }
    }
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
          int RW>
__device__ inline void v3b_lds_core_ndhwc(const T* __restrict__ A,
                                          const T* __restrict__ Wt,
                                          T* __restrict__ D,
                                          ValuParams p)
{
    static_assert(BD % RD == 0, "v3b_lds_3d: BD must be divisible by RD");
    static_assert(BH % RH == 0, "v3b_lds_3d: BH must be divisible by RH");
    static_assert(BW % RW == 0, "v3b_lds_3d: BW must be divisible by RW");
    static_assert(
        (BD + KD - 1) * (BH + KH - 1) * (BW + KW - 1) * BK * static_cast<int>(sizeof(T)) <= 65536,
        "v3b_lds_3d: LDS tile exceeds 64 KB");

    const int LD = BD + KD - 1, LH = BH + KH - 1, LW = BW + KW - 1;
    __shared__ T tile[LD * LH * LW * BK]; // [pz][py][px][kc], kc innermost

    const int Dblocks = (p.Do + BD - 1) / BD;
    const int Hblocks = (p.Ho + BH - 1) / BH;
    const int Wblocks = (p.Wo + BW - 1) / BW;
    const int Kblocks = (p.C + BK - 1) / BK;

    long blk     = blockIdx.x;
    const int kb = blk % Kblocks;
    blk /= Kblocks;
    const int wb = blk % Wblocks;
    blk /= Wblocks;
    const int hb = blk % Hblocks;
    blk /= Hblocks;
    const int db = blk % Dblocks;
    const int n  = blk / Dblocks; // grid sized so n < N

    const int do_base = db * BD, ho_base = hb * BH, wo_base = wb * BW, k_base = kb * BK;
    const int id0 = do_base - p.pd, ih0 = ho_base - p.ph, iw0 = wo_base - p.pw; // s=d=1

    T zero;
    valu_from_f32(0.0f, zero);
    const int nthreads = blockDim.x;
    const int tile_sz  = LD * LH * LW * BK;
    for(int idx = threadIdx.x; idx < tile_sz; idx += nthreads)
    {
        const int kc = idx % BK;
        int t        = idx / BK;
        const int px = t % LW;
        t /= LW;
        const int py = t % LH;
        const int pz = t / LH;
        const int id = id0 + pz, ih = ih0 + py, iw = iw0 + px, kk = k_base + kc;
        const bool in =
            id >= 0 && id < p.Di && ih >= 0 && ih < p.Hi && iw >= 0 && iw < p.Wi && kk < p.C;
        tile[idx] = in ? A[((((long)n * p.Di + id) * p.Hi + ih) * p.Wi + iw) * p.C + kk] : zero;
    }
    __syncthreads();

    const int tid    = threadIdx.x;
    const int klocal = tid % BK;
    int rest         = tid / BK;
    const int wtile  = rest % (BW / RW);
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

    const T* wbp = (k < p.C) ? Wt + (long)k * KD * KH * KW : nullptr;
#pragma unroll
    for(int kz = 0; kz < KD; ++kz)
#pragma unroll
        for(int ky = 0; ky < KH; ++ky)
#pragma unroll
            for(int kx = 0; kx < KW; ++kx)
            {
                const float wv = wbp ? valu_to_f32(wbp[(kz * KH + ky) * KW + kx]) : 0.0f;
#pragma unroll
                for(int tz = 0; tz < RD; ++tz)
#pragma unroll
                    for(int ty = 0; ty < RH; ++ty)
#pragma unroll
                        for(int tx = 0; tx < RW; ++tx)
                            acc[tz][ty][tx] +=
                                wv *
                                valu_to_f32(tile[((((oz + tz + kz) * LH) + (oy + ty + ky)) * LW +
                                                  (ox + tx + kx)) *
                                                     BK +
                                                 klocal]);
            }

    if(k >= p.C)
        return;
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
                const long o = ((((long)n * p.Do + do_) * p.Ho + ho) * p.Wo + wo) * p.C + k;
                valu_from_f32(acc[tz][ty][tx], D[o]);
            }
        }
    }
}

} // namespace conv_depthwise_direct
} // namespace miopen

#endif // MIOPEN_DEPTHWISE_VALU_KERNELS_HPP
