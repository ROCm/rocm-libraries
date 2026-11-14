// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The by-value argument blocks of the Winograd convolution kernels (CONV_WINOGRAD_FWD,
// CONV_WINOGRAD_BWD). Shared verbatim by the host pack, which fills them, and by the kernels, which
// HIPRTC compiles with this file as an embedded include -- so it uses only builtin types and no
// standard headers.
//
// Every launch is written for a "forward-form" problem: in (N, CI, HI, WI) -> out (N, CO, HO, WO),
// a 3x3 correlation with top/left padding (ph, pw) and explicit output extents, groups.
//   forward:       in = x,  out = y,  CI = C, CO = K, pad = (pre_h, pre_w).
//   backward data: in = dy, out = dx, CI = K, CO = C, pad = (2 - pre_h, 2 - pre_w); the filter
//                  pre-pass rotates the filter by 180 degrees and swaps its K/C roles.
// `flip` (true convolution) reflects the filter in the same pre-pass. Stride 2 (`mode`, below)
// splits the 3x3 filter into its four 2x2 phase sub-filters (zero taps padded) and runs one
// stride-1 2x2 convolution over 4x the channels: phase-gathered input channels forward,
// phase-scattered output channels backward data.

#pragma once

// NOLINTBEGIN(modernize-avoid-c-arrays,modernize-macro-to-enum,readability-avoid-nested-conditional-operator,readability-isolate-declaration)
// HIPRTC compiles this file as an embedded include, with no standard library: C arrays,
// macros and C-style code are deliberate, and its arithmetic types are part of the kernels.

#define K5_NCHW 0
#define K5_NHWC 1
#define K5_FORWARD 0
#define K5_BACKWARD_DATA 1
#define K5_COPY_RANK 4

// The cooperative kernel on CDNA (wave64 MFMA, transform-row layout: 4 x kw waves, bt = 16 x nt
// tiles), shared by the kernel and the host plan so the persistent grid matches what is resident.
// Waves per SIMD (amdgpu_waves_per_eu) on the unified 512-register file (gfx90a and later): a
// lane needs about 16 x mt x nt accumulators (4 bins of mt x nt 16 x 16 atoms) plus 104 others
// (120 for bf16's fp32 transform, except the F(3,2) forward phase gather, @p phases, whose jobs
// need fewer), and 56 more per transform job past the first; a block of 4 x kw waves needs kw per
// SIMD. The compiler holds the registers to 512 / that, so the occupancy is at least that.
constexpr int k5CoopCdnaWpe(int mt, int nt, int kw, bool packed, bool phases)
{
    const int jt = 2 * nt > 4 * kw ? 2 * nt / (4 * kw) : 1;
    const int regs = 16 * mt * nt + (packed || phases ? 104 : 120) + 56 * (jt - 1);
    const int wpe = 512 / regs < 1 ? 1 : 512 / regs > 8 ? 8 : 512 / regs;
    return wpe < kw ? kw : wpe;
}

// LDS bytes of a transform-row-layout block (K5CoopCfg::LDSB): two V buffers of 16 bins x bt
// tiles x 16 halves, or the epilogue's Z passes where those are larger.
constexpr int k5CoopLdsBytes(int nt, int kw, int m, bool wave64)
{
    const int bt = 16 * nt, vb = 16 * bt * 16;
    const int kwp = kw < 2 ? kw : m == 2 ? 2 : 1;
    const int zs = wave64 && 4 * m * 16 * kwp * (bt + 4) * 4 <= 2 * vb * 2 ? bt + 4 : bt;
    const int zb = 4 * m * 16 * kwp * zs;
    return 2 * vb * 2 > zb * 4 ? 2 * vb * 2 : zb * 4;
}

struct K5Args
{
    const void* in; // x (forward) or dy (backward data)
    const void* w; // filter, the graph's K x C/G x 3 x 3 (KCRS for NCHW, KRSC for NHWC)
    void* out; // y (forward) or dx (backward data)
    void* u; // workspace: transformed filter
    void* v; // workspace: transformed input (multipass)
    float* mw; // workspace: Winograd-domain products (multipass)
    int n, groups;
    int ci, co; // forward-form channels, all groups
    int hi, wi, ho, wo;
    int ph, pw; // forward-form top/left padding (may be negative)
    int th, tw; // tiles per image along H and W
    int tiles; // n * th * tw
    int flip; // nonzero: true convolution, the filter is reflected
    // Stride 2 by phases (F(3,2) on 2x2 sub-filters, 16 bins like F(2,3)):
    //   0 stride 1;
    //   1 forward: the input is phase-gathered. Virtual in-channel 4c + p (p = 2 * row parity +
    //     column parity) reads x[c][2r + p/2 - xph][2s + p%2 - xpw]; ci = 4C, ph = pw = 0;
    //   2 backward data: the output is phase-scattered. Virtual out-channel 4c + p at (q, s)
    //     writes dx[c][2q + p/2 - xph][2s + p%2 - xpw]; co = 4C, in = dy with ph = pw = 1.
    int mode;
    int xh, xw; // stride 2: the original x (forward) or dx (backward data) extents
    int xph, xpw; // stride 2: the original top / left padding
    int order; // cooperative kernel item order: 0 K blocks fastest, 1 tile blocks fastest
};

/// Pass 3 of the multipass engine: C[b] (m x n, fp32) = A[b] (m x k) * B[b] (k x n), row-major.
struct K5GemmArgs
{
    const void* a;
    const void* b;
    float* c;
    int m, n, k;
    int lda, ldb, ldc;
    long long sa, sb, sc;
};

/// Backward weights (ConvWinogradWrw.h), F(3,r): x (N, C, HI, WI) and dy (N, K, HO, WO), both
/// packed NCHW (other layouts staged), into dw (K, C/G, 3, 3) at any strides; 3x3 stride 1
/// correlation with top/left padding (ph, pw).
struct K5WrwArgs
{
    const void* x;
    const void* dy;
    void* dw;
    void* u; // workspace: transformed dy, [g][p][K/G][ldt]
    void* v; // workspace: transformed x, [g][p][C/G][ldt]
    float* part; // workspace: fp32 GEMM partials, [split][g][p][K/G][C/G]
    int n, groups;
    int c, k; // channels over all groups
    int hi, wi, ho, wo;
    int ph, pw;
    int th, tw; // dy tiles per image along H and W (ceil(HO / r), ceil(WO / r))
    int tiles; // n * th * tw: the slice's tiles
    int ldt; // tile stride of U and V: tiles rounded up to K5W_LPAD (zero tiles past `tiles`)
    int flip; // nonzero: true convolution, dw is stored reflected
    int splits; // GEMM splits over the tiles
    int splitTiles; // tiles per split, a multiple of K5W_LPAD
    int accumulate; // GEMM: add onto the partials (batch slices after the first)
    long long dwStrides[4]; // (k, c, r, s) element strides of dw
};

/// A strided copy of up to K5_COPY_RANK axes, for staging an operand in or out of the workspace.
struct K5CopyArgs
{
    const void* src;
    void* dst;
    long long dims[K5_COPY_RANK];
    long long srcStrides[K5_COPY_RANK];
    long long dstStrides[K5_COPY_RANK];
    long long total;
    int rank;
    int elementBytes;
};

// NOLINTEND(modernize-avoid-c-arrays,modernize-macro-to-enum,readability-avoid-nested-conditional-operator,readability-isolate-declaration)
