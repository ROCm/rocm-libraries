// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument blocks of the small-group convolution kernels (K4). Shared verbatim by the host pack,
// which fills them, and by the kernels, which HIPRTC compiles with this file as an embedded
// include: builtin types only, no standard headers.
#pragma once

/// One small-group convolution as a gather over channels-last activations:
///   out[n][od osd + ood][oh osh + ooh][ow osw + oow][g cg + o]
///     = sum over taps (t, r, s) < (T, R, S) and the group's cg channels i of
///       in[n][od sd - pd + t][oh ST - ph + r][ow ST - pw + s][g cg + i]
///       w16[block][(t R + r) S + s][o][i].
/// Forward gathers x into y (unit output steps, no offsets). Backward data gathers dy into dx at
/// unit stride, by stride phases otherwise: phase (ood, ooh, oow) writes every osd-th dx plane,
/// osh-th row and osw-th column from its offset, from the taps that reach them, each phase its
/// own expanded filter (ConvGroupExpand). The filter is pre-expanded per 16-channel block:
/// w16[block][tap][n 16][k 16], zero where n and k sit in different groups. 2-D convolutions are
/// the depth-1 case (one plane, one depth tap).
struct K4Args
{
    const void* in;
    const void* w;
    void* out;
    int n, c; // images, channels of in and out (all groups)
    int di, hi, wi; // in extents
    int dout, ho, wo; // the gather's output extents (this phase's)
    int sd; // the gather's depth stride (rows and columns: the compile-time ST)
    int pd, ph, pw; // the gather's leading padding
    int outD, outH, outW; // out extents
    int osd, osh, osw; // out plane, row and column step
    int ood, ooh, oow; // out plane, row and column offset
    int cg; // channels a group
    // Phase-fused backward data at stride 2 (k4ConvMmaPhased): the row and column phases' gather
    // paddings, and the element stride between the four phases' expanded filters (phase
    // 2 row phase + column phase).
    int fph[2];
    int fpw[2];
    long long wps;
    // Three-bf16 (CONV_GROUP_P 3): the element stride between the input's and the expanded
    // filter's three parts (hi | hi' | lo and hi | lo | hi'); unused otherwise.
    long long xps;
    long long wpt;
};

/// The filter expansion: w16[block][tap][n][k] for taps (v, q, u) < (tc, rc, sc) from the
/// graph's filter w[k][c][t][r][s] (any strides: element (k, c, t, r, s) at k sK + c sC + t sT +
/// r sR + s sS), cg channels a group each way. Tap (v, q, u) reads filter tap (tBase + tStep v,
/// rBase + rStep q, sBase + sStep u), zero outside the filter; reflect (true convolution)
/// reflects that again. swap: n is the input channel and k the output channel (backward data),
/// else the other way round.
struct K4ExpandArgs
{
    const void* src;
    void* dst;
    long long sK, sC, sT, sR, sS;
    int c; // channels (all groups)
    int cg;
    int t, r, s; // filter extent
    int tc, rc, sc; // expanded taps
    int tBase, tStep, rBase, rStep, sBase, sStep;
    int swap;
    int reflect;
    int elementBytes;
    // Wide groups (16 < cg <= 64): the input (and output) 16-channel blocks a group spans,
    // padded; the expansion is then dense per group: w16[group][out block][tap][in block][n][k],
    // zero past cg. 0: the block-diagonal layout above.
    int nbi;
};

/// Backward weights (CONV_GROUP_WRW): per group, filter tap (t, r, s) and channels k, c of it,
///   dw[g cg + k][c][t][r][s] = sum over images and dy positions of
///     dy[n][g cg + k][od][oh][ow] x[n][g cg + c][od sd - pd + t][oh ST - ph + r][ow ST - pw + s].
/// x and dy are read in place, both packed channels-last or both packed channels-first (the
/// kernel's CF). The dy positions run in tiles of TH rows x TW columns of one (image, output
/// plane); the grid is (16 CB-channel tiles) x (depth taps) x splits, a split a contiguous run of
/// `tiles` / `splits` tiles. A workgroup writes its fp32 partials to its split's slot:
/// [split][g][k][c][tap], taps (t, r, s) flattened; ConvGroupWrwReduce adds the slots in a fixed
/// order into dw (deterministic, no atomics).
struct K4WrwArgs
{
    const void* x;
    const void* dy;
    float* partial;
    int n, c, cg; // images, channels (all groups, x and dy alike), channels a group
    int di, hi, wi; // x extents
    int dout, ho, wo; // dy extents
    int sd, pd, ph, pw; // depth stride (rows and columns: the compile-time ST), leading padding
    int ft; // filter depth (a grid axis)
    int tilesW, tilesH; // column and row tiles of an output plane
    int splits;
    long long tiles; // images x output planes x row tiles x column tiles
    long long partialElems; // dw elements: one slot of the partials
};

/// The fixed-order sum of the partial slots into dw (element (k, c, t, r, s) at k sK + c sC +
/// t sT + r sR + s sS, c within the group), in dw's element type; reflect: true convolution,
/// the taps reflected.
struct K4WrwReduceArgs
{
    const float* partial;
    void* dw;
    long long partialElems;
    long long sK, sC, sT, sR, sS;
    int slots;
    int cg;
    int t, r, s;
    int reflect;
};

/// The reduce's slot groups: the slot count rounded up to a power of two, at most 16; a
/// 256-thread block reduces 256 / groups dw elements.
#define K4_WRW_REDUCE_GROUPS 16
#ifdef __HIPCC_RTC__
#define K4_HD __device__
#elif defined(__HIPCC__)
#define K4_HD __host__ __device__
#else
#define K4_HD
#endif
K4_HD inline int k4WrwReduceGroups(int slots)
{
    int g = 1;
    while(g < slots && g < K4_WRW_REDUCE_GROUPS)
        g *= 2;
    return g;
}
