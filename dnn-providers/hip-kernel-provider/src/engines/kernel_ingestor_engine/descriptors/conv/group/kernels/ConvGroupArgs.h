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
    int padding0;
};
