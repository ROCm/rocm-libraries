// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument blocks of the small-group convolution kernels (K4). Shared verbatim by the host pack,
// which fills them, and by the kernels, which HIPRTC compiles with this file as an embedded
// include: builtin types only, no standard headers.
#pragma once

/// One small-group convolution as a gather over channels-last activations:
///   out[n][oh osh + ooh][ow osw + oow][g cg + o]
///     = sum over taps (r, s) < (R, S) and the group's cg channels i of
///       in[n][oh ST - ph + r][ow ST - pw + s][g cg + i] w16[block][r S + s][o][i].
/// Forward gathers x into y (osh = osw = 1, no offsets). Backward data gathers dy into dx at unit
/// stride, by stride phases otherwise: phase (ooh, oow) writes dx rows ooh, ooh + osh, ... from
/// the taps that reach them, each its own expanded filter (ConvGroupExpand). The filter is
/// pre-expanded per 16-channel block: w16[block][tap][n 16][k 16], zero where n and k sit in
/// different groups.
struct K4Args
{
    const void* in;
    const void* w;
    void* out;
    int n, c; // images, channels of in and out (all groups)
    int hi, wi; // in extents
    int ho, wo; // the gather's output extents (this phase's)
    int ph, pw; // the gather's leading padding
    int outH, outW; // out extents
    int osh, osw; // out row and column step
    int ooh, oow; // out row and column offset
    int cg; // channels a group
    int padding0;
};

/// The filter expansion: w16[block][tap][n][k] for taps (q, u) < (rc, sc) from the graph's filter
/// w[k][c][r][s] (any strides: element (k, c, r, s) at k sK + c sC + r sR + s sS), cg channels a
/// group each way. Tap (q, u) reads filter tap (rBase + rStep q, sBase + sStep u), zero outside
/// the filter; reflect (true convolution) reflects that again. swap: n is the input channel and k
/// the output channel (backward data), else the other way round.
struct K4ExpandArgs
{
    const void* src;
    void* dst;
    long long sK, sC, sR, sS;
    int c; // channels (all groups)
    int cg;
    int r, s; // filter extent
    int rc, sc; // expanded taps
    int rBase, rStep, sBase, sStep;
    int swap;
    int reflect;
    int elementBytes;
    int padding0;
};
