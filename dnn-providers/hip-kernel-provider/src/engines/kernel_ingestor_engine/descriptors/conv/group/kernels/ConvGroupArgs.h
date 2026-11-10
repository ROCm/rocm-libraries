// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument blocks of the small-group convolution kernels (K4). Shared verbatim by the host pack,
// which fills them, and by the kernels, which HIPRTC compiles with this file as an embedded
// include: builtin types only, no standard headers.
#pragma once

/// One small-group convolution as a gather: out[n][oh][ow][g cg + o] = sum over taps (r, s) and
/// the group's cg channels of in[n][oh sh - ph + r][ow sw - pw + s][g cg + i] w16[...], both
/// activations channels-last. Forward gathers x into y; backward data (unit stride) gathers dy
/// into dx with the filter reflected and its channel roles swapped (in ConvGroupExpand). The
/// filter is pre-expanded per 16-channel block: w16[block][tap][n 16][k 16], zero where n and k
/// sit in different groups.
struct K4Args
{
    const void* in;
    const void* w;
    void* out;
    int n, c; // images, channels of in and out (all groups)
    int hi, wi; // in extents
    int ho, wo; // out extents
    int ph, pw; // the gather's leading padding
    int cg; // channels a group
    int padding0;
};

/// The filter expansion: w16[block][tap][n][k] from the graph's filter w[k][c][r][s] (any
/// strides, element (k, c, r, s) at k sK + c sC + r sR + s sS), cg channels a group each way.
/// Forward: n the output channel, k the input channel. Backward data: n the input channel (dx),
/// k the output channel (dy), the tap reflected. reflect (true convolution) reflects it again.
struct K4ExpandArgs
{
    const void* src;
    void* dst;
    long long sK, sC, sR, sS;
    int c; // channels (all groups)
    int cg;
    int r, s; // filter extent
    int backward;
    int reflect;
    int elementBytes;
    int padding0;
};
