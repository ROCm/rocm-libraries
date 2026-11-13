// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument blocks of the stride-1 halo-tile direct convolution kernels (K2). Shared verbatim by
// the host pack, which fills them, and by the kernels, which HIPRTC compiles with this file as an
// embedded include: builtin types only, no standard headers.
#pragma once

/// The convolution in its forward form: x (forward x, backward data dy) channels-last
/// N D H W c, y (forward y, backward data dx) channels-last N D H W k, and the weights
/// pre-formatted by ConvDirectFormat. Shapes are runtime; only the filter extent and the tile
/// are compile-time.
struct K2Args
{
    const void* x;
    const void* w; ///< formatted weights
    void* y;
    int n, c, k; ///< c: reduction channels, k: output channels
    int di, hi, wi;
    int dout, ho, wo;
    int pd, ph, pw; ///< leading padding of the forward form
    int kpad, ca; ///< formatted weight columns; reduction chunks of `ka` in the formatted weights
    int th, tw, ds; ///< block tile: output rows, columns, depth planes
    int tilesW, tilesH, tilesD;
    /// Split-C (SPLIT configs): grid.z = splits, each block reduces `cps` channel chunks and
    /// writes fp32 partials ws[z][output element] for ConvDirectReduce to add in order.
    int splits, cps;
    float* ws;
};

/// The weight pre-pass: wf[((tap * cpad / ka + r / ka) * kpad + n) * ka + r % ka], zero past the
/// real channels, read from the graph's filter through its element strides (k, c, t, r, s).
/// Backward data rotates the taps and swaps the channel roles (columns n are input channels, the
/// reduction r runs over output channels); true convolution (flip) reflects the taps, which
/// cancels the rotation. Three-bf16 (ConvDirectFormatSplit) reads fp32 and writes bf16 with a 3x
/// reduction: [hi | lo | hi] of each weight.
struct K2FormatArgs
{
    const void* w;
    void* wf;
    long long wStrides[5]; ///< k, c, t, r, s of the forward filter K C T R S
    int k0, c0; ///< the forward convolution's output and input channels
    int kd, r, s; ///< filter extent
    int kpad, cpad, ka;
    int backward;
    int flip;
};

/// Three-bf16 input pre-pass: channels-last fp32 x (pixels x c) to bf16 (pixels x 3c), [hi | hi | lo].
struct K2SplitArgs
{
    const void* x;
    void* xs;
    long long pixels;
    int c;
    int padding0;
};

/// Split-C reduce: y[i] = sum over z in order of ws[z * count + i], rounded once.
struct K2ReduceArgs
{
    const float* ws;
    void* y;
    long long count;
    int splits;
    int padding0;
};

/// The largest halo (pixels) of a block of `bm` output pixels the host planner may choose
/// (tile widths 8 to 16, or narrower maps; the planner shrinks the height to fit). Sizes the
/// kernels' LDS; the planner must keep (th + r - 1) * (tw + s - 1) within it.
constexpr int k2HaloMax(int bm, int r, int s)
{
    int best = 0;
    for(int tw = 8; tw <= 16; ++tw)
    {
        int th = bm / tw;
        if(th < 1)
        {
            th = 1;
        }
        const int h = (th + r - 1) * (tw + s - 1);
        if(h > best)
        {
            best = h;
        }
    }
    return best;
}
