// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument blocks of the stride-1 halo-tile direct convolution kernels (K2). Shared verbatim by
// the host pack, which fills them, and by the kernels, which HIPRTC compiles with this file as an
// embedded include: builtin types only, no standard headers.
#pragma once

// NOLINTBEGIN(modernize-avoid-c-arrays,readability-use-std-min-max)
// HIPRTC compiles this file as an embedded include, with no standard library: C arrays,
// macros and C-style code are deliberate, and its arithmetic types are part of the kernels.

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
    /// Block tile: output rows, columns, depth planes. Run-tile configs (RUN): tw is the strip
    /// width, th the virtual rows a plane, tilesW the tiles a strip and tilesH the strips.
    int th, tw, ds;
    int tilesW, tilesH, tilesD;
    /// Split-C (SPLIT configs): grid.z = splits, each block reduces `cps` channel chunks and
    /// writes fp32 partials ws[z][output element] for ConvDirectReduce to add in order.
    int splits, cps;
    /// XCDs the device deals workgroups to round-robin in launch order (each with its own L2).
    /// Above 1 (and with one split), the kernels remap their block index so each XCD runs runs of
    /// `xcdRun` consecutive tiles (column blocks fastest): neighbouring halos meet in one L2,
    /// while the grid as a whole still advances in launch order.
    int xcds;
    int xcdRun;
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

/// Most tiles of 16 fn x ka formatted elements one block of the rows weight pre-pass formats per
/// step (k2rFormatWeights, which takes fewer when the grid has more blocks).
constexpr int K2R_FORMAT_TILES = 4;
/// The rows pre-pass grid keeps at least this many blocks (one tile each) before batching tiles.
constexpr int K2R_FORMAT_MIN_BLOCKS = 256;

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

/// Run tiles (RUN configs): the most pixels a block of `bm` consecutive pixels of a strip of
/// `ws` columns stages: the rows it can cover plus r - 1, each ws + s - 1 pixels wide.
constexpr int k2RunHalo(int bm, int r, int s, int ws)
{
    return ((ws - 1 + bm + ws - 1) / ws + r - 1) * (ws + s - 1);
}

/// Channels-first halo tiles (CF 1) are at least K2_CF_MIN_TW columns wide where the output is,
/// so a channel row's halo run is long and contiguous; their halo bound is 3/2 of k2HaloMax so
/// the wide, short tiles fit.
constexpr int K2_CF_MIN_TW = 32;
constexpr int k2CfHaloMax(int bm, int r, int s)
{
    return k2HaloMax(bm, r, s) * 3 / 2;
}

// NOLINTEND(modernize-avoid-c-arrays,readability-use-std-min-max)
