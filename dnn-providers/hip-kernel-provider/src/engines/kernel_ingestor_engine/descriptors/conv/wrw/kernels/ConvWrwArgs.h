// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument blocks of the matrix-core backward-weights kernels (K3). Shared verbatim by the host
// pack, which fills them, and by the kernels, which HIPRTC compiles with this file as an
// embedded include: builtin types only, no standard headers.
#pragma once

// NOLINTBEGIN(modernize-avoid-c-arrays,modernize-macro-to-enum)
// HIPRTC compiles this file as an embedded include, with no standard library: C arrays,
// macros and C-style code are deliberate, and its arithmetic types are part of the kernels.

/// Backward weights: dw[g][k][c][t][r][s] = sum over images and output positions of
/// dy[n][g k][od][oh][ow] * x[n][g c][od sd + t dd - pd][oh sh + r dh - ph][ow sw + s dw - pw].
/// x and dy are read in place through their element strides (one of the two packed layouts
/// the kernel is compiled for); the kernel writes fp32 partials, [split][g][tap][k][c], which
/// ConvWrwReduce adds in split order into dw. Shapes, padding and the depth/height filter,
/// stride and dilation are runtime; specialized loops fix the filter width, its stride and
/// dilation at compile time. Loops 8–12 keep all filter geometry runtime; only their tile is fixed.
struct K3Args
{
    const void* x;
    const void* dy;
    float* partial;
    int n, c, k, groups; ///< c and k over all groups
    int di, hi, wi; ///< input
    int dout, ho, wo; ///< output
    int ft, fr, fs; ///< filter
    int sd, sh, sw;
    int pd, ph, pw;
    int dd, dh, dw;
    /// Reduction steps: rows (n, od, oh) times width chunks of the kernel's BQ positions,
    /// split evenly over `splits` (the grid's third axis).
    int chunks; ///< width chunks per output row
    int splits;
    long long steps; ///< rows * chunks; loops 9–12: ceil(n * dout * ho * wo / BQ)
    long long stepsPerSplit;
    /// Row-ring kernels: steps per track of the height classes rho < trackLong; the other
    /// classes' tracks have trackLen - 1 (at most ceil(ho / dh) + fr - 1; K3Track).
    int trackLen;
    int trackLong;
    /// Direct loop: row segments per image and rows per segment. Packed rows (loop 7): the
    /// virtual row width (columns a step row takes) and the output rows a step packs.
    int segs;
    int segRows;
    int walk; ///< 0 cells (tile fastest), 1 stream (tile-major runs)
    /// Element strides: x (n, c, d, h, w), dy (n, k, d, h, w).
    long long xs[5];
    long long ys[5];
    /// Elements of one image of x and dy (buffer loops: 32-bit offsets; loops 9–12: 64-bit globals).
    long long xImage;
    long long yImage;
    /// dw elements per split of the partials: groups * taps * (k / groups) * (c / groups).
    long long partialElems;
    /// Nonzero at one split: the kernel writes dw itself in its element type; `partial` is
    /// unused. DWS 1 requires contiguous channels (or one channel); the bounded DWS 2
    /// tap-tiled epilogue also supports channels-first multi-tap filters.
    int dwStore;
    int flip; ///< nonzero: true convolution, dw written with its taps reflected
    void* dwOut; ///< dw (`dw` is the width dilation)
    /// dw element strides: k and flattened tap; c is 1, or `taps` for the DWS 2 epilogue.
    long long dwk;
    long long dwtap;
    /// XCD-contiguous block order (CONV_K3_XCD): the device's XCDs; 1 or 0 keeps the launch order.
    int xcds;
    int padding1;
};

/// Adds the splits' fp32 partials in split order and writes dw in its element type and layout.
struct K3ReduceArgs
{
    const float* partial;
    void* dw;
    long long elems; ///< dw elements
    long long partialElems; ///< per split
    int splits;
    int groups;
    int kg; ///< output channels per group
    int cg; ///< input channels per group
    int taps; ///< ft * fr * fs
    int flip; ///< nonzero: true convolution, dw written with its taps reflected
    /// dw element strides: k (over all groups), c (within a group), tap (t r s flattened; the
    /// layouts the engine serves keep taps packed).
    long long wk;
    long long wc;
    long long wtap;
    /// 1: the launch runs ConvWrwReduceTile (dw K C T R S, more than one tap, at most
    /// K3_REDUCE_TILE_MAX_SPLITS splits), a block a tile of tileK output channels by tileC
    /// input channels of one group (tilesK x tilesC tiles a group), written as contiguous runs;
    /// 2: ConvWrwReduceRows (K C T R S of at most K3_REDUCE_ROW_MAX_TAPS taps at one split, on
    /// gfx950); 0: ConvWrwReduce, quads of channels of one tap and output channel; 3:
    /// ConvWrwReduceDeep, the quad reduce with `quads` quads a block and 16 loads a thread in
    /// flight (many splits into few quads); 4: ConvWrwReduceFold (K3ReduceFoldArgs). All sum in
    /// the same order.
    int tile;
    int tileK;
    int tileC;
    int tilesK;
    int tilesC;
    int quads; ///< tile 3: quads a block (16, 32 or 64; 4 threads each); unused otherwise
};

/// The width fold (a width stride s dividing the filter width, groups 1). Channels-first x is
/// staged by ConvWrwFoldX as x'[n][j C + c][d][h][u] = x[n][c][d][h][u s + j - shift] (zero
/// outside the row; shift = the width padding mod s, so the folded filter has exactly S / s
/// taps); channels-last x is x'[n][d][h][u][j C + c] = x[n][d][h][u s + j][c] already. The
/// config runs the stride-1 problem over x', and ConvWrwReduceFold writes its taps back to dw:
/// folded channel j C + c, tap tau -> dw tap (tau - pre) s + j + pad (taps outside dropped).
struct K3FoldArgs
{
    const void* x;
    void* xf;
    long long rows; ///< n x c x d x h: the rows of x
    long long planes; ///< d x h
    long long width; ///< x's width L
    int c; ///< x's channels
    int s; ///< the fold (the width stride)
    int u; ///< folded width, ceil((L + shift) / s)
    int shift; ///< the padding's phase: x' position u s + j reads x position u s + j - shift
};

/// ConvWrwReduceFold: the quad reduce of the folded problem's partials (`r`, its geometry: c the
/// folded channels, taps the folded taps) in the deep reduce's order, written to the graph's dw.
struct K3ReduceFoldArgs
{
    K3ReduceArgs r;
    long long wk; ///< dw strides: output channel, input channel, tap (t r s flattened)
    long long wc;
    long long wtap;
    int c; ///< the graph's input channels (folded channel j C + c)
    int s; ///< the fold
    int fw; ///< folded filter width
    int pre; ///< folded width padding, ceil(pad / s)
    int pad; ///< the graph's width padding less the staged shift
    int filterW; ///< the graph's filter width
    int outer; ///< filter depth x height (taps outside the width)
    int flip; ///< true convolution: dw taps reflected
};

/// Three-bf16 (an fp32 graph, a config of math 1: the TF32 engine's): ConvWrwSplit writes
/// x and dy, both packed, as 3N bf16 images each, xs = [x hi | x hi | x lo] and ys = [dy hi |
/// dy lo | dy hi], so the bf16 kernel's sum over 3N images is x hi dy hi + x hi dy lo + x lo dy
/// hi (lo * lo dropped), accumulated in fp32 like any other partials.
struct K3SplitArgs
{
    const float* x;
    const float* dy;
    unsigned short* xs;
    unsigned short* ys;
    long long xElems; ///< elements of x (one third of xs)
    long long yElems; ///< elements of dy (one third of ys)
};

/// The tile reduce's LDS (tileK x tileC x taps floats at most) and its split limit (more
/// splits: the quad blocks' four split groups in four threads read faster).
#define K3_REDUCE_TILE_FLOATS 4096
#define K3_REDUCE_TILE_MAX_SPLITS 8
/// Loop 10 has its own ABI: the unchanged K3 geometry plus FP64 partial storage.
/// geometry.partial and geometry.dwOut are unused; no FP32 workspace alias is dereferenced.
struct K3AccurateArgs
{
    K3Args geometry;
    double* partial;
};

/// Fixed-order FP64 fold, rounded once to the graph's FP32 output.
struct K3AccurateReduceArgs
{
    K3ReduceArgs geometry;
    const double* partial;
};
/// The row reduce's channels per block; it takes taps * channels within 4096 floats of LDS.
#define K3_REDUCE_ROW_CHANNELS 64
#define K3_REDUCE_ROW_MAX_TAPS 64
/// The deep quad reduce: at least this many splits, the quad reduce's grid at most a
/// 1 / K3_REDUCE_DEEP_UNIT_SHARE of the dispatch units (where its few waves wait on load latency;
/// wider grids are bandwidth-bound and ran 1-3% slower deep on gfx950), and the most quads a
/// block.
#define K3_REDUCE_DEEP_MIN_SPLITS 32
#define K3_REDUCE_DEEP_UNIT_SHARE 3
#define K3_REDUCE_DEEP_MAX_QUADS 64
/// ConvWrwFoldX's tile: x elements of one row a block stages through LDS.
#define K3_FOLD_TILE 2048

// NOLINTEND(modernize-avoid-c-arrays,modernize-macro-to-enum)
