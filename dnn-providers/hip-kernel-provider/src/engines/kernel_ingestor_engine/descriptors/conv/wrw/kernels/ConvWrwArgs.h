// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument blocks of the matrix-core backward-weights kernels (K3). Shared verbatim by the host
// pack, which fills them, and by the kernels, which HIPRTC compiles with this file as an
// embedded include: builtin types only, no standard headers.
#pragma once

/// Backward weights: dw[g][k][c][t][r][s] = sum over images and output positions of
/// dy[n][g k][od][oh][ow] * x[n][g c][od sd + t dd - pd][oh sh + r dh - ph][ow sw + s dw - pw].
/// x and dy are read in place through their element strides (one of the two packed layouts
/// the kernel is compiled for); the kernel writes fp32 partials, [split][g][tap][k][c], which
/// ConvWrwReduce adds in split order into dw. Shapes, padding and the depth/height filter,
/// stride and dilation are runtime; the filter width, its stride and dilation, and the tile are
/// compile-time.
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
    long long steps; ///< rows * chunks
    long long stepsPerSplit;
    /// Row-ring kernel: steps per track (ceil(ho / dh) + fr - 1).
    int trackLen;
    /// Direct loop: row segments per image and rows per segment. Packed rows (loop 7): the
    /// virtual row width (columns a step row takes) and the output rows a step packs.
    int segs;
    int segRows;
    int walk; ///< 0 cells (tile fastest), 1 stream (tile-major runs)
    /// Element strides: x (n, c, d, h, w), dy (n, k, d, h, w).
    long long xs[5];
    long long ys[5];
    /// Elements of one image of x and of dy (each image is one 32-bit-addressed buffer).
    long long xImage;
    long long yImage;
    /// dw elements per split of the partials: groups * taps * (k / groups) * (c / groups).
    long long partialElems;
    /// Nonzero (one split, dw K T R S C or of one tap): the kernel writes dw itself in its
    /// element type, channels contiguous, and no reduce runs; `partial` is unused.
    int dwStore;
    int flip; ///< nonzero: true convolution, dw written with its taps reflected
    void* dwOut; ///< dw (`dw` is the width dilation)
    /// dw element strides: k (over all groups) and tap (t r s flattened); c is 1.
    long long dwk;
    long long dwtap;
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
    /// Nonzero: the launch runs ConvWrwReduceRows (dw K C T R S, at most 2 splits), whose block
    /// writes the contiguous run of K3_REDUCE_ROW_CHANNELS channels' taps of one output channel;
    /// zero: ConvWrwReduce, quads of channels of one tap and output channel. A record of the
    /// host's choice; the kernels do not read it.
    int rows;
    int padding0; ///< keeps the struct size a multiple of 8 on host and device alike
};

/// Three-bf16 (an fp32 graph under TF32 permission, a config of math 1): ConvWrwSplit writes
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

/// The row reduce's channels per block; it takes taps * channels within 4096 floats of LDS.
#define K3_REDUCE_ROW_CHANNELS 64
#define K3_REDUCE_ROW_MAX_TAPS 64
