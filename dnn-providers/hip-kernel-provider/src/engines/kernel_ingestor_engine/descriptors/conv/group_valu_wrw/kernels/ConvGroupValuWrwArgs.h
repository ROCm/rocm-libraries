// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The argument block of the VALU grouped backward-weights kernels (ConvGroupValuWrw.cpp).
// Shared verbatim by the host, which fills it, and the kernels, which HIPRTC compiles with this
// file as an embedded include -- so builtin types only, no standard headers.

#pragma once

// NOLINTBEGIN(modernize-avoid-c-arrays)
// HIPRTC compiles this file as an embedded include, with no standard library: C arrays,
// macros and C-style code are deliberate, and its arithmetic types are part of the kernels.

/// One backward weights dw (k, c / groups, t, r, s) from x (n, c, di, hi, wi) and dy (n, k, dOut,
/// ho, wo) (2-D problems run with di = dOut = fd = 1, 1-D also with hi = ho = 1). The reduction
/// runs over the rows n * dOut * ho, cut into `splits` runs of `rowsPerSplit` rows: with one
/// split the main kernel writes dw, otherwise it writes one fp32 sum per (split, tile, filter
/// element) to `partial` and the fold kernel adds the splits in order. A tile is one block of CB
/// input and KB output channels of one group at one depth tap (tiles = fd * tilesPlane).
struct ConvGroupValuWrwArgs
{
    const void* x;
    const void* dy;
    float* partial;
    void* dw;
    long long xs[4]; ///< element strides of x: n, c, h, w
    long long ys[4]; ///< element strides of dy: n, k, h, w
    long long ws[4]; ///< element strides of dw: k, c, r, s
    long long xBytes; ///< bytes from x to one past its last element
    long long yBytes; ///< the same for dy
    int n;
    int groups;
    int cpg; ///< input channels a group
    int kpg; ///< output channels a group
    int hi;
    int wi;
    int ho;
    int wo;
    int ph; ///< pre-padding along the height (the width's is the config's)
    int sh; ///< stride along the height (the width's is the config's)
    int dh; ///< dilation along the height (the width's is the config's)
    int lanes; ///< lanes an output row: a power of two, at most the wave
    int rows; ///< n * ho
    int rowsPerSplit;
    int splits;
    int cBlocks; ///< tiles along the input channels of a group: cpg / CB
    int kBlocks; ///< tiles along the output channels of a group: kpg / KB
    int tiles; ///< fd * groups * cBlocks * kBlocks
    int flip; ///< true convolution: the filter is reflected
    // Row-ring configs (GVW_RING): a lane group walks `chunk` consecutive output rows of one
    // image (an item); items are image-major, `itemsPerImage` an image, `itemsPerSplit` a split.
    int chunk;
    int itemsPerImage;
    int items;
    int itemsPerSplit;
    // Depth (3-D; 1 / 0 below): x depth, dy depth, depth padding, stride, dilation, filter taps.
    int di;
    int dOut;
    int pd;
    int sd;
    int dd;
    int fd;
    int rowsAnImage; ///< dOut * ho
    int tilesPlane; ///< groups * cBlocks * kBlocks (tiles of one depth tap)
    int padding0;
    long long xsd; ///< element stride of x along the depth
    long long ysd; ///< the same for dy
    long long wsd; ///< the same for dw (its t axis)
};

// NOLINTEND(modernize-avoid-c-arrays)
