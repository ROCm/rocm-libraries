// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The argument block of the VALU grouped forward / backward-data kernels (ConvGroupValu.cpp).
// Shared verbatim by the host, which fills it, and the kernels, which HIPRTC compiles with this
// file as an embedded include -- so builtin types only, no standard headers.

#pragma once

// NOLINTBEGIN(modernize-avoid-c-arrays)
// HIPRTC compiles this file as an embedded include, with no standard library: C arrays,
// macros and C-style code are deliberate, and its arithmetic types are part of the kernels.

/// One grouped convolution in the kernel's roles: out (n, groups x ko, dOut, ho, wo) =
/// in (n, groups x ci, di, hi, wi) correlated with the prepared filter. Forward: in = x, out = y,
/// ci / ko the graph's channels a group. Backward data: in = dy, out = dx, ci = the graph's output
/// channels a group, ko its input channels, the filter reflected and its roles swapped by the
/// prepare kernel, the padding (span - 1 - pad), at unit stride over dy upsampled by the graph's
/// stride (uh, ud; the width's is the config's): dx = (dy with stride - 1 zeros between its
/// elements) correlated with the reflected filter. 1-D and 2-D problems lead with extent-1 axes.
/// The output rows n x dOut x ho are cut into blocks of `rowsPerBlock`.
struct ConvGroupValuArgs
{
    const void* in;
    void* out;
    const void* w; ///< the graph's filter (k, c / groups, t, r, s), read by the prepare kernel
    float* wt; ///< prepared filter: [group][ko block][ci][t][r][s][KB] fp32
    long long is[5]; ///< element strides of in: n, c, d, h, w
    long long os[5]; ///< element strides of out: n, k, d, h, w
    long long ws[5]; ///< element strides of w: k, c, t, r, s
    long long inBytes; ///< bytes from in to one past its last element
    long long outBytes;
    int n;
    int groups;
    int ci; ///< input channels a group (kernel roles)
    int ko; ///< output channels a group (kernel roles)
    int di;
    int hi;
    int wi;
    int dOut;
    int ho;
    int wo;
    int pd; ///< padding along depth and height (the width's is the config's)
    int ph;
    int sd; ///< stride along depth and height (the width's is the config's)
    int sh;
    int dd; ///< dilation along depth and height (the width's is the config's)
    int dh;
    int lanes; ///< lanes an output row: a power of two, at most the wave
    int rows; ///< n x dOut x ho
    int rowsPerBlock;
    int rowBlocks;
    int kBlocks; ///< ko / KB
    int wGraphK; ///< the graph filter's output channels a group (for the prepare kernel)
    int wGraphC; ///< its input channels a group
    int backward; ///< the prepare kernel swaps roles and reflects (backward data)
    int flip; ///< convolution mode: reflect the filter
    int prepared; ///< elements of wt
    int ud; ///< backward data: the graph's depth stride (in upsampled), else 1
    int uh; ///< the same along the height
};

// NOLINTEND(modernize-avoid-c-arrays)
