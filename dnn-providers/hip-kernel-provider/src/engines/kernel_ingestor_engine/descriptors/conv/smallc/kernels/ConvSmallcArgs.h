// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument blocks of the small-input-channel forward convolution kernels (CONV_SMALLC_FWD).
// Shared verbatim by the host pack, which fills them, and by the kernels, which HIPRTC compiles
// with this file as an embedded include: builtin types only, no standard headers.
#pragma once

/// The convolution: x (n, c, d, h, w) and y (n, k, d, h, w) through their element strides, the
/// weights pre-formatted by ConvSmallcFormat. Extents and padding are runtime; the filter,
/// stride, dilation and tile are compile-time.
struct SmallcArgs
{
    const void* x;
    const void* w; ///< formatted weights wf[n][kp], kp = rho * sp + s (zeros past the filter)
    void* y;
    int n, c, k;
    int di, hi, wi;
    int dout, ho, wo;
    int pd, ph, pw;
    int kpad; ///< formatted weight rows: k rounded up to the block's output channels
    int tilesW, tilesH;
    int padding0;
    long long xs[5]; ///< x element strides: n, c, d, h, w
    long long ys[5]; ///< y element strides: n, k, d, h, w
};

/// The weight pre-pass: wf[n][kp] (or by k within blocks of bn columns, see bn) for kp < kpl (filter rows rho = (c, t, r) of sp taps each,
/// rows padded to whole MMA steps), read from the graph's filter through its element strides
/// (k, c, t, r, s); zero past C, the filter or K. flip reflects the taps (true convolution).
struct SmallcFormatArgs
{
    const void* w;
    void* wf;
    long long wStrides[5];
    int k, c, kd, r, s;
    int sp, kpl, kpad;
    int flip;
    int bn; ///< 0: wf[n][kp] (MMA); else wf[n / bn][kp][n % bn] (FMA: a block's rows by k)
};
