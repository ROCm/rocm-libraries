// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument blocks and LDS arithmetic of the stem backward-data kernels (CONV_STEM_BWD). Shared
// verbatim by the host pack, which fills the blocks and gates a config's LDS, and by the
// kernels, which HIPRTC compiles with this file as an embedded include: builtin types and
// constexpr functions only, no standard headers.
#pragma once

/// dx[n][c][y][x] for few input channels: super-pixel (i, j) stands for the sh x sw dx pixels
/// (sh i + p, sw j + q) and reads dy rows i + oh - u, columns j + ow - v (u < U, v < V) against
/// the transformed filter b. Element strides, ordered (n, k, h, w) for dy and (n, c, h, w) for
/// dx.
struct ConvStemBwdArgs
{
    const void* dy;
    const void* b; ///< [NB][U V][kp], ConvStemBwdTransform's
    void* dx;
    int n, c, k, kp; ///< kp: k padded to whole chunks
    int hi, wi, ho, wo;
    int oh, ow; ///< the window's offsets
    long long dys[4];
    long long dxs[4];
    long long dyImage; ///< elements of one image of dy (its buffer's extent)
    // 3-D (CONV_SB_FT depth taps, depth stride 1): dx planes d of dy's dout, depth padding pd;
    // the reduction runs over (depth tap, k), kpr the per-tap k padding (kp = FT kpr); dyd and
    // dxd the planes' element strides. The grid's z is n d.
    int d, dout, pd, kpr;
    long long dyd, dxd;
    // Space-to-depth (CONV_SB_SD, stride 1): super-pixels of SD x SD dx pixels on a virtual
    // stride SD; super-row y of dy holds rows SD y + sdh + e (e < SD; columns alike), the
    // reduction runs over (channel chunk, (eh, ew), k in the chunk) with kp = SD SD kpr. ho and wo
    // stay dy's own extents.
    int sdh, sdw;
};

/// b[nb = (c, p, q)][u][v][k] = w[k][c][r][s] with r = sh (u - oh) + p + ph and s = sw (v - ow)
/// + q + pw (reflected for true convolution), zero where no filter tap pairs the phase with the
/// offset and past c and k. Filter element strides ordered (k, c, r, s).
struct ConvStemBwdTransformArgs
{
    const void* w;
    void* b;
    int nb, u, v, kp;
    int c, k, r, s, sh, sw, ph, pw, oh, ow, flip;
    int padding0; ///< keeps the struct size a multiple of 8 on host and device alike
    long long ws[4];
    // 3-D: b's k axis is (depth tap, k), kpr per tap; ft depth taps (0: 2-D), wsd the filter's
    // depth stride.
    int ft, kpr;
    long long wsd;
    // Space-to-depth: b's k axis is (channel chunk, (eh, ew), k in the chunk), kpr padded channels
    // (kp = SD SD kpr); tap r = sh (u - oh) + p + ph - dh - eh (sh the virtual stride; columns
    // alike).
    int dh, dw;
};

/// Elements a staged k run is padded by: 16 bytes, which keeps every run's 16-byte stores
/// aligned and spreads the rows over the LDS banks.
constexpr int sbPad(int elementBytes)
{
    return 16 / elementBytes;
}

/// LDS bytes of a block: the dy window, (th + u - 1) rows by (tw + v - 1) columns, and b's chunk,
/// nb by u v, each a run of kc output channels plus the pad.
constexpr long long sbLdsBytes(int elementBytes, int u, int v, int nb, int kc, int th, int tw)
{
    return (static_cast<long long>(th + u - 1) * (tw + v - 1) + static_cast<long long>(nb) * u * v)
           * (kc + sbPad(elementBytes)) * elementBytes;
}
