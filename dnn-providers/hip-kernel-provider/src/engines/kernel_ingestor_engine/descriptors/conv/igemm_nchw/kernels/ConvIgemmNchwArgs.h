// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument block of the channels-first backward-data implicit-GEMM kernels (K7). Shared verbatim
// by the host pack, which fills it, and by the kernels, which HIPRTC compiles with this file as
// an embedded include: builtin types only, no standard headers.
#pragma once

/// Backward data of a convolution along the width, read and written in place: dy packed
/// N K (rows) Wy, the filter packed K C T R S, dx packed N C (rows) Wx, where (rows) are the depth
/// and height positions. Along depth and height each filter extent is at most its stride
/// (T <= sd, R <= sh), so a dx row (d, h) takes at most one dy row and one filter row:
/// d + pd = sd * qd + td and h + ph = sh * qh + th, live when td < T, th < R and (qd, qh) lies in
/// dy; a dx row with none is zero. The identity (extent 1, stride 1, no padding) is the 1-D case.
/// Shapes are runtime; only the tile, the stride and the taps per phase are compile-time.
///
/// With x + pad = stride * q + phase, dx[c][x] = sum over k and j of
/// w[k][c][td][th][phase + stride * j] * dy[k][qd][qh][q - j]. Grid: (q tiles, channel tiles,
/// images * dx rows).
struct K7Args
{
    const void* dy;
    const void* w;
    void* dx;
    int n; ///< images
    int c; ///< dx channels
    int k; ///< dy channels (the reduction)
    int rows; ///< dx rows per image: depth times height
    int wx; ///< dx width
    int wy; ///< dy width
    int fs; ///< filter taps along the width (S)
    int pad; ///< pre-padding along the width
    int qBase; ///< the first phase index a tile starts from: pad / stride
    int rowsY; ///< dy rows per image: dy depth times dy height
    int hx; ///< dx height
    int hy; ///< dy height
    int dyDepth; ///< dy depth
    int sd, pd, fd; ///< depth: stride, pre-padding, filter extent (T)
    int sh, ph, fh; ///< height: stride, pre-padding, filter extent (R)
    int fStride; ///< filter elements per (k, c): T * R * S
    long long dyImage; ///< dy elements per image
    long long dxImage; ///< dx elements per image
};
