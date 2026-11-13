// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument block of the channels-first backward-data implicit-GEMM kernels (K7). Shared verbatim
// by the host pack, which fills it, and by the kernels, which HIPRTC compiles with this file as
// an embedded include: builtin types only, no standard headers.
#pragma once

/// Backward data of a convolution along the width, read and written in place: dy packed
/// N K (rows) Wy, the filter packed K C T R S, dx packed N C (rows) Wx, where (rows) are the depth
/// and height positions. A dx row (d, h) takes dy row (qd, qh) through filter row (td, th) for
/// every d + pd = sd * qd + dd * td and h + ph = sh * qh + dh * th (a dilation only at stride 1)
/// with the filter row inside the filter and the dy row inside dy; a dx row with none is zero. The identity (extent 1, stride 1, no padding)
/// is the 1-D case.
/// Shapes are runtime; only the tile, the stride and the taps per phase are compile-time.
///
/// With x + pad = stride * q + phase, dx[c][x] = sum over the row pairs, k and j of
/// w[k][c][td][th][phase + stride * j] * dy[k][qd][qh][q - j]; at stride 1 with dilation dil,
/// dx[c][x] = sum of w[k][c][td][th][j] * dy[k][qd][qh][x + pad - dil j]. Grid: (q tiles, channel tiles,
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
    int dil; ///< dilation along the width (stride 1 only; else 1)
    int dd, dh; ///< dilation along depth and height (stride 1 on that axis only; else 1)
    int padding0, padding1;
    long long dyImage; ///< dy elements per image
    long long dxImage; ///< dx elements per image
};
