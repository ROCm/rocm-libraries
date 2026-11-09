// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument block of the channels-first backward-data implicit-GEMM kernels (K7). Shared verbatim
// by the host pack, which fills it, and by the kernels, which HIPRTC compiles with this file as
// an embedded include: builtin types only, no standard headers.
#pragma once

/// Backward data of a convolution along the width, read and written in place: dy packed
/// N K (rows) Wy, the filter packed K C S, dx packed N C (rows) Wx, where (rows) are the depth
/// and height positions, which the convolution maps one to one (filter extent 1, stride 1, no
/// padding). Shapes are runtime; only the tile, the stride and the taps per phase are
/// compile-time.
///
/// With x + pad = stride * q + phase, dx[c][x] = sum over k and j of
/// w[k][c][phase + stride * j] * dy[k][q - j]. Grid: (q tiles, channel tiles, images * rows).
struct K7Args
{
    const void* dy;
    const void* w;
    void* dx;
    int n; ///< images
    int c; ///< dx channels
    int k; ///< dy channels (the reduction)
    int rows; ///< positions per image along the identity axes
    int wx; ///< dx width
    int wy; ///< dy width
    int fs; ///< filter taps along the width
    int pad; ///< pre-padding along the width
    int qBase; ///< the first phase index a tile starts from: pad / stride
    int padding0;
    long long dyImage; ///< dy elements per image
    long long dxImage; ///< dx elements per image
};
