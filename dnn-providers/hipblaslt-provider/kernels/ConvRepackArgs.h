// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument blocks of the explicit-GEMM convolution repack kernels (ConvRepack.cpp). Shared
// verbatim by the host, which fills them, and the kernels, which HIPRTC compiles with this file
// as an embedded include -- so builtin types only, no standard headers.

#pragma once

#define CONV_REPACK_MAX_RANK 6
#define CONV_REPACK_SPATIAL_DIMS 3

/// dst[i] = src[i] for every multi-index i of `dims`, each side addressed through its own
/// element strides. Moves bytes, not values: one kernel per element size, none per type.
/// Covers permutes (layout transposes) and subsampling (a stride larger than packed).
struct StridedCopyArgs
{
    const void* src;
    void* dst;
    long long dims[CONV_REPACK_MAX_RANK];
    long long srcStrides[CONV_REPACK_MAX_RANK];
    long long dstStrides[CONV_REPACK_MAX_RANK];
    long long total;
    int rank;
    /// Transpose kernel only (ConvRepackLaunch.hpp): the axis unit-stride in src and the one
    /// unit-stride in dst; the rest are batch.
    int srcUnitAxis;
    int dstUnitAxis;
    /// Transpose kernel only: every offset src and dst reach fits 32 bits.
    int narrow;
};

/// dst[i] = convert(src[i]) for every multi-index i of `dims`, src fp32 and dst the entry
/// point's type: the one rounding of a sum accumulated in fp32 across several GEMMs, or of
/// `batches` fp32 partial sums added here.
struct StridedCastArgs
{
    const float* src;
    void* dst;
    long long dims[CONV_REPACK_MAX_RANK];
    long long srcStrides[CONV_REPACK_MAX_RANK];
    long long dstStrides[CONV_REPACK_MAX_RANK];
    long long total;
    /// Each element is the sum of `batches` source elements batchStride apart, in batch order
    /// (0 or 1: the element itself).
    long long batchStride;
    int rank;
    int batches;
};

/// The convolution geometry im2col and col2im share. Every problem is described as 3-D: missing
/// leading spatial axes have extent 1, pad 0, stride 1, dilation 1 and tensor stride 0.
///
/// The column matrix has one row per (image, output position) and one column per (group,
/// channel, filter element), column = g * cPerGroup * filterSize + (channelsInner ? e *
/// cPerGroup + c : c * filterSize + e) -- the order of the filter's own layout (KRSC or KCRS) so
/// the GEMM reads the filter in place. It is packed per image, positions innermost when
/// columnsFirst (element at (n * columns + column) * outputSize + o, like a channels-first
/// tensor) and columns innermost otherwise (at (n * outputSize + o) * columns + column, like a
/// channels-last one). Filter element e reads the input through tap e, or through tap
/// filter - 1 - e for true convolution.
struct ConvColumnGeometry
{
    long long n;
    long long groups;
    long long cPerGroup;
    long long input[CONV_REPACK_SPATIAL_DIMS];
    long long output[CONV_REPACK_SPATIAL_DIMS];
    long long filter[CONV_REPACK_SPATIAL_DIMS];
    long long prePadding[CONV_REPACK_SPATIAL_DIMS];
    long long stride[CONV_REPACK_SPATIAL_DIMS];
    long long dilation[CONV_REPACK_SPATIAL_DIMS];
    // Element strides of x, ordered (n, c, d, h, w).
    long long xStrides[5];
    int channelsInner;
    int flipFilter;
    int columnsFirst;
    int padding0;
};

/// How a specialized repack kernel tiles its work, chosen by the host for the problem; unused
/// fields are 0. Rows variants: `lanes` threads per column-matrix row (dx position), `rows` rows
/// per 256-thread block. Tile variant: an output tile of tileH x tileW positions of
/// channelsPerBlock channels, each channel's input halo haloH x haloW staged in LDS. Grouped
/// col2im: `rows` dx positions along width and `groupsPerBlock` groups per block, `lanes` threads per position, the column rows of a filter
/// row staged haloW output positions at a time.
struct RepackLaunchShape
{
    int lanes;
    int rows;
    int tileH;
    int tileW;
    int haloH;
    int haloW;
    int groupsPerBlock;
    int channelsPerBlock;
};

/// col[r, column] = the input element row r's output position reads through the column's filter
/// element, or 0 in the padding. One thread per column-matrix element; moves bytes.
struct Im2ColArgs
{
    const void* x;
    void* col;
    ConvColumnGeometry geometry;
    long long total;
    RepackLaunchShape launch;
};

/// dx[n, c, i] = the sum, over every (output position, filter element) that reads input position
/// i, of the fp32 column matrix entry for them, converted once to dx's type; 0 when none does.
/// One thread per dx element, so no atomics and a deterministic sum. A filter whose taps never
/// overlap (stride at least the dilated extent on every axis, a single tap included) gives
/// each element at most one term: its column matrix is in dx's type (`col` then points at
/// that). convColumnsDisjoint decides which.
struct Col2ImArgs
{
    const float* col;
    void* x;
    ConvColumnGeometry geometry;
    long long total;
    RepackLaunchShape launch;
};

/// True when no two taps of @p g's filter reach the same input element: on every axis the
/// stride is at least the dilated filter extent. Then backward data's columns hold dx's type.
inline bool convColumnsDisjoint(const ConvColumnGeometry& g)
{
    for(int axis = 0; axis < CONV_REPACK_SPATIAL_DIMS; ++axis)
    {
        if(g.stride[axis] < (g.filter[axis] - 1) * g.dilation[axis] + 1)
        {
            return false;
        }
    }
    return true;
}
