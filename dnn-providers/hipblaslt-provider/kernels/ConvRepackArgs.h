// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument blocks of the explicit-GEMM convolution repack kernels (ConvRepack.cpp). Shared
// verbatim by the host, which fills them, and the kernels, which HIPRTC compiles with this file
// as an embedded include -- so builtin types only, no standard headers.

#pragma once

// NOLINTBEGIN(modernize-avoid-c-arrays,modernize-macro-to-enum)
// HIPRTC compiles this file as an embedded include, with no standard library: C arrays,
// macros and C-style code are deliberate, and its arithmetic types are part of the kernels.

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

#define CONV_REPACK_EPI_OPERANDS 4
#define CONV_REPACK_EPI_SCALARS 8
#define CONV_REPACK_EPI_OPS 8

// The pointwise steps a fused convolution epilogue runs, in the order of
// hipdnn_plugin_sdk::conv_epilogue::OpCode.
#define CONV_REPACK_EPI_ADD_OPERAND 0
#define CONV_REPACK_EPI_ADD_SCALAR 1
#define CONV_REPACK_EPI_MUL_OPERAND 2
#define CONV_REPACK_EPI_MUL_SCALAR 3
#define CONV_REPACK_EPI_RELU 4
#define CONV_REPACK_EPI_GELU 5
#define CONV_REPACK_EPI_GELU_TANH 6
#define CONV_REPACK_EPI_SWISH 7
#define CONV_REPACK_EPI_SIGMOID 8
#define CONV_REPACK_EPI_TANH 9
#define CONV_REPACK_EPI_ELU 10
#define CONV_REPACK_EPI_ABS 11
#define CONV_REPACK_EPI_SOFTPLUS 12

/// A fused convolution epilogue, selected at run time step by step: the pointwise chain after
/// the convolution, applied in fp32 to each element before its one rounding. op[i] is a
/// CONV_REPACK_EPI_* code; slot[i] its operand (ADD_OPERAND, MUL_OPERAND) or its first value.
/// Operands are read at the element's channel (per channel) or at 0; operandType 0 fp32,
/// 1 fp16, 2 bf16. The channel is the coordinate of the cast's axis channelAxis, so no element
/// pays a division for it; or, with channelAxis -1, the cast is one flat walk over a dense
/// tensor and the channel of element o is (o / flatChannelStride) % flatChannels, carried along
/// the walk by additions.
struct ConvRepackEpilogue
{
    const void* operand[CONV_REPACK_EPI_OPERANDS];
    int operandType[CONV_REPACK_EPI_OPERANDS];
    int operandPerChannel[CONV_REPACK_EPI_OPERANDS];
    float scalar[CONV_REPACK_EPI_SCALARS];
    int op[CONV_REPACK_EPI_OPS];
    int slot[CONV_REPACK_EPI_OPS];
    long long flatChannelStride;
    long long flatChannels;
    int ops;
    int channelAxis;
};

/// A strided cast (StridedCastArgs) whose elements pass through `epilogue` before their
/// rounding: the last pass of a fused explicit-GEMM convolution.
struct StridedCastEpilogueArgs
{
    StridedCastArgs cast;
    ConvRepackEpilogue epilogue;
};

/// data[i] = lower where data[i] <= lower, upper where data[i] >= upper, else unchanged (NaN
/// included), for the `total` elements of a dense tensor of the entry point's type, in place
/// (`src` is `dst`). The relu / clamp tail of a fused epilogue whose GEMM already wrote the
/// rounded result: with both bounds exact in that type, clamping the rounded value gives the
/// rounding of the clamped one, so the result still has one rounding. Elements left unchanged
/// are not written back.
struct ClampArgs
{
    const void* src;
    void* dst;
    long long total;
    float lower;
    float upper;
};

/// dst[c * rows + r] = src[c * srcStride] for every r < rows and c < channels: a column-major
/// rows x channels matrix whose every row is the per-channel vector src, in the entry point's
/// element size. The broadcast bias a fused epilogue's GEMM adds as its C.
struct BroadcastArgs
{
    const void* src;
    void* dst;
    long long rows;
    long long channels;
    long long srcStride;
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

// NOLINTEND(modernize-avoid-c-arrays,modernize-macro-to-enum)
