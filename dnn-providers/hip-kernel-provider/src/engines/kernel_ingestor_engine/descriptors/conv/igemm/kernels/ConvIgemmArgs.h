// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Argument blocks of the implicit-GEMM convolution kernels (K1). Shared verbatim by the host
// pack, which fills them, and by the kernels, which HIPRTC compiles with this file as an
// embedded include: builtin types only, no standard headers.
#pragma once

/// The convolution: channels-last x and y (N D H W C), filter K T R S C/g (forward) or, for
/// backward data, transposed to [g][T R S][C/g][K/g]. Backward data reads y (dy) and writes x
/// (dx). Shapes are runtime; only the tile config is compile-time.
struct K1Args
{
    void* x;
    const void* w;
    void* y;
    int n, c, k, groups; // c and k over all groups
    int di, hi, wi; // input
    int dout, ho, wo; // output
    int ft, fr, fs; // filter
    int sd, sh, sw;
    int pd, ph, pw;
    int dd, dh, dw;
    long long m; // n * dout * ho * wo
    /// Split-K: splits > 1 divides the reduction; each split writes fp32 partials (partial +
    /// split * outElems, indexed like the output) for ConvIgemmReduce to add in order.
    int splits;
    int padding0;
    float* partial;
    long long outElems;
    /// Backward data: element strides of the transposed filter per input channel and per tap
    /// (0: [g][tap][c][k], i.e. k and c k). The pack picks whichever transposed layout the
    /// graph's filter reaches by one tiled transpose.
    long long wRowStride;
    long long wTapStride;
};

/// Adds the splits' fp32 partials in split order into the output's element type.
struct K1ReduceArgs
{
    const float* partial;
    void* out;
    long long elems;
    int splits;
    int padding0;
};

#define CONV_IGEMM_COPY_RANK 6

/// One strided copy of `total` elements of `elementBytes` each: element i of the destination
/// order (dims, outermost first) is read at its coordinates through srcStrides and written
/// through dstStrides. Stages operands to and from the layouts the convolution reads.
struct ConvIgemmCopyArgs
{
    const void* src;
    void* dst;
    long long dims[CONV_IGEMM_COPY_RANK];
    long long srcStrides[CONV_IGEMM_COPY_RANK];
    long long dstStrides[CONV_IGEMM_COPY_RANK];
    long long total;
    int rank;
    int elementBytes;
};

/// Group fusion: the filter of f consecutive groups expanded into one block-diagonal group of
/// f * cg input channels (zeros where an output channel's own group does not reach), written
/// packed K T R S (f cg). Source element (k, c, tap) at k * srcK + c * srcC + tap * srcTap.
struct ConvIgemmExpandArgs
{
    const void* src;
    void* dst;
    long long k;
    long long taps;
    long long cg;
    long long kg;
    long long f;
    long long srcK;
    long long srcC;
    long long srcTap;
    int elementBytes;
    int padding0;
};

/// The transpose's square tile: one 256-thread block per tile of each batch.
#define CONV_IGEMM_TRANSPOSE_TILE 64

/// A batched 2-D transpose of `elementBytes`-byte elements: dst[b][j][i] = src[b][i][j], the
/// source `rows` x `cols` row-major. Stages packed channels-first activations to channels-last
/// (rows = C, cols = D H W) and back (rows = D H W, cols = C).
struct ConvIgemmTransposeArgs
{
    const void* src;
    void* dst;
    long long batches;
    long long rows;
    long long cols;
    int elementBytes;
    int padding0;
};

/// Three-bf16 operand split along a channel axis (TF32 pre-pass of the engines that run an fp32
/// graph as a bf16 problem with a tripled reduction): a packed fp32 tensor viewed as
/// [outer][groups][cg][inner] becomes bf16 [outer][groups][3 cg][inner], channel block `part`
/// (0, 1, 2) of each group holding by 2 bits of `pattern` the high part (0), the low part (1) or
/// the high part with infinities zeroed (2: the block that meets the other operand's low part).
struct K1Split3Tensor
{
    const float* src;
    unsigned short* dst;
    long long outer;
    long long inner;
    int groups;
    int cg;
    int pattern;
    int padding0;
};

/// The two operands one launch splits (an absent one: src null).
struct K1Split3Args
{
    K1Split3Tensor t[2];
};
