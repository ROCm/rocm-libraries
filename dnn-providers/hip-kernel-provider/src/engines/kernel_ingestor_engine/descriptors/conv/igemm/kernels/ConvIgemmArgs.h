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
