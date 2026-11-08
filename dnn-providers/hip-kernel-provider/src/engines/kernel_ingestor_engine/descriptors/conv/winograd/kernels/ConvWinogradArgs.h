// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The by-value argument blocks of the Winograd convolution kernels (CONV_WINOGRAD_FWD,
// CONV_WINOGRAD_BWD). Shared verbatim by the host pack, which fills them, and by the kernels, which
// HIPRTC compiles with this file as an embedded include -- so it uses only builtin types and no
// standard headers.
//
// Every launch is written for a "forward-form" problem: in (N, CI, HI, WI) -> out (N, CO, HO, WO),
// a 3x3 correlation with top/left padding (ph, pw) and explicit output extents, groups.
//   forward:       in = x,  out = y,  CI = C, CO = K, pad = (pre_h, pre_w).
//   backward data: in = dy, out = dx, CI = K, CO = C, pad = (2 - pre_h, 2 - pre_w); the filter
//                  pre-pass rotates the filter by 180 degrees and swaps its K/C roles.
// `flip` (true convolution) reflects the filter in the same pre-pass.

#pragma once

#define K5_NCHW 0
#define K5_NHWC 1
#define K5_FORWARD 0
#define K5_BACKWARD_DATA 1
#define K5_COPY_RANK 4

struct K5Args
{
    const void* in; // x (forward) or dy (backward data)
    const void* w; // filter, the graph's K x C/G x 3 x 3 (KCRS for NCHW, KRSC for NHWC)
    void* out; // y (forward) or dx (backward data)
    void* u; // workspace: transformed filter
    void* v; // workspace: transformed input (multipass)
    float* mw; // workspace: Winograd-domain products (multipass)
    int n, groups;
    int ci, co; // forward-form channels, all groups
    int hi, wi, ho, wo;
    int ph, pw; // forward-form top/left padding (may be negative)
    int th, tw; // tiles per image along H and W
    int tiles; // n * th * tw
    int flip; // nonzero: true convolution, the filter is reflected
};

/// Pass 3 of the multipass engine: C[b] (m x n, fp32) = A[b] (m x k) * B[b] (k x n), row-major.
struct K5GemmArgs
{
    const void* a;
    const void* b;
    float* c;
    int m, n, k;
    int lda, ldb, ldc;
    long long sa, sb, sc;
};

/// A strided copy of up to K5_COPY_RANK axes, for staging an operand in or out of the workspace.
struct K5CopyArgs
{
    const void* src;
    void* dst;
    long long dims[K5_COPY_RANK];
    long long srcStrides[K5_COPY_RANK];
    long long dstStrides[K5_COPY_RANK];
    long long total;
    int rank;
    int elementBytes;
};
