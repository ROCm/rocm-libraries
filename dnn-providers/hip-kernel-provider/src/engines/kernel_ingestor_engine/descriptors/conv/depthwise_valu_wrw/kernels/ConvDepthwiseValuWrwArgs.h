// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The argument block of the VALU depthwise backward-weights kernels (ConvDepthwiseValuWrw.cpp).
// Shared verbatim by the host, which fills it, and the kernels, which HIPRTC compiles with this
// file as an embedded include -- so builtin types only, no standard headers.

#pragma once

/// One depthwise backward weights: dw (c, 1, kd, kh, kw) from x (n, c, di, hi, wi) and dy
/// (n, c, do, ho, wo), both packed in one layout. 1-D and 2-D problems lead with extent-1 axes
/// (pad 0, stride 1, dilation 1). The reduction runs over rows = n * do * ho, in `partitions`
/// partitions of `rowsPerPartition` rows: the partial kernel writes one fp32 sum per
/// (partition, channel, tap) to `partial`, the fold kernel sums them in partition order.
struct ConvDepthwiseValuWrwArgs
{
    const void* x;
    const void* dy;
    float* partial;
    void* dw;
    int n;
    int c;
    int di;
    int hi;
    int wi;
    int dOut;
    int ho;
    int wo;
    int kd;
    int kh;
    int kw;
    int pd;
    int ph;
    int pw;
    int sd;
    int sh;
    int sw;
    int dd;
    int dh;
    int dw_;
    int rows;
    int partitions;
    int rowsPerPartition;
    int taps;
};
