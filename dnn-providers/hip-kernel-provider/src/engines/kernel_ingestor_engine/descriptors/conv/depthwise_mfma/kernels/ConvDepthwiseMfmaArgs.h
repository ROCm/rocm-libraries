// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The argument block of the matrix-core depthwise kernel (ConvDepthwiseMfma.cpp). Shared verbatim
// by the host, which fills it, and the kernel, which HIPRTC compiles with this file as an embedded
// include -- so builtin types only, no standard headers.

#pragma once

/// One unit-stride 2-D depthwise convolution in forward terms, channels-last: `in` (n, hi, wi, c)
/// convolved with `filter` (c, kh, kw) into `out` (n, ho, wo, c), padding ph / pw before. Backward
/// data is passed as the forward convolution of dy into dx: in = dy, out = dx, mirrored padding,
/// the filter read reflected. A block streams `rows` output rows; `chunks` = ceil(ho / rows).
struct ConvDepthwiseMfmaArgs
{
    const void* in;
    const void* filter;
    void* out;
    int n;
    int c;
    int hi;
    int wi;
    int ho;
    int wo;
    int ph;
    int pw;
    int rows;
    int chunks;
};
