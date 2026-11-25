// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The argument block of the VALU depthwise kernel (ConvDepthwiseValu.cpp). Shared verbatim by
// the host, which fills it, and the kernel, which HIPRTC compiles with this file as an embedded
// include -- so builtin types only, no standard headers.

#pragma once

/// One depthwise convolution in forward terms: `in` (n, c, (di,) hi, wi) convolved with `filter`
/// (c, (kd,) kh, kw) into `out` (n, c, (dOut,) ho, wo). Backward data is passed as the forward
/// convolution of dy into dx: in = dy, out = dx, mirrored padding, reflected filter. 2-D
/// problems leave the depth fields at extent 1, pad 0, stride 1, dilation 1.
struct ConvDepthwiseValuArgs
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
    int kh;
    int kw;
    int ph;
    int pw;
    int sh;
    int sw;
    int dh;
    int dw;
    int di;
    int dOut;
    int kd;
    int pd;
    int sd;
    int dd;
};
