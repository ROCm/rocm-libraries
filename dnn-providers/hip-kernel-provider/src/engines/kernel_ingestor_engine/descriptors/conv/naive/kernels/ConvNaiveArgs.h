// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The by-value argument block every naive convolution kernel takes. Shared verbatim by the
// host pack, which fills it, and by the kernels, which HIPRTC compiles with this file as an
// embedded include -- so it uses only builtin types and no standard headers.
//
// Every convolution is described as 3-D. A 1-D or 2-D problem pads the leading spatial axes
// with extent 1, pad 0, stride 1, dilation 1 and tensor stride 0, which leaves every index
// computation unchanged.
//
// Tensors are addressed purely through their strides, so any layout works: NCHW, NHWC,
// their 1-D and 3-D forms, and non-packed views. The x/w/y names follow the forward
// convolution for every direction:
//   forward          reads x and w, writes y
//   backward data    reads y (dy) and w, writes x (dx)
//   backward weights reads x and y (dy), writes w (dw)

#pragma once

#define CONV_NAIVE_SPATIAL_DIMS 3
#define CONV_NAIVE_TENSOR_DIMS 5

struct ConvNaiveArgs
{
    // Not const: which of the three a kernel writes depends on its direction.
    void* x; // x (forward, backward weights) or dx (backward data)
    void* w; // w (forward, backward data) or dw (backward weights)
    void* y; // y (forward) or dy (backward data, backward weights)

    long long n; // batch
    long long c; // input channels, all groups
    long long k; // output channels, all groups
    long long groups;

    long long inSpatial[CONV_NAIVE_SPATIAL_DIMS]; // x spatial extents, (d, h, w)
    long long outSpatial[CONV_NAIVE_SPATIAL_DIMS]; // y spatial extents
    long long filter[CONV_NAIVE_SPATIAL_DIMS]; // w spatial extents
    long long prePadding[CONV_NAIVE_SPATIAL_DIMS];
    long long stride[CONV_NAIVE_SPATIAL_DIMS];
    long long dilation[CONV_NAIVE_SPATIAL_DIMS];

    // Element strides, ordered (n, c, d, h, w) for x and y and (k, c, d, h, w) for w.
    long long xStrides[CONV_NAIVE_TENSOR_DIMS];
    long long wStrides[CONV_NAIVE_TENSOR_DIMS];
    long long yStrides[CONV_NAIVE_TENSOR_DIMS];

    // Total output elements of the launch: y for forward, x for backward data, w for backward
    // weights. Each thread computes one and strides over the rest.
    long long total;

    // Nonzero for true convolution, which reflects the filter; zero for cross-correlation.
    int flipFilter;
    int padding0; // keeps the struct size a multiple of 8 on host and device alike
};
