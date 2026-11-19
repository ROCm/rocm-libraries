// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The argument block of the narrow-output convolution kernels (ConvNarrow.cpp). Shared verbatim
// by the host, which fills it, and the kernels, which HIPRTC compiles with this file as an
// embedded include -- so builtin types only, no standard headers.

#pragma once

// NOLINTBEGIN(modernize-avoid-c-arrays)
// HIPRTC compiles this file as an embedded include, with no standard library: C arrays,
// macros and C-style code are deliberate, and its arithmetic types are part of the kernels.

/// One dense convolution in the kernels' roles: `in` (n, ci, di, hi, wi) is streamed once and
/// reduced over its ci channels into the ko <= KO narrow channels of `out` (n, ko, dOut, ho, wo).
/// Forward: in = x, out = y, ci = C, ko = K. Backward data: in = dy, out = dx, ci = K, ko = C.
/// Strides, dilations and paddings are the graph's (forward roles); the width stride is the
/// config's. 1-D and 2-D problems lead with extent-1 axes.
///
/// A block owns a segment of `ds` output planes, a band of `rs` output rows and a strip of `tw`
/// output columns of one image (blocks numbered segment fastest, then strip, band, image).
struct ConvNarrowArgs
{
    const void* in;
    void* out;
    const void* w; ///< the graph's filter (k, c, t, r, s), read by the prepare kernel
    void* wp; ///< the prepared filter (layout: ConvNarrow.cpp)
    long long is[5]; ///< element strides of in: n, ci, d, h, w
    long long os[5]; ///< element strides of out: n, ko, d, h, w
    long long ws[5]; ///< element strides of w: k, c, t, r, s
    long long inImageBytes; ///< bytes of one in image, its first element to one past its last
    long long outImageBytes;
    int n;
    int ci;
    int ko;
    int cpad; ///< ci rounded up to the reduction step of the prepared filter
    int di;
    int hi;
    int wi;
    int dOut;
    int ho;
    int wo;
    int pd; ///< the graph's pre-padding
    int ph;
    int pw;
    int sd; ///< the graph's depth and height stride (the width's is the config's)
    int sh;
    int dd; ///< the graph's dilation
    int dh;
    int dw;
    int tw; ///< output columns a strip
    int strips;
    int rs; ///< output rows a band
    int bands;
    int ds; ///< output planes a segment
    int segs;
    int blocks; ///< n x segs x bands x strips
    int wGraphK; ///< the graph filter's output channels (prepare kernel)
    int wGraphC; ///< its input channels
    int flip; ///< convolution mode: the filter reflected
    int prepared; ///< elements of wp
};

/// One dense backward-weights convolution in the kernel's roles (ConvNarrowWrw.cpp): dw of a
/// layer with at most 8 channels on one side. `in` (n, ci, di, hi, wi) is the wide operand, whose
/// pixels the reduction runs over; `nar` (n, ko, dn, hn, wn) the narrow one, each in pixel paired
/// with the narrow pixel every tap reaches: an output layer (K <= 8) has in = x, nar = dy, and the
/// x pixel q meets dy pixel (q + pad - tap * dilation) / stride where that is exact (MAPB 1); a
/// stem (C <= 8) has in = dy, nar = x, and the dy pixel p meets x pixel p * stride - pad + tap *
/// dilation (MAPB 0). Strides, dilations and paddings are the graph's.
///
/// The main kernel's blocks each own `cb` wide channels (a channel block) and a contiguous run of
/// the steps of the reduction (image, in plane, band of RB in rows, strip of WR in columns, in
/// that order, strip fastest): `steps` in all, cut into `splits` runs. Each block writes fp32
/// partial sums [split][wide channel][column] to `part`; the reduce kernel adds the splits in
/// order and writes dw.
struct ConvNarrowWrwArgs
{
    const void* in;
    const void* nar;
    void* dw;
    float* part;
    long long is[5]; ///< element strides of in: n, ci, d, h, w
    long long ns[5]; ///< element strides of nar: n, ko, d, h, w
    long long ws[5]; ///< element strides of dw: k, c, t, r, s
    long long inImageBytes;
    long long narImageBytes;
    int n;
    int ci;
    int ko;
    int di;
    int hi;
    int wi;
    int dn;
    int hn;
    int wn;
    int pd; ///< the graph's pre-padding
    int ph;
    int pw;
    int sd; ///< the graph's stride
    int sh;
    int sw;
    int dd; ///< the graph's dilation
    int dh;
    int dw2;
    int strips; ///< ceil(wi / WR)
    int bands; ///< ceil(hi / RB)
    int steps; ///< n x di x bands x strips
    int splits;
    int cblocks; ///< ceil(ci / channels a block)
    int flip; ///< convolution mode: the filter reflected
    int wideIsC; ///< 1: in's channels are the filter's input channels (an output layer)
    int dwElems; ///< elements of dw (K x C x taps)
};

// NOLINTEND(modernize-avoid-c-arrays)
