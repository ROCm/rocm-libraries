// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The runtime operands of a fused convolution epilogue: the second kernel argument of every conv
// kernel compiled with CONV_EPI 1. Shared verbatim by the host packs, which fill it, and by the
// kernels, which HIPRTC compiles with this file as an embedded include -- so it uses only builtin
// types and no standard headers.
//
// The generated epilogue code (ConvEpilogue.h) depends only on the pattern of the fused graph:
// which operations, which operand dtypes and broadcast kinds. Everything a graph instance
// supplies (operand pointers, clip bounds, slopes, alphas, betas) arrives here at launch, so one
// code object serves every graph of a pattern.

#pragma once

// NOLINTBEGIN(modernize-avoid-c-arrays,modernize-macro-to-enum)
// HIPRTC compiles this file as an embedded include, with no standard library: C arrays and
// macros are deliberate, and its arithmetic types are part of the kernels.

#define CONV_EPI_OPERANDS 4
#define CONV_EPI_SCALARS 8

struct ConvEpiArgs
{
    /// Per-channel operands (bias, scale) and device scalars, in the order the pattern names.
    const void* operand[CONV_EPI_OPERANDS];
    /// For passes that walk the output flat (split reductions): the channel of flat offset o is
    /// (o / chanStride) % chans. 1 for channels-last, the spatial size for channels-first.
    long long chanStride;
    int chans;
    int padding0; // keeps the struct size a multiple of 8 on host and device alike
    /// Values: clip bounds, slopes, alphas, betas and pass-by-value scalar operands.
    float scalar[CONV_EPI_SCALARS];
};

// NOLINTEND(modernize-avoid-c-arrays,modernize-macro-to-enum)
