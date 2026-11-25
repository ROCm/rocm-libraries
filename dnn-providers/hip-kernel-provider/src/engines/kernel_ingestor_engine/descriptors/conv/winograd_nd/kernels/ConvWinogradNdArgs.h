// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

// Forward-form correlation, generic positive element strides in N,C,H,W roles.
// U[G][P][CO/G][CI/G], V[G][P][CI/G][tiles], M[G][P][CO/G][tiles], all FP32.
// tileBegin is a global spatial/batch tile index; only V/M are spatially chunked.
// NOLINTBEGIN(modernize-avoid-c-arrays)
struct KNdArgs
{
    const void* in;
    const void* w;
    void* out;
    float* u;
    float* v;
    float* product;
    long long n, ci, co, groups;
    long long input[2], output[2], pad[2], dilation[2];
    long long inStride[4], outStride[4], wStride[4];
    long long axisTiles[2], residues[2];
    long long tileBegin;
    int tiles;
    int rotate;
};
// NOLINTEND(modernize-avoid-c-arrays)
