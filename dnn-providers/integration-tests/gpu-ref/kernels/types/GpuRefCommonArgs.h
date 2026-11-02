// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// --- Common kernel argument structs ---
// Shared between device kernels and host launch code for ABI compatibility.

struct ScaleUniformArgs
{
    const void* src;
    void* dst;
    long long count;
    double minValue;
    double maxValue;
};

// --- Domain fill kinds ---
// The kinds a harness fills an input with beyond a uniform draw (see GpuRefFillKinds.cpp).

enum class FillKind : int
{
    ZEROS = 0,
    NONZERO = 1,
    LARGE = 2,
    SUBNORMAL = 3,
    SPECIAL = 4
};

/// Most dimensions a filled tensor can have.
#define GPU_REF_FILL_MAX_RANK 8

/// One tensor's fill. The kernel runs one thread per logical element, in row-major order, and
/// writes it at the offset the dims and strides give, so the bytes between elements of a padded
/// tensor are never written. `uniform` holds two draws per element in (0, 1], the first
/// `count` of them for the value and the next `count` for the redraw or the sign; ZEROS reads
/// none. Every value that is a bit pattern of the element type (planted non-finite,
/// subnormal codes) is written as raw bits, so no conversion can round, saturate or flush it.
struct FillKindArgs
{
    const float* uniform;
    void* dst;
    long long count;
    long long rank;
    // NOLINTBEGIN(modernize-avoid-c-arrays)
    long long dims[GPU_REF_FILL_MAX_RANK];
    long long strides[GPU_REF_FILL_MAX_RANK];
    // NOLINTEND(modernize-avoid-c-arrays)
    int kind;
    unsigned int seed;
    /// NONZERO and SPECIAL's free share: the range. LARGE: the magnitude range [lo, hi].
    double lo;
    double hi;
    /// NONZERO: the value an element takes when both draws round to zero.
    double fallback;
    /// SUBNORMAL: the largest subnormal magnitude code, a code k being k * the smallest subnormal.
    unsigned long long maxSubnormalCode;
    /// SPECIAL: how many elements are planted, and the raw bits planted in them.
    unsigned long long quota;
    unsigned long long plantedBits;
};
