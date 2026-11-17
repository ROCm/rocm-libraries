// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Shared argument struct for GPU reference validator kernels.
// Included by both device code (HipRTC) and host launch code.
// Only POD types allowed — no host or device includes.

#pragma once

// NOLINTBEGIN(misc-non-private-member-variables-in-classes,
//             readability-identifier-naming,
//             modernize-avoid-c-arrays)
struct ValidatorArgs
{
    const void* reference;
    const void* implementation;
    int* failureFlag; // single int: 0 = all passed, 1 = any element failed
    long long totalElements;
    double absoluteTolerance;
    double relativeTolerance;

    // Stride-based indexing for non-contiguous (unpacked) tensors.
    // ndim == 0 means packed mode (use linear indexing, ignore strides/dims).
    // ndim > 0 means strided mode (decompose linear index using dims/strides).
    long long refStrides[8];
    long long implStrides[8];
    long long dims[8];
    int ndim;
};

// Running totals for the relative-RMS validator, zero-initialised by the host and
// folded into by one thread per block. Magnitudes are non-negative, so their IEEE-754
// bit patterns order the same way as the values: atomicMax on the bits is a max on
// the doubles.
struct RmsAccumulators
{
    double squareDifference;
    unsigned long long maxRefMagnitudeBits;
    unsigned long long maxImplMagnitudeBits;
    int nanOrInf; // 0 = every element finite, 1 = at least one NaN/Inf seen
};

struct RmsValidatorArgs
{
    const void* reference;
    const void* implementation;
    RmsAccumulators* accumulators;
    long long totalElements;

    // Same layout contract as ValidatorArgs: ndim == 0 is packed.
    long long refStrides[8];
    long long implStrides[8];
    long long dims[8];
    int ndim;
};
// NOLINTEND(misc-non-private-member-variables-in-classes,
//           readability-identifier-naming,
//           modernize-avoid-c-arrays)
