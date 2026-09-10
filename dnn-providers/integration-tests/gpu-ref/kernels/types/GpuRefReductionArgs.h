// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// --- Reduction mode enum ---

enum class ReductionMode : int
{
    ADD = 0,
    MUL = 1,
    MIN_OP = 2,
    MAX_OP = 3,
    AMAX = 4,
    AVG = 5,
    NORM1 = 6,
    NORM2 = 7,
    MUL_NO_ZEROS = 8
};

// --- Reduction argument struct ---
// Shared between device kernels and host launch code for ABI compatibility.

struct ReductionArgs
{
    // IO tensors
    const void* input;
    void* output;

    // Index flattening/unflattening metadata
    // NOLINTBEGIN(modernize-avoid-c-arrays)
    long long inputStrides[5];
    long long outputStrides[5];
    long long outputLogicalStrides[5];
    // NOLINTEND(modernize-avoid-c-arrays)

    // Reduction metadata
    long long reductionRank;
    long long reductionDomainSize;
    // NOLINTBEGIN(modernize-avoid-c-arrays)
    long long reductionDomainAxes[5];
    long long reductionDomainStride[5];
    // NOLINTEND(modernize-avoid-c-arrays)
};
