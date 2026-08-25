// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Shared argument structs for GPU reference matmul kernels.
// Included by both device code (HipRTC) and host launch code.
// Only POD types allowed — no host or device includes.

#pragma once

// --- Matmul argument structs ---
// Shared between device kernels and host launch code for ABI compatibility.

struct MatmulArgs
{
    const void* a;
    const void* b;
    void* c;

    // Metadata for broadcasting, supports up to 5 dimensions
    // NOLINTBEGIN(modernize-avoid-c-arrays)
    long long aDims[5];
    long long aStrides[5];
    long long bDims[5];
    long long bStrides[5];
    long long cDims[5];
    long long cStrides[5];
    // NOLINTEND(modernize-avoid-c-arrays)
};
