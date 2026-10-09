// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// --- Resample mode enum ---

enum class ResampleMode : int
{
    MAXPOOL = 0,
    AVGPOOL_EXCLUDE_PADDING = 1,
    AVGPOOL_INCLUDE_PADDING = 2
};

// --- Padding mode enum ---

enum class PaddingMode : int
{
    NEG_INF_PAD = 0,
    ZERO_PAD = 1
};

// --- Resample argument structs ---
// Shared between device kernels and host launch code for ABI compatibility.

struct ResampleFwdArgs
{
    const void* x;
    void* y;
    void* index;
    long long n;
    long long c;

    // NOLINTBEGIN(modernize-avoid-c-arrays)
    long long xSpatialDims[3];
    long long xStrides[5];
    long long ySpatialDims[3];
    long long yStrides[5];
    long long prePadding[3];
    long long stride[3];
    long long window[3];
    // NOLINTEND(modernize-avoid-c-arrays)
};
