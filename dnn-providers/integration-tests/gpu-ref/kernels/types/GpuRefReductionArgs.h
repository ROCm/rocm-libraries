// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// --- Reduction argument structs ---
// Shared between device kernels and host launch code for ABI compatibility.

struct ReductionArgs
{
    const void* input;
    const long long* inputStrides;
    void* output;
    const long long* outputStrides;
    const long long* outputShapeStrides;
    long long reductionRank;
    long long reductionDomainSize;
    const long long* reductionDomainAxes;
    const long long* reductionDomainStride;
};
