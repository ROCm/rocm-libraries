// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// --- Batchnorm argument structs ---
// Shared between device kernels and host launch code for ABI compatibility.

struct BatchnormFwdArgs
{
    const void* input;
    const void* scale;
    const void* bias;
    const void* estMean;
    const void* invVar;
    void* output;
    long long c;
    long long hw;
    long long batchSize;
    long long cStride;
    long long hwStride;
    long long batchStride;
};
