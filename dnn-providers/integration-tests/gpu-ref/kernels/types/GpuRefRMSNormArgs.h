// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// --- RMSNorm argument structs ---
// Shared between device kernels and host launch code for ABI compatibility.

struct RMSNormFwdArgs
{
    const void* input;
    const void* scale;
    const void* bias;
    void* output;
    void* invRms;
    long long innerSize;
    long long stride;
    double eps;
};

struct RMSNormBwdArgs
{
    const void* gradOutput;
    const void* input;
    const void* scale;
    const void* invRms;
    void* gradInput;
    void* gradScale;
    void* gradBias;
    long long innerSize;
    long long outerSize;
    long long stride;
};
