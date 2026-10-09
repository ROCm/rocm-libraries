// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// --- Batchnorm argument structs ---
// Shared between device kernels and host launch code for ABI compatibility.

struct BatchnormFwdInfCommonArgs
{
    const void* input;
    const void* scale;
    const void* bias;
    const void* estMean;
    void* output;
    long long c;
    long long hw;
    long long batchSize;
    long long cStride;
    long long hwStride;
    long long batchStride;
};

struct BatchnormFwdInfArgs
{
    const void* invVar;
    BatchnormFwdInfCommonArgs common;
};

struct BatchnormFwdInfWithVarArgs
{
    const void* estVar;
    double epsilon;
    BatchnormFwdInfCommonArgs common;
};

struct BatchnormFwdTrainArgs
{
    const void* input;
    const void* scale;
    const void* bias;
    void* output;
    double epsilon;
    double momentum;
    void* mean;
    void* invVariance;
    const void* prevResultRunningMean;
    const void* prevResultRunningVariance;
    void* nextResultRunningMean;
    void* nextResultRunningVariance;
    long long n;
    long long c;
    long long hw;
};
