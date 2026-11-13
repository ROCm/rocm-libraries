// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Shared argument structs for GPU reference layernorm kernels.
// Included by both device code (HipRTC) and host launch code.
// Only POD types allowed — no host or device includes.

#pragma once

// --- Layernorm forward argument structs ---
// Shared between device kernels and host launch code for ABI compatibility.

struct LayernormFwdArgs
{
    const void* x;
    void* y;
    const void* scale;
    const void* bias;
    void* mean;
    void* rstd;
    double epsilon;
};

// --- Layernorm backward argument structs ---
// Shared between device kernels and host launch code for ABI compatibility.

struct LayernormBwdArgs
{
    const void* dy;
    const void* x;
    const void* scale;
    const void* mean;
    const void* rstd;
    void* dx;
    void* workspace;
    double epsilon;
};

// --- Layernorm backward weights argument structs ---
// Shared between device kernels and host launch code for ABI compatibility.

struct LayernormBwdWeightsArgs
{
    const void* dy;
    const void* x;
    const void* mean;
    const void* rstd;
    void* dscale;
    void* dbias;
    const void* workspace;
};
