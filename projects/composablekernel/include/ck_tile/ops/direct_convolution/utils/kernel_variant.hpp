// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "conv_params.hpp"
#include "launch_params.hpp"
#include <hip/hip_runtime.h>


namespace ck_tile::direct_conv
{

// Define the matching functions for a conv2d kernel variant.
struct KernelVariant
{
    // Returns true if this variant supports the given parameters.
    bool (*is_applicable)(const Conv2dParams& par);

    // Returns true if config_idx is compatible with the given parameters.
    bool (*config_is_compatible)(const Conv2dParams& par, int config_idx);

    // Returns the launch parameters for a given config index.
    LaunchParams (*get_launch_params)(int config_idx, const Conv2dParams& par);

    // Launch the kernel for a given config index.
    void (*launch)(int config_idx,
                   const LaunchParams& lp,
                   const Conv2dParams& par,
                   const void* in,
                   const void* wei,
                   void* out,
                   void* workspace,
                   hipStream_t stream);

    // Returns the workspace size in bytes (0 if no workspace needed).
    size_t (*get_workspace_size)(int config_idx, const Conv2dParams& par);

    // Number of configurations used by this kernel.
    int num_configs;
};

} // namespace ck_tile::direct_conv
