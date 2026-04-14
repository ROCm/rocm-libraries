// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

namespace ck_tile::direct_conv
{

struct LaunchParams
{
    dim3 grid;
    dim3 block_size;
    size_t dynamic_shared_bytes = 0;
};

} // namespace ck_tile::direct_conv
