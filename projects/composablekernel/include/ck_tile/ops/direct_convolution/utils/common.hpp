// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::direct_conv {

enum class Version
{
    v1, // Default kernel version that uses either SwizzleT or SwizzleXOR but only primitve CK Tile abstractions.
    v2, // Native CK Tile kernel that use XOR swizzle and tile distribution to express the tensor coordinate to thread id mapping.
    v3, // CK Tile kernel using async_load_tile with tile windows for input loads.
};

} // namespace ck_tile::direct_conv
