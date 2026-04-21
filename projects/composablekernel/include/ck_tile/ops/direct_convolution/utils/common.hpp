// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::direct_conv {

enum class SwizzleType
{
    None,   // No swizzling; direct mapping from global to LDS to registers.
    CyclicShift, // Swizzling using cyclic-shift modular addition (SwizzleT).
    XOR      // Swizzling using XOR (SwizzleXOR).
};

enum class Version
{
    v1, // Default kernel version that uses either SwizzleT or SwizzleXOR but only primitve CK Tile abstractions.
    v2, // Native CK Tile kernel that use XOR swizzle and tile distribution to express the tensor coordinate to thread id mapping.
    v3, // CK Tile kernel using async_load_tile with tile windows for input loads.
};

enum class EpilogueType
{
    RegistersToLdsToGlobalMemory, // Use LDS to collect the MFMA results before writing to global memory
    RegistersToGlobalMemory       // Write MFMA results directly from registers to global memory
};

} // namespace ck_tile::direct_conv
