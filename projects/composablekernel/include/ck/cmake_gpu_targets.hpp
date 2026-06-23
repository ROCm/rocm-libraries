// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

/// @file
/// @brief Compile-time query helpers for GPU targets configured through CMake.

namespace ck {

// Constexpr query functions for CMake-configured GPU targets.
//
// These allow compile-time queries against the set of GPU targets configured
// via CMake (CK_CMAKE_GPU_TARGET_IDS), replacing the need for separate
// CMake-injected feature macros (CK_USE_GFX950, CK_USE_XDL, etc.).
//
// Usage:
//   if constexpr (ck::cmakeTargetsContain(0x0950)) { ... }  // GFX950
//   if constexpr (ck::cmakeTargetsContainFamily(0x09)) { ... }  // GFX9 family

/**
 * @brief Check if a specific GPU target ID is present in the CMake-configured target list.
 * @param target The target ID hex value to search for (e.g. 0x0950 for GFX950).
 * @return true if target is in CK_CMAKE_GPU_TARGET_IDS, false otherwise.
 */
inline static constexpr bool cmakeTargetsContain([[maybe_unused]] uint32_t target)
{
#ifdef CK_CMAKE_GPU_TARGET_IDS
    constexpr uint32_t ids[] = {CK_CMAKE_GPU_TARGET_IDS};
    for(auto id : ids)
    {
        if(id == target)
        {
            return true;
        }
    }
#endif
    return false;
}

/**
 * @brief Check if any CMake-configured GPU target belongs to the specified family.
 * @details Target IDs use a 16-bit layout: family ID in the high byte and chip variant
 *          in the low byte. For example, gfx908 is 0x0908 and gfx90a is 0x090A,
 *          so both belong to the 0x09 family.
 * @param family The target family hex value to search for (e.g. 0x09 for gfx9).
 * @return true if any target in CK_CMAKE_GPU_TARGET_IDS belongs to family, false otherwise.
 */
inline static constexpr bool cmakeTargetsContainFamily([[maybe_unused]] uint32_t family)
{
#ifdef CK_CMAKE_GPU_TARGET_IDS
    constexpr uint32_t ids[] = {CK_CMAKE_GPU_TARGET_IDS};
    for(auto id : ids)
    {
        if((id >> 8) == family)
        {
            return true;
        }
    }
#endif
    return false;
}

/**
 * @brief Check if the CMake-configured target list contains ONLY the specified target ID.
 * @param target The target ID hex value to check for exclusivity (e.g. 0x0950 for GFX950).
 * @return true if CK_CMAKE_GPU_TARGET_IDS contains only this target, false otherwise.
 *         Returns false when CK_CMAKE_GPU_TARGET_IDS is not defined.
 */
inline static constexpr bool cmakeTargetsContainOnly([[maybe_unused]] uint32_t target)
{
#ifdef CK_CMAKE_GPU_TARGET_IDS
    constexpr uint32_t ids[] = {CK_CMAKE_GPU_TARGET_IDS};
    for(auto id : ids)
    {
        if(id != target)
        {
            return false;
        }
    }
    return true;
#endif
    return false;
}

} // namespace ck
