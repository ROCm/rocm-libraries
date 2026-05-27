// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: types — index_t, GpuTarget. No runtime, no CK deps.
//
// Shared type definitions for rocm_ck.
//
// Architecture properties (wavefront size, CDNA/RDNA, valid tiles)
// live in arch_properties.hpp, which depends on this header.

#pragma once

#include <cstdint>

namespace rocm_ck {

/// Fixed-width integer type for kernel index/size arguments.
/// Matches ck_tile::index_t but avoids pulling in CK Tile headers.
using index_t = std::int32_t;

/// 64-bit index type for large strides (batch_stride * nhead can exceed int32).
/// Matches ck_tile::long_index_t.
using long_index_t = std::int64_t;

/// GPU ISA target for architecture-specific validation.
///
/// Uses ISA target identifiers (matching -mcpu flags), not marketing names.
/// For target sets (e.g., "all CDNA" or "gfx942+"), see TargetSet in
/// arch_properties.hpp.
enum class GpuTarget
{
    gfx90a,  // CDNA 2
    gfx942,  // CDNA 3
    gfx950,  // CDNA 4
    gfx1100, // RDNA 3
    gfx1101, // RDNA 3
    gfx1102, // RDNA 3
    gfx1150, // RDNA 3.5
    gfx1151, // RDNA 3.5
    gfx1200, // RDNA 4
    gfx1201, // RDNA 4
};

} // namespace rocm_ck
