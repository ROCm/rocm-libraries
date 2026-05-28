// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Common Pipeline enum. No runtime, no CK deps.

#pragma once

#include <cstdint>

namespace ck_common {

/// Pipeline variants for memory/compute optimization.
/// Matches tile_engine PIPELINE_MAP for full compatibility.
enum class Pipeline : uint8_t
{
    Mem,          // Memory-bound pipeline
    CompV1,       // Compute pipeline v1
    CompV2,       // Compute pipeline v2
    CompV3,       // Compute pipeline v3
    CompV4,       // Compute pipeline v4 (double buffering)
    CompV5,       // Compute pipeline v5
    CompV6,       // Compute pipeline v6
    PreShuffleV1, // Weight preshuffle pipeline v1
    PreShuffleV2  // Weight preshuffle pipeline v2 (optimized)
};

} // namespace ck_common
