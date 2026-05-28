// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Common Epilogue enum. No runtime, no CK deps.

#pragma once

#include <cstdint>

namespace ck_common {

/// Epilogue strategies for output processing.
/// Matches tile_engine epilogue options for full compatibility.
enum class Epilogue : uint8_t
{
    None,
    Default,       // DefaultGemm2DEpilogue
    CShuffle,      // CShuffleEpilogue (cross-shuffle)
    Bias,          // Bias addition
    Activation,    // Fused activation
    BiasActivation // Fused bias + activation
};

} // namespace ck_common
