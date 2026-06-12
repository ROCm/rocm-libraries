// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/asm_sdpa_engine/plans/SdpaBwdParams.hpp"

#include <optional>

/**
 * @brief HIP kernel provider plugin-specific execution settings.
 *
 * This structure holds settings that control HIP kernel execution behavior.
 * Values are populated from engine knobs via initializeExecutionSettings().
 */
struct Settings
{
    /// Accumulator precision for backward SDPA dQ gradient.
    /// nullopt means no user preference (default: A32).
    std::optional<asm_sdpa_engine::AccumulatorType> accumulatorType;
};
