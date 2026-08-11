// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/asm_sdpa_engine/plans/SdpaBwdParams.hpp"

#include <optional>

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <hipdnn_plugin_sdk/ingestor/GenericPlanBuilder.hpp>
#endif

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

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    /// The kernel ingestor's per-call knob filter (RFC 0017 §4's catalog -> filter ->
    /// rank ordering). See GenericPlanBuilder.hpp's KnobFilter doc for why this lives
    /// here rather than being threaded through some other channel:
    /// getMaxWorkspaceSize() receives only this Settings object, not an IEngineConfig,
    /// so a knob setting reaches it only via a field initializeExecutionSettings()
    /// populated first.
    hipdnn_plugin_sdk::ingestor::KnobFilter ingestorKnobFilter;
#endif
};
