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
    /// rank ordering). Populated by initializeExecutionSettings(); getMaxWorkspaceSize()
    /// reads it from here since it receives only this Settings object.
    hipdnn_plugin_sdk::ingestor::KnobFilter ingestorKnobFilter;
#endif
};
