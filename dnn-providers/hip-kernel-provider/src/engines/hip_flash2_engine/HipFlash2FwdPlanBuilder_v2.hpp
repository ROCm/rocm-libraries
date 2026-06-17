// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// HipFlash2FwdPlanBuilder: Full IPlanBuilder<Handle,Settings,Context> implementation
// for Flash-Attention 2 V7. Mirrors asm_sdpa_engine::SdpaFwdPlanBuilder.
//
// This header supersedes the stub HipFlash2FwdPlanBuilder in HipFlash2FwdPlanBuilder.hpp.
// The implementation lives in HipFlash2FwdPlanBuilder.cpp.

#pragma once

#include "HipFlash2FwdPlanBuilder.hpp"   // Flash2FwdParams, useFlash2ForShape
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/Knob_generated.h>

#include <vector>

namespace hip_flash2_engine
{

/**
 * @brief Plan builder for Flash-Attention 2 V7 forward pass.
 *
 * Checks applicability (FP16, gfx942/gfx950, head_dim ∈ {64,128},
 * no dropout/alibi/group-batch), extracts SDPA graph parameters,
 * loads the arch-specific precompiled .co, and stores a HipFlash2FwdPlan
 * in the execution context.
 */
class HipFlash2FwdPlanBuilder
    : public hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>
{
public:
    // ── IPlanBuilder interface ────────────────────────────────────────────────

    bool isApplicable(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const Settings& executionSettings) const override;

    void initializeExecutionSettings(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        Settings& executionSettings) const override;

    /**
     * @brief Build a HipFlash2FwdPlan from the graph.
     *
     * Extracts Q/K/V/O UIDs, sequence lengths, head counts, and strides.
     * Loads the arch-specific precompiled .co via hipModuleLoad.
     * Stores the resulting HipFlash2FwdPlan in executionContext.
     * The plan's execute() dispatches launch via hipModuleLaunchKernel.
     */
    void buildPlan(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        Context& executionContext) const override;

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> getCustomKnobs(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

private:
    /// Extract Flash2FwdParams from a validated SDPA graph.
    Flash2FwdParams extractParams(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const;
};

} // namespace hip_flash2_engine
