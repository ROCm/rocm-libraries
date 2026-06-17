// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include <string>

namespace asm_sdpa_engine
{

class SdpaBwdPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>
{
public:
    bool isApplicable(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const Handle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const Settings& executionSettings) const override;

    void initializeExecutionSettings(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        Settings& executionSettings) const override;

    void buildPlan(const Handle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   Context& executionContext) const override;

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getCustomKnobs(
            const Handle& handle,
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
};

namespace bwd_dispatch
{

// Pipeline-stage tag used by the registry-lookup helper to select which
// CSV-derived registry to walk.
enum class PipelineStage
{
    ODO,
    DQDKDV,
    DQ_CONVERT
};

// Value stored in the bf16_cvt CSV column for FP16 rows (where rounding mode
// is not applicable). Shared between the builder and its unit tests.
inline constexpr int BF16_CVT_FP16_SENTINEL = 3;

// Resolves a registry row matching the supplied dispatch tuple. Returns the
// composite key (arch + knl_name) on a hit, or an empty string when no row
// in the chosen registry matches. The bf16Cvt argument is the integer the
// CSV stores for that column (0/1/2 for the bf16 rounding modes; 3 for fp16).
// Exposed for unit testing of the dispatch logic.
std::string lookupKernelNameKey(PipelineStage stage,
                                const std::string& archId,
                                const std::string& dataType,
                                int hdimQ,
                                int hdimV,
                                int mask,
                                int atomic32,
                                int pssk,
                                int pddv,
                                int mode,
                                int bf16Cvt);

} // namespace bwd_dispatch

} // namespace asm_sdpa_engine
