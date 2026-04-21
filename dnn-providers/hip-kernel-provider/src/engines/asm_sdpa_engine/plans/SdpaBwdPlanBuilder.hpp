// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipKernelContext.hpp"
#include "HipKernelHandle.hpp"
#include "HipKernelSettings.hpp"

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace asm_sdpa_engine
{

class SdpaBwdPlanBuilder
    : public hipdnn_plugin_sdk::IPlanBuilder<HipKernelHandle, HipKernelSettings, HipKernelContext>
{
public:
    bool isApplicable(
        const HipKernelHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const HipKernelHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const HipKernelSettings& executionSettings) const override;

    void initializeExecutionSettings(
        const HipKernelHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        HipKernelSettings& executionSettings) const override;

    void buildPlan(const HipKernelHandle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   HipKernelContext& executionContext) const override;

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getCustomKnobs(
            const HipKernelHandle& handle,
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
};

} // namespace asm_sdpa_engine
