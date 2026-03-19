// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "SdpaKernelContext.hpp"
#include "SdpaKernelHandle.hpp"
#include "SdpaKernelSettings.hpp"

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace sdpa_kernel_provider
{

// template <typename SdpaKernelHandle, typename SdpaKernelSettings, typename SdpaKernelContext>
// class PlanBuilder
class SdpaKernelPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<SdpaKernelHandle,
                                                                     SdpaKernelSettings,
                                                                     SdpaKernelContext>
{
public:
    bool isApplicable(const SdpaKernelHandle& handle,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const SdpaKernelHandle& handle,
                               const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const SdpaKernelSettings& executionSettings) const override;

    void initializeExecutionSettings(
        const SdpaKernelHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        SdpaKernelSettings& executionSettings) const override;

    void buildPlan(const SdpaKernelHandle& handle,
                   const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   SdpaKernelContext& executionContext) const override;

    std::vector<hipdnn_data_sdk::data_objects::KnobT>
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getCustomKnobs(const SdpaKernelHandle& handle,
                       const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
};

}
