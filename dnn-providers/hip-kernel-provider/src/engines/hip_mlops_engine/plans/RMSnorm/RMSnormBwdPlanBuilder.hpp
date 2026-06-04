// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "HipKernelContext.hpp"
#include "HipKernelHandle.hpp"
#include "HipKernelSettings.hpp"
#include "IDevicePropertyProvider.hpp"
#include "hip/IKernelCompiler.hpp"
#include "hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp"

namespace hip_kernel_provider::rmsnorm
{
class RMSnormBwdPlanBuilder
    : public hipdnn_plugin_sdk::IPlanBuilder<HipKernelHandle, HipKernelSettings, HipKernelContext>
{
public:
    RMSnormBwdPlanBuilder(const IKernelCompiler& kernelCompiler,
                          const IDevicePropertyProvider& devicePropertyProvider);
    ~RMSnormBwdPlanBuilder() override = default;

    // Disallow copy and assignment
    RMSnormBwdPlanBuilder(const RMSnormBwdPlanBuilder&) = delete;
    RMSnormBwdPlanBuilder& operator=(const RMSnormBwdPlanBuilder&) = delete;

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

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> getCustomKnobs(
        const HipKernelHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

private:
    const IKernelCompiler& _kernelCompiler;
    const IDevicePropertyProvider& _devicePropertyProvider;
};

} // hip_kernel_provider::rmsnorm
