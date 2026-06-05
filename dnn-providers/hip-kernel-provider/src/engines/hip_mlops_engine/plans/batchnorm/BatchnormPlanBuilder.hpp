// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "device/IDevicePropertyProvider.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

namespace hip_kernel_provider::batchnorm
{

using namespace compilation;
using namespace device;

class BatchnormPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>
{
public:
    BatchnormPlanBuilder(const IKernelCompiler& kernelCompiler,
                         const IDevicePropertyProvider& devicePropertyProvider);
    ~BatchnormPlanBuilder() override = default;

    // Disallow copy and assignment
    BatchnormPlanBuilder(const BatchnormPlanBuilder&) = delete;
    BatchnormPlanBuilder& operator=(const BatchnormPlanBuilder&) = delete;

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

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> getCustomKnobs(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

private:
    const IKernelCompiler& _kernelCompiler;
    const IDevicePropertyProvider& _devicePropertyProvider;
};

} // namespace hip_kernel_provider::batchnorm
