// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "ExamplePluginContext.hpp"
#include "ExamplePluginHandle.hpp"
#include "ExamplePluginSettings.hpp"
#include "hip/IDevicePropertyProvider.hpp"
#include "hip/IKernelCompiler.hpp"

namespace example_plugin
{

/// PlanBuilder for GPU-based ReLU forward operations.
///
/// Handles single-node pointwise RELU_FWD graphs with FLOAT data type.
/// Provides a custom "example.relu.negative_slope" knob for leaky ReLU.
class ReluPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<ExamplePluginHandle,
                                                               ExamplePluginSettings,
                                                               ExamplePluginContext>
{
public:
    ReluPlanBuilder(const IKernelCompiler& compiler,
                    const IDevicePropertyProvider& devicePropertyProvider);
    ~ReluPlanBuilder() override = default;

    ReluPlanBuilder(const ReluPlanBuilder&) = delete;
    ReluPlanBuilder& operator=(const ReluPlanBuilder&) = delete;

    bool isApplicable(const ExamplePluginHandle& handle,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const ExamplePluginHandle& handle,
                               const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const ExamplePluginSettings& executionSettings) const override;

    void initializeExecutionSettings(
        const ExamplePluginHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        ExamplePluginSettings& executionSettings) const override;

    void buildPlan(
        const ExamplePluginHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        ExamplePluginContext& executionContext) const override;

    std::vector<hipdnn_data_sdk::data_objects::KnobT>
        getCustomKnobs(const ExamplePluginHandle& handle,
                       const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

private:
    const IKernelCompiler& _compiler;
    const IDevicePropertyProvider& _devicePropertyProvider;
};

} // namespace example_plugin
