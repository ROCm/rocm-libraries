// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TEMPLATE REFERENCE: This file demonstrates the PlanBuilder pattern for a pointwise ReLU
// operation. Study the 5 key methods: isApplicable(), getCustomKnobs(), initializeExecutionSettings(),
// getMaxWorkspaceSize(), buildPlan(). Then replace this file with your operation's PlanBuilder. See
// also ConvFwdPlanBuilder for a convolution example of the same pattern.

#pragma once

#include <memory>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "ExampleProviderContext.hpp"
#include "ExampleProviderHandle.hpp"
#include "ExampleProviderSettings.hpp"
#include "hip/IKernelCompiler.hpp"

namespace example_provider
{

/// PlanBuilder for GPU-based ReLU forward operations.
///
/// Handles single-node pointwise RELU_FWD graphs with FLOAT data type.
/// Provides a custom "example.relu.negative_slope" knob for leaky ReLU.
class ReluPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<ExampleProviderHandle,
                                                               ExampleProviderSettings,
                                                               ExampleProviderContext>
{
public:
    explicit ReluPlanBuilder(const IKernelCompiler& compiler);
    ~ReluPlanBuilder() override = default;

    ReluPlanBuilder(const ReluPlanBuilder&) = delete;
    ReluPlanBuilder& operator=(const ReluPlanBuilder&) = delete;

    bool isApplicable(
        const ExampleProviderHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const ExampleProviderHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const ExampleProviderSettings& executionSettings) const override;

    void initializeExecutionSettings(
        const ExampleProviderHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        ExampleProviderSettings& executionSettings) const override;

    void buildPlan(
        const ExampleProviderHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
            engineConfig,
        ExampleProviderContext& executionContext) const override;

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> getCustomKnobs(
        const ExampleProviderHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

private:
    const IKernelCompiler& _compiler;
};

} // namespace example_provider
