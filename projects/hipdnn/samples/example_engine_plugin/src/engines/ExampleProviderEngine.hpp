// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include "ExampleProviderHandle.hpp"
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace example_provider
{

// TEMPLATE ADAPTATION: Copy as-is and rename the class. This generic engine coordinator delegates
// to PlanBuilders and contains no operation-specific logic. The only change needed is the class
// name.

/// Generic engine coordinator. Copy this class as-is for your own plugin.
///
/// This engine contains no operation-specific logic. It manages a collection
/// of PlanBuilders and delegates all work (applicability checks, knob
/// reporting, workspace sizing, and plan creation) to them. The typical
/// pattern is one PlanBuilder per engine (1:1), but multiple PlanBuilders
/// can be added to a single engine via addPlanBuilder() if needed.
/// Typically, customized behavior can be added by writing your own
/// PlanBuilder, not by modifying this class.
class ExampleProviderEngine : public hipdnn_plugin_sdk::IEngine<ExampleProviderHandle,
                                                                ExampleProviderSettings,
                                                                ExampleProviderContext>
{
public:
    explicit ExampleProviderEngine(int64_t id);

    int64_t id() const override;

    bool isApplicable(
        ExampleProviderHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(ExampleProviderHandle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    size_t getMaxWorkspaceSize(const ExampleProviderHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
                                   engineConfig) const override;

    void initializeExecutionContext(
        const ExampleProviderHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        ExampleProviderContext& executionContext) const override;

    void addPlanBuilder(
        std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<ExampleProviderHandle,
                                                        ExampleProviderSettings,
                                                        ExampleProviderContext>> planBuilder);

private:
    int64_t _id;
    std::vector<std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<ExampleProviderHandle,
                                                                ExampleProviderSettings,
                                                                ExampleProviderContext>>>
        _planBuilders;
};

} // namespace example_provider
