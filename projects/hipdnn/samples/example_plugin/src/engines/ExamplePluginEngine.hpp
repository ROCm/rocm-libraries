// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include "ExamplePluginHandle.hpp"
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace example_plugin
{

/// Engine implementation for the example plugin.
///
/// This engine manages a collection of plan builders and delegates
/// applicability checks and plan creation to them.
class ExamplePluginEngine : public hipdnn_plugin_sdk::IEngine<ExamplePluginHandle,
                                                              ExamplePluginSettings,
                                                              ExamplePluginContext>
{
public:
    explicit ExamplePluginEngine(int64_t id);

    int64_t id() const override;

    bool isApplicable(ExamplePluginHandle& handle,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(ExamplePluginHandle& handle,
                    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    size_t getMaxWorkspaceSize(
        const ExamplePluginHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const override;

    void initializeExecutionContext(
        const ExamplePluginHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        ExamplePluginContext& executionContext) const override;

    void addPlanBuilder(
        std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<ExamplePluginHandle,
                                                        ExamplePluginSettings,
                                                        ExamplePluginContext>> planBuilder);

private:
    int64_t _id;
    std::vector<std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<ExamplePluginHandle,
                                                                ExamplePluginSettings,
                                                                ExamplePluginContext>>>
        _planBuilders;
};

} // namespace example_plugin
