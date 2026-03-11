// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ExamplePluginHandle.hpp"
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

namespace example_plugin
{

/// Scaffold engine that demonstrates the engine interface structure.
///
/// This engine always reports itself as not applicable. It serves as a
/// starting point for implementing a new engine -- replace the TODO
/// sections with real logic.
class AdvancedEngineScaffold : public hipdnn_plugin_sdk::IEngine<ExamplePluginHandle,
                                                                 ExamplePluginSettings,
                                                                 ExamplePluginContext>
{
public:
    explicit AdvancedEngineScaffold(int64_t id);

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

private:
    int64_t _id;
};

} // namespace example_plugin
