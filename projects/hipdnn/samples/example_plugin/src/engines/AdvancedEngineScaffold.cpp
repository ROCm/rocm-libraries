// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "AdvancedEngineScaffold.hpp"

#include <hipdnn_data_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace example_plugin
{

AdvancedEngineScaffold::AdvancedEngineScaffold(int64_t id)
    : _id(id)
{
}

int64_t AdvancedEngineScaffold::id() const
{
    return _id;
}

bool AdvancedEngineScaffold::isApplicable(
    ExamplePluginHandle& /*handle*/,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const
{
    // TODO: Implement applicability checks for your operation.
    // See ExamplePluginEngine + ReluPlanBuilder for a working example.
    return false;
}

void AdvancedEngineScaffold::getDetails(
    ExamplePluginHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    hipdnnPluginConstData_t& detailsOut) const
{
    // TODO: Add custom knobs here if your engine needs configuration options.
    flatbuffers::FlatBufferBuilder builder;
    auto engineDetails = hipdnn_data_sdk::data_objects::CreateEngineDetails(builder, _id);
    builder.Finish(engineDetails);

    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detachedBuffer));
}

size_t AdvancedEngineScaffold::getMaxWorkspaceSize(
    const ExamplePluginHandle& /*handle*/,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/) const
{
    // TODO: Return the workspace size your engine requires.
    return 0;
}

void AdvancedEngineScaffold::initializeExecutionContext(
    const ExamplePluginHandle& /*handle*/,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    ExamplePluginContext& /*executionContext*/) const
{
    // TODO: Build and set a plan on the execution context.
    // See ReluPlanBuilder::buildPlan() for a working example.
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        "AdvancedEngineScaffold is a placeholder and cannot execute operations");
}

} // namespace example_plugin
