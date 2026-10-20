// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <memory>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/plans/MiopenReductionChecks.hpp"
#include "engines/plans/MiopenReductionPlan.hpp"
#include "engines/plans/MiopenReductionPlanBuilder.hpp"

namespace miopen_plugin
{

bool MiopenReductionPlanBuilder::isApplicable(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    // The whole gate, including the override-shape check, lives in MiopenReductionChecks so it
    // can be unit tested without a MIOpen handle.
    return reduction_applicability::isSupported(opGraph);
}

size_t MiopenReductionPlanBuilder::getMaxWorkspaceSize(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const HipdnnMiopenSettings& executionSettings) const
{
    // Building the plan only creates descriptors and queries MIOpen; no kernel is compiled.
    const auto& attrs
        = opGraph.getNodeWrapper(0)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ReductionAttributes>();
    const MiopenReductionPlan plan(handle, attrs, opGraph.getTensorMap());
    return plan.getWorkspaceSize(handle);
}

void MiopenReductionPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] HipdnnMiopenSettings& executionSettings) const
{
    // Legacy reduce has no tunable solver, so no execution settings apply.
}

void MiopenReductionPlanBuilder::buildPlan(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    HipdnnMiopenContext& executionContext) const
{
    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    const auto& attrs
        = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::ReductionAttributes>();

    HIPDNN_PLUGIN_LOG_INFO("Building reduction "
                           << hipdnn_flatbuffers_sdk::data_objects::EnumNameReductionMode(
                                  attrs.mode())
                           << " plan for node: " << nodeName);

    auto plan = std::make_unique<MiopenReductionPlan>(handle, attrs, opGraph.getTensorMap());
    executionContext.setPlan(std::move(plan));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> MiopenReductionPlanBuilder::getCustomKnobs(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace miopen_plugin
