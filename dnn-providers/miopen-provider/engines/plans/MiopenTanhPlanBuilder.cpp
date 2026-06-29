// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "MiopenTanhPlanBuilder.hpp"
#include "engines/plans/MiopenTanhApplicabilityChecks.hpp"
#include "engines/plans/MiopenTanhPlan.hpp"

namespace miopen_plugin
{

bool MiopenTanhPlanBuilder::isApplicable(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return tanh_applicability::isTanhSupported(opGraph);
}

size_t MiopenTanhPlanBuilder::getMaxWorkspaceSize(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const HipdnnMiopenSettings& executionSettings) const
{
    // TANH operations do not require workspace memory.
    return 0u;
}

void MiopenTanhPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] HipdnnMiopenSettings& executionSettings) const
{
    // No execution settings are needed for TANH operations.
}

void MiopenTanhPlanBuilder::buildPlan(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    HipdnnMiopenContext& executionContext) const
{
    // Preconditions are validated in isApplicable; no need to re-check here.
    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    HIPDNN_PLUGIN_LOG_INFO("Building TANH plan for node: " << nodeName);

    const auto& attrs
        = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

    auto plan = std::make_unique<MiopenTanhPlan>(attrs, opGraph.getTensorMap());
    executionContext.setPlan(std::move(plan));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> MiopenTanhPlanBuilder::getCustomKnobs(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    // TANH operations do not expose any custom knobs.
    return {};
}

} // namespace miopen_plugin
