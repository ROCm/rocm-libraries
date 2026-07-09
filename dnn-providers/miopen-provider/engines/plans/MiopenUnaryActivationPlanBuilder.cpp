// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <memory>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/plans/MiopenUnaryActivationPlan.hpp"
#include "engines/plans/MiopenUnaryActivationPlanBuilder.hpp"

namespace miopen_plugin
{

bool MiopenUnaryActivationPlanBuilder::isApplicable(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return _isSupportedFn(opGraph);
}

size_t MiopenUnaryActivationPlanBuilder::getMaxWorkspaceSize(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const HipdnnMiopenSettings& executionSettings) const
{
    // Unary activations do not require workspace memory.
    return 0u;
}

void MiopenUnaryActivationPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] HipdnnMiopenSettings& executionSettings) const
{
    // No execution settings are needed for unary activations.
}

void MiopenUnaryActivationPlanBuilder::buildPlan(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    HipdnnMiopenContext& executionContext) const
{
    // Preconditions are validated in isApplicable; no need to re-check here.
    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    HIPDNN_PLUGIN_LOG_INFO("Building " << _opName << " plan for node: " << nodeName);

    const auto& attrs
        = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

    auto plan = std::make_unique<MiopenUnaryActivationPlan>(attrs, opGraph.getTensorMap());
    executionContext.setPlan(std::move(plan));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
    MiopenUnaryActivationPlanBuilder::getCustomKnobs(
        [[maybe_unused]] const HipdnnMiopenHandle& handle,
        [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    // Unary activations do not expose any custom knobs.
    return {};
}

} // namespace miopen_plugin
