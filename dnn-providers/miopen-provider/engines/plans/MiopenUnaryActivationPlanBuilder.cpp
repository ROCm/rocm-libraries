// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <memory>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/plans/MiopenUnaryActivationChecks.hpp"
#include "engines/plans/MiopenUnaryActivationPlan.hpp"
#include "engines/plans/MiopenUnaryActivationPlanBuilder.hpp"

namespace miopen_plugin
{

bool MiopenUnaryActivationPlanBuilder::isApplicable(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    // Execute-time override shapes can diverge from the compile-time dims this
    // builder matched exactly; the plan bakes those dims into the MIOpen
    // descriptors it builds, so decline rather than risk a mismatch (RFC 0008 §4.6).
    if(opGraph.getGraph().is_override_shape_enabled())
    {
        HIPDNN_PLUGIN_LOG_INFO("Unary activation plan builder does not support override shapes");
        return false;
    }

    return unary_activation_applicability::isSupported(opGraph);
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

    const auto& attrs
        = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

    HIPDNN_PLUGIN_LOG_INFO("Building "
                           << hipdnn_flatbuffers_sdk::data_objects::EnumNamePointwiseMode(
                                  attrs.operation())
                           << " plan for node: " << nodeName);

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
