// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <memory>

#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/plans/MiopenResampleFwdChecks.hpp"
#include "engines/plans/MiopenResampleFwdPlan.hpp"
#include "engines/plans/MiopenResampleFwdPlanBuilder.hpp"

namespace miopen_plugin
{

bool MiopenResampleFwdPlanBuilder::isApplicable(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    if(opGraph.getGraph().is_override_shape_enabled())
    {
        HIPDNN_PLUGIN_LOG_INFO("Resample forward plan builder does not support override shapes");
        return false;
    }

    return resample_fwd_applicability::isSupported(opGraph);
}

size_t MiopenResampleFwdPlanBuilder::getMaxWorkspaceSize(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const HipdnnMiopenSettings& executionSettings) const
{
    const MiopenResampleFwdPlan plan(resample_fwd_applicability::resolve(opGraph));
    return plan.getWorkspaceSize(handle);
}

void MiopenResampleFwdPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] HipdnnMiopenSettings& executionSettings) const
{
}

void MiopenResampleFwdPlanBuilder::buildPlan(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    HipdnnMiopenContext& executionContext) const
{
    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto& attrs
        = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes>();

    HIPDNN_PLUGIN_LOG_INFO(
        "Building resample forward "
        << hipdnn_flatbuffers_sdk::data_objects::EnumNameResampleMode(attrs.resample_mode())
        << " plan for node: " << nodeWrapper.name());

    auto plan
        = std::make_unique<MiopenResampleFwdPlan>(resample_fwd_applicability::resolve(opGraph));
    executionContext.setPlan(std::move(plan));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
    MiopenResampleFwdPlanBuilder::getCustomKnobs(
        [[maybe_unused]] const HipdnnMiopenHandle& handle,
        [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace miopen_plugin
