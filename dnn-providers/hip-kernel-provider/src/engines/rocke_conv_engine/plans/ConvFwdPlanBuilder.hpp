// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

namespace rocke_conv_engine
{

class ConvFwdPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>
{
public:
    bool isApplicable(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const Handle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const Settings& executionSettings) const override;

    void initializeExecutionSettings(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        Settings& executionSettings) const override;

    void buildPlan(const Handle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   Context& executionContext) const override;

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getCustomKnobs(
            const Handle& handle,
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
};

} // namespace rocke_conv_engine
