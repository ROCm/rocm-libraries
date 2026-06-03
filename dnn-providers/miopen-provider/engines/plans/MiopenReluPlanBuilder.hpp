// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "HipdnnMiopenContext.hpp"
#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"

namespace miopen_plugin
{

class MiopenReluPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<HipdnnMiopenHandle,
                                                                     HipdnnMiopenSettings,
                                                                     HipdnnMiopenContext>
{
public:
    MiopenReluPlanBuilder() = default;
    ~MiopenReluPlanBuilder() override = default;

    // Disallow copy and assignment
    MiopenReluPlanBuilder(const MiopenReluPlanBuilder&) = delete;
    MiopenReluPlanBuilder& operator=(const MiopenReluPlanBuilder&) = delete;

    bool isApplicable(
        const HipdnnMiopenHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const HipdnnMiopenHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const HipdnnMiopenSettings& executionSettings) const override;

    void initializeExecutionSettings(
        const HipdnnMiopenHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        HipdnnMiopenSettings& executionSettings) const override;

    void buildPlan(const HipdnnMiopenHandle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   HipdnnMiopenContext& executionContext) const override;

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> getCustomKnobs(
        const HipdnnMiopenHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
};

} // namespace miopen_plugin
