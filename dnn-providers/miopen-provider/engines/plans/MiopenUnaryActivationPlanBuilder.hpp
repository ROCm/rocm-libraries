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

// Shared PlanBuilder for all unary pointwise activations (ReLU family, Sigmoid, Tanh, ...).
// None of buildPlan, getMaxWorkspaceSize, initializeExecutionSettings, or getCustomKnobs
// differ per activation, so a single builder handles them all: applicability dispatches on
// the node's pointwise mode in unary_activation_applicability::isSupported, and the plan
// itself is differentiated by MiopenActivationDescriptor.
class MiopenUnaryActivationPlanBuilder
    : public hipdnn_plugin_sdk::
          IPlanBuilder<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
{
public:
    MiopenUnaryActivationPlanBuilder() = default;
    ~MiopenUnaryActivationPlanBuilder() override = default;

    MiopenUnaryActivationPlanBuilder(const MiopenUnaryActivationPlanBuilder&) = delete;
    MiopenUnaryActivationPlanBuilder& operator=(const MiopenUnaryActivationPlanBuilder&) = delete;

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
