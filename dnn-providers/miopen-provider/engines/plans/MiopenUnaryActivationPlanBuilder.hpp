// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <functional>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "HipdnnMiopenContext.hpp"
#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"

namespace miopen_plugin
{

// Shared PlanBuilder for all unary pointwise activations. None of buildPlan,
// getMaxWorkspaceSize, initializeExecutionSettings, or getCustomKnobs differ per activation;
// the only thing that varies is which applicability check to run and the op's display name
// used in log messages, both supplied at construction time (e.g.
// MiopenUnaryActivationPlanBuilder("Relu", relu_applicability::isReluSupported)).
class MiopenUnaryActivationPlanBuilder
    : public hipdnn_plugin_sdk::
          IPlanBuilder<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
{
public:
    using IsSupportedFn
        = std::function<bool(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&)>;

    MiopenUnaryActivationPlanBuilder(std::string opName, IsSupportedFn isSupportedFn)
        : _opName(std::move(opName))
        , _isSupportedFn(std::move(isSupportedFn))
    {
    }

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

private:
    std::string _opName;
    IsSupportedFn _isSupportedFn;
};

} // namespace miopen_plugin
