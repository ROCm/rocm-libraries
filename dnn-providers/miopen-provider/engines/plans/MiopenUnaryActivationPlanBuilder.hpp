// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "HipdnnMiopenContext.hpp"
#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"
#include "engines/plans/MiopenUnaryActivationPlan.hpp"

namespace miopen_plugin
{

// Shared PlanBuilder for all unary pointwise activations. None of buildPlan,
// getMaxWorkspaceSize, initializeExecutionSettings, or getCustomKnobs differ per activation;
// the only thing that varies is which applicability check to run and the op's display name
// used in log messages, both of which are supplied by the concrete subclass.
//
// IsSupportedFn is a free function with signature
//   bool(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&)
// e.g. relu_applicability::isReluSupported.
template <bool (*IsSupportedFn)(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&)>
class MiopenUnaryActivationPlanBuilder
    : public hipdnn_plugin_sdk::
          IPlanBuilder<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>
{
public:
    explicit MiopenUnaryActivationPlanBuilder(std::string opName)
        : _opName(std::move(opName))
    {
    }
    ~MiopenUnaryActivationPlanBuilder() override = default;

    MiopenUnaryActivationPlanBuilder(const MiopenUnaryActivationPlanBuilder&) = delete;
    MiopenUnaryActivationPlanBuilder& operator=(const MiopenUnaryActivationPlanBuilder&) = delete;

    bool isApplicable(
        [[maybe_unused]] const HipdnnMiopenHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override
    {
        return IsSupportedFn(opGraph);
    }

    size_t getMaxWorkspaceSize(
        [[maybe_unused]] const HipdnnMiopenHandle& handle,
        [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        [[maybe_unused]] const HipdnnMiopenSettings& executionSettings) const override
    {
        // Unary activations do not require workspace memory.
        return 0u;
    }

    void initializeExecutionSettings(
        [[maybe_unused]] const HipdnnMiopenHandle& handle,
        [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
            engineConfig,
        [[maybe_unused]] HipdnnMiopenSettings& executionSettings) const override
    {
        // No execution settings are needed for unary activations.
    }

    void buildPlan(
        [[maybe_unused]] const HipdnnMiopenHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
            engineConfig,
        HipdnnMiopenContext& executionContext) const override
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
        getCustomKnobs([[maybe_unused]] const HipdnnMiopenHandle& handle,
                       [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&
                           opGraph) const override
    {
        // Unary activations do not expose any custom knobs.
        return {};
    }

private:
    std::string _opName;
};

} // namespace miopen_plugin
