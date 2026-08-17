// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ReluPlanBuilder.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "ReluParams.hpp"
#include "ReluPlan.hpp"

namespace example_provider
{

ReluPlanBuilder::ReluPlanBuilder(const IKernelCompiler& compiler)
    : _compiler(compiler)
{
}

bool ReluPlanBuilder::isApplicable(
    const ExampleProviderHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    using NodeAttributes = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
    using PointwiseMode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;

    // Must contain only pointwise attributes
    if(!opGraph.hasOnlySupportedAttributes({NodeAttributes::PointwiseAttributes}))
    {
        return false;
    }

    // Must be a single-node graph
    if(opGraph.nodeCount() != 1)
    {
        return false;
    }

    // The single node must be RELU_FWD
    const auto& node = opGraph.getNode(0);
    const auto* attrs = node.attributes_as_PointwiseAttributes();
    if(attrs == nullptr)
    {
        return false;
    }

    if(attrs->operation() != PointwiseMode::RELU_FWD)
    {
        return false;
    }

    // Only FLOAT data type is supported; HALF/BFLOAT16 are not yet implemented
    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    return nodeWrapper.computeDataType() == DataType::FLOAT;
}

size_t ReluPlanBuilder::getMaxWorkspaceSize(
    const ExampleProviderHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const ExampleProviderSettings& /*executionSettings*/) const
{
    // ReLU does not require workspace
    return 0;
}

void ReluPlanBuilder::initializeExecutionSettings(
    const ExampleProviderHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    ExampleProviderSettings& executionSettings) const
{
    // Read negative slope knob from engine config if present
    if(engineConfig.hasKnobSetting("example.relu.negative_slope"))
    {
        const auto& knobSetting = engineConfig.getKnobSettingByName("example.relu.negative_slope");
        const auto& floatVal
            = knobSetting.valueAs<hipdnn_flatbuffers_sdk::data_objects::FloatValue>();
        executionSettings.reluNegativeSlope = floatVal.value();
    }
}

void ReluPlanBuilder::buildPlan(
    const ExampleProviderHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    ExampleProviderContext& executionContext) const
{
    // Extract tensor UIDs from the pointwise attributes
    const auto& node = opGraph.getNode(0);
    const auto* attrs = node.attributes_as_PointwiseAttributes();
    if(attrs == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "Expected PointwiseAttributes on node 0 but found null");
    }

    auto inputUid = attrs->in_0_tensor_uid();
    auto outputUid = attrs->out_0_tensor_uid();

    // Extract tensor dimensions from the graph's TensorAttributes
    const auto& tensorMap = opGraph.getTensorMap();

    auto inputIt = tensorMap.find(inputUid);
    if(inputIt == tensorMap.end())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "Input tensor with UID " + std::to_string(inputUid) + " not found in graph tensor map");
    }

    const auto* inputDims = inputIt->second->dims();
    if(inputDims == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Input tensor has null dimensions");
    }

    // Compute total element count from dimensions
    int64_t numElements = 1;
    for(flatbuffers::uoffset_t i = 0; i < inputDims->size(); ++i)
    {
        numElements *= inputDims->Get(i);
    }

    // Get negative slope from execution settings
    const auto& settings = executionContext.executionSettings();

    const ReluParams params{inputUid, outputUid, numElements, settings.reluNegativeSlope};
    auto plan = std::make_unique<ReluPlan>(params);
    plan->compile(_compiler);

    executionContext.setPlan(std::move(plan));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> ReluPlanBuilder::getCustomKnobs(
    const ExampleProviderHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const
{
    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> knobs;

    hipdnn_flatbuffers_sdk::data_objects::KnobT knob;
    knob.knob_id = "example.relu.negative_slope";
    knob.description = "Negative slope for leaky ReLU (0.0 = standard ReLU)";

    // Default value: 0.0 (standard ReLU)
    hipdnn_flatbuffers_sdk::data_objects::FloatValueT defaultValue;
    defaultValue.value = 0.0;
    knob.default_value.Set(defaultValue);

    // Constraint: 0.0 to 1.0
    hipdnn_flatbuffers_sdk::data_objects::FloatConstraintT constraint;
    constraint.min_value = 0.0;
    constraint.max_value = 1.0;
    knob.constraint.Set(constraint);

    knobs.push_back(std::move(knob));
    return knobs;
}

} // namespace example_provider
