// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ConvFwdPlanBuilder.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "ConvFwdParams.hpp"
#include "ConvFwdPlan.hpp"

namespace example_provider
{

static constexpr int64_t DEFAULT_BLOCK_SIZE = 256;

ConvFwdPlanBuilder::ConvFwdPlanBuilder(const IKernelCompiler& compiler)
    : _compiler(compiler)
{
}

bool ConvFwdPlanBuilder::isApplicable(
    const ExampleProviderHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    using NodeAttributes = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
    using ConvMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode;
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;

    // Must contain only convolution forward attributes
    if(!opGraph.hasOnlySupportedAttributes({NodeAttributes::ConvolutionFwdAttributes}))
    {
        return false;
    }

    // Must be a single-node graph
    if(opGraph.nodeCount() != 1)
    {
        return false;
    }

    // The single node must be a ConvolutionFwd with CROSS_CORRELATION mode
    const auto& node = opGraph.getNode(0);
    const auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    if(attrs == nullptr)
    {
        return false;
    }

    if(attrs->conv_mode() != ConvMode::CROSS_CORRELATION)
    {
        return false;
    }

    // Only unit dilation is supported
    if(attrs->dilation() != nullptr)
    {
        for(flatbuffers::uoffset_t i = 0; i < attrs->dilation()->size(); ++i)
        {
            if(attrs->dilation()->Get(i) != 1)
            {
                HIPDNN_PLUGIN_LOG_INFO("ConvFwdPlanBuilder: Non-unit dilation not supported");
                return false;
            }
        }
    }

    // Only FLOAT data type is supported
    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    return nodeWrapper.computeDataType() == DataType::FLOAT;
}

size_t ConvFwdPlanBuilder::getMaxWorkspaceSize(
    const ExampleProviderHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const ExampleProviderSettings& /*executionSettings*/) const
{
    // The convolution kernel in this engine does not require a workspace.
    return 0;
}

void ConvFwdPlanBuilder::initializeExecutionSettings(
    const ExampleProviderHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    ExampleProviderSettings& executionSettings) const
{
    // Read block size knob from engine config if present
    if(engineConfig.hasKnobSetting("BLOCK_SIZE"))
    {
        const auto& knobSetting = engineConfig.getKnobSettingByName("BLOCK_SIZE");
        const auto& intVal = knobSetting.valueAs<hipdnn_flatbuffers_sdk::data_objects::IntValue>();
        executionSettings.blockSize = intVal.value();
    }
}

void ConvFwdPlanBuilder::buildPlan(
    const ExampleProviderHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    ExampleProviderContext& executionContext) const
{
    const auto& node = opGraph.getNode(0);
    const auto* attrs = node.attributes_as_ConvolutionFwdAttributes();
    if(attrs == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "Expected ConvolutionFwdAttributes on node 0 but found null");
    }

    auto inputUid = attrs->x_tensor_uid();
    auto weightUid = attrs->w_tensor_uid();
    auto outputUid = attrs->y_tensor_uid();

    // Get block size from execution settings
    const auto& settings = executionContext.executionSettings();
    const int64_t blockSize = settings.blockSize;

    // Extract tensor dimensions from the graph tensor map
    const auto& tensorMap = opGraph.getTensorMap();

    // Input tensor (N, C, H, W)
    auto inputIt = tensorMap.find(inputUid);
    if(inputIt == tensorMap.end())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "Input tensor not found in graph tensor map");
    }
    const auto* inputDims = inputIt->second->dims();
    if(inputDims == nullptr || inputDims->size() < 4)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Input tensor must have >= 4 dimensions");
    }

    // Weight tensor (K, C, R, S)
    auto weightIt = tensorMap.find(weightUid);
    if(weightIt == tensorMap.end())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "Weight tensor not found in graph tensor map");
    }
    const auto* weightDims = weightIt->second->dims();
    if(weightDims == nullptr || weightDims->size() < 4)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Weight tensor must have >= 4 dimensions");
    }

    // Output tensor (N, K, outH, outW)
    auto outputIt = tensorMap.find(outputUid);
    if(outputIt == tensorMap.end())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "Output tensor not found in graph tensor map");
    }
    const auto* outputDims = outputIt->second->dims();
    if(outputDims == nullptr || outputDims->size() < 4)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Output tensor must have >= 4 dimensions");
    }

    // NCHW layout: dims[0]=N, dims[1]=C, dims[2]=H, dims[3]=W
    auto n = inputDims->Get(0);
    auto c = inputDims->Get(1);
    auto h = inputDims->Get(2);
    auto w = inputDims->Get(3);
    auto k = weightDims->Get(0);
    auto r = weightDims->Get(2);
    auto s = weightDims->Get(3);

    // Padding and stride from convolution attributes
    int64_t padH = 0;
    int64_t padW = 0;
    int64_t strideH = 1;
    int64_t strideW = 1;

    if(attrs->pre_padding() != nullptr && attrs->pre_padding()->size() >= 2)
    {
        padH = attrs->pre_padding()->Get(0);
        padW = attrs->pre_padding()->Get(1);
    }

    if(attrs->stride() != nullptr && attrs->stride()->size() >= 2)
    {
        strideH = attrs->stride()->Get(0);
        strideW = attrs->stride()->Get(1);
    }

    auto outH = outputDims->Get(2);
    auto outW = outputDims->Get(3);

    const ConvFwdParams params{inputUid,
                               weightUid,
                               outputUid,
                               n,
                               c,
                               h,
                               w,
                               k,
                               r,
                               s,
                               outH,
                               outW,
                               padH,
                               padW,
                               strideH,
                               strideW,
                               blockSize};
    auto plan = std::make_unique<ConvFwdPlan>(params);
    plan->compile(_compiler);

    executionContext.setPlan(std::move(plan));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> ConvFwdPlanBuilder::getCustomKnobs(
    const ExampleProviderHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const
{
    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> knobs;

    hipdnn_flatbuffers_sdk::data_objects::KnobT knob;
    knob.knob_id = "BLOCK_SIZE";
    knob.description = "Thread block size for convolution kernel";

    // Default value: 256
    hipdnn_flatbuffers_sdk::data_objects::IntValueT defaultValue;
    defaultValue.value = DEFAULT_BLOCK_SIZE;
    knob.default_value.Set(defaultValue);

    // Constraint: valid values are 64, 128, 256
    hipdnn_flatbuffers_sdk::data_objects::IntConstraintT constraint;
    constraint.min_value = 64;
    constraint.max_value = 256;
    constraint.valid_values = {64, 128, 256};
    knob.constraint.Set(std::move(constraint));

    knobs.push_back(std::move(knob));
    return knobs;
}

} // namespace example_provider
