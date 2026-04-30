/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include <string>
#include <tuple>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "MiopenConvFwdBiasActivPlanBuilder.hpp"
#include "Workarounds.hpp"
#include "engines/plans/MiopenConvFwdBiasActivPlan.hpp"

namespace miopen_plugin
{

MiopenConvFwdBiasActivPlanBuilder::MiopenConvFwdBiasActivPlanBuilder(bool deterministic)
    : _deterministic(deterministic)
{
}

namespace
{

bool isNodeBias(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attr)
{
    return attr.operation() == hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD;
}

bool isNodeActivFwd(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attr)
{
    using PointwiseMode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    switch(attr.operation())
    {
    case PointwiseMode::ABS:
    case PointwiseMode::ELU_FWD:
    case PointwiseMode::GELU_APPROX_TANH_FWD:
    case PointwiseMode::GELU_FWD:
    case PointwiseMode::RELU_FWD:
    case PointwiseMode::SIGMOID_FWD:
    case PointwiseMode::SOFTPLUS_FWD:
    case PointwiseMode::SWISH_FWD:
    case PointwiseMode::TANH_FWD:
        return true;
    default:
        return false;
    }
}

std::tuple<const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes&,
           const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes*,
           const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes&>
    getNodeAttrs(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    if(opGraph.nodeCount() < 2 || opGraph.nodeCount() > 3)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ConvFwdBiasActiv plan builder supports only graphs with 2 or 3 nodes. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    // Expect that the graph is sorted in topological order
    // Expect the first node to be convolution forward operation
    const auto& convNodeWrapper = opGraph.getNodeWrapper(0);
    const auto convNodeName = convNodeWrapper.name();
    if(convNodeWrapper.attributesType()
       != hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "First node in the graph (" + convNodeName
                + ") must be convolution forward. Found node of type: "
                + std::string(hipdnn_flatbuffers_sdk::data_objects::toString(
                    convNodeWrapper.attributesType())));
    }
    const auto& convAttr
        = convNodeWrapper
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>();

    // Expect the second node to be either bias or activation forward
    const auto& secondNodeWrapper = opGraph.getNodeWrapper(1);
    const auto secondNodeName = secondNodeWrapper.name();
    if(secondNodeWrapper.attributesType()
       != hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Second node in the graph (" + secondNodeName
                + ") must be pointwise operation. Found node of type: "
                + std::string(hipdnn_flatbuffers_sdk::data_objects::toString(
                    secondNodeWrapper.attributesType())));
    }
    const auto& secondNodeAttr
        = opGraph.getNodeWrapper(1)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

    if(isNodeActivFwd(secondNodeAttr))
    {
        // The second node is activation forward
        // If activation node is already present, then graph must have only 2 nodes
        if(opGraph.nodeCount() != 2)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Activation forward cannot be followed by another node. Found "
                    + std::to_string(opGraph.nodeCount()) + " nodes in the graph.");
        }

        const auto& activAttr = secondNodeAttr;
        return {convAttr, nullptr, activAttr};
    }

    if(!isNodeBias(secondNodeAttr))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Second node in the graph (" + secondNodeName
                + ") must be either bias addition or activation forward. Found pointwise "
                  "operation: "
                + std::string(
                    hipdnn_flatbuffers_sdk::data_objects::toString(secondNodeAttr.operation())));
    }

    // The second node is bias
    const auto& biasAttr = secondNodeAttr;

    // Expect the third node to be activation forward
    // If bias node is present, the graph must have 3 nodes
    if(opGraph.nodeCount() != 3)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Graph must have 3 nodes when bias is present.");
    }

    const auto& thirdNodeWrapper = opGraph.getNodeWrapper(2);
    const auto thirdNodeName = thirdNodeWrapper.name();
    if(thirdNodeWrapper.attributesType()
       != hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Third node in the graph (" + thirdNodeName
                + ") must be pointwise operation. Found node of type: "
                + std::string(hipdnn_flatbuffers_sdk::data_objects::toString(
                    thirdNodeWrapper.attributesType())));
    }
    const auto& thirdNodeAttr
        = opGraph.getNodeWrapper(2)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

    if(!isNodeActivFwd(thirdNodeAttr))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Third node in the graph (" + thirdNodeName
                + ") must be activation forward. Found pointwise operation: "
                + std::string(
                    hipdnn_flatbuffers_sdk::data_objects::toString(thirdNodeAttr.operation())));
    }

    const auto& activAttr = thirdNodeAttr;
    return {convAttr, &biasAttr, activAttr};
}

auto getNodeAttrsLogErrors(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
    -> std::optional<decltype(getNodeAttrs(opGraph))>
{
    try
    {
        return getNodeAttrs(opGraph);
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return {};
    }
}

void nodeAttrsCheckTensors(
    const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes& convAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* biasAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;

    // Check the connections between the convolution and bias/activation nodes
    if(biasAttr != nullptr)
    {
        if(!biasAttr->in_1_tensor_uid().has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "Bias node must have two input tensors");
        }

        // One of the bias inputs must be the convolution output
        if(biasAttr->in_0_tensor_uid() != convAttr.y_tensor_uid()
           && biasAttr->in_1_tensor_uid().value() != convAttr.y_tensor_uid())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "One of the bias node inputs must be the convolution output tensor");
        }

        // The activation input must be the bias output
        if(activAttr.in_0_tensor_uid() != biasAttr->out_0_tensor_uid())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Activation node input must be the bias node output tensor");
        }
    }
    else
    {
        // The activation input must be the convolution output
        if(activAttr.in_0_tensor_uid() != convAttr.y_tensor_uid())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Activation node input must be the convolution node output tensor");
        }
    }

    // Verify that all tensors are either virtual or non-virtual as expected
    // Convolution: input and weight tensors must be non-virtual, output tensor must be virtual
    const auto& convTensorAttrX
        = miopen_utils::findTensorAttributes(tensorMap, convAttr.x_tensor_uid());
    const auto& convTensorAttrW
        = miopen_utils::findTensorAttributes(tensorMap, convAttr.w_tensor_uid());
    const auto& convTensorAttrY
        = miopen_utils::findTensorAttributes(tensorMap, convAttr.y_tensor_uid());

    if(convTensorAttrX.virtual_() || convTensorAttrW.virtual_() || !convTensorAttrY.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Convolution input and weight tensors must be "
            "non-virtual, output tensor must be virtual");
    }

    if(biasAttr != nullptr)
    {
        auto biasIn1Uid = biasAttr->in_1_tensor_uid();
        if(!biasIn1Uid.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Bias node must have a second input tensor (in_1)");
        }

        // Bias: input tensor from convolution must be virtual, other input must be non-virtual, output must be virtual
        const auto& biasIn0Attr
            = miopen_utils::findTensorAttributes(tensorMap, biasAttr->in_0_tensor_uid());
        const auto& biasIn1Attr = miopen_utils::findTensorAttributes(tensorMap, *biasIn1Uid);
        const auto& biasOutAttr
            = miopen_utils::findTensorAttributes(tensorMap, biasAttr->out_0_tensor_uid());

        const auto& biasConvOutputAttr
            = (biasAttr->in_0_tensor_uid() == convAttr.y_tensor_uid()) ? biasIn0Attr : biasIn1Attr;
        const auto& biasOtherInputAttr
            = (biasAttr->in_0_tensor_uid() == convAttr.y_tensor_uid()) ? biasIn1Attr : biasIn0Attr;

        if(!biasConvOutputAttr.virtual_() || biasOtherInputAttr.virtual_()
           || !biasOutAttr.virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Bias node input from convolution must be virtual, other input must be "
                "non-virtual, output must be virtual");
        }

        // Activation: input from bias must be virtual, output must be non-virtual
        const auto& activInAttr
            = miopen_utils::findTensorAttributes(tensorMap, activAttr.in_0_tensor_uid());
        const auto& activOutAttr
            = miopen_utils::findTensorAttributes(tensorMap, activAttr.out_0_tensor_uid());

        if(!activInAttr.virtual_() || activOutAttr.virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Activation node input must be virtual, output must be non-virtual");
        }

        // Intermediate data-types
        auto biasDataType = biasOtherInputAttr.data_type();
        if(convTensorAttrY.data_type() != DataType::FLOAT
           && convTensorAttrY.data_type() != biasDataType)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Convolution node output must be float or the same type as bias");
        }

        if(biasOutAttr.data_type() != DataType::FLOAT && biasOutAttr.data_type() != biasDataType)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Bias node output must be float or the same type as bias");
        }
    }
    else
    {
        // Activation: input from convolution must be virtual, output must be non-virtual
        const auto& activInAttr
            = miopen_utils::findTensorAttributes(tensorMap, activAttr.in_0_tensor_uid());
        const auto& activOutAttr
            = miopen_utils::findTensorAttributes(tensorMap, activAttr.out_0_tensor_uid());

        if(!activInAttr.virtual_() || activOutAttr.virtual_())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Activation node input must be virtual, output must be non-virtual");
        }

        // Intermediate data type
        if(activInAttr.data_type() != DataType::FLOAT)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "Convolution node output must be float");
        }
    }
}

bool nodeAttrsCheckTensorsLogErrors(
    const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes& convAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* biasAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    try
    {
        nodeAttrsCheckTensors(convAttr, biasAttr, activAttr, tensorMap);
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }
}

void checkComputeTypes(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
    const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes& convAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* biasAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    uint32_t convAttrIdx = 0;
    uint32_t biasAttrIdx = 1;
    uint32_t activAttrIdx = (biasAttr != nullptr) ? 2 : 1;

    if(graph.getNode(convAttrIdx).compute_data_type()
       != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Convolution node compute data type must be float");
    }

    if(biasAttr != nullptr)
    {
        auto biasIn1Uid = biasAttr->in_1_tensor_uid();
        if(!biasIn1Uid.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Bias node must have a second input tensor (in_1)");
        }
        int64_t biasIdx = convAttr.y_tensor_uid() != biasAttr->in_0_tensor_uid()
                              ? biasAttr->in_0_tensor_uid()
                              : *biasIn1Uid;

        if(tensorMap.at(biasIdx)->data_type() != graph.getNode(biasAttrIdx).compute_data_type())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Bias node compute data type must be the same as the bias tensor type");
        }
    }

    if(graph.getNode(activAttrIdx).compute_data_type()
       != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Activation node compute data type must be float");
    }
}

bool checkComputeTypesLogErrors(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
    const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes& convAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* biasAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    try
    {
        checkComputeTypes(graph, convAttr, biasAttr, tensorMap);
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }
}

} // namespace

bool MiopenConvFwdBiasActivPlanBuilder::isApplicable(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    REJECT_IF_WORKAROUND_ISSUE_5409(handle);

    auto nodeAttrs = getNodeAttrsLogErrors(opGraph);
    if(!nodeAttrs.has_value())
    {
        return false;
    }

    if(!nodeAttrsCheckTensorsLogErrors(std::get<0>(nodeAttrs.value()),
                                       std::get<1>(nodeAttrs.value()),
                                       std::get<2>(nodeAttrs.value()),
                                       opGraph.getTensorMap()))
    {
        return false;
    }

    if(!checkComputeTypesLogErrors(opGraph,
                                   std::get<0>(nodeAttrs.value()),
                                   std::get<1>(nodeAttrs.value()),
                                   opGraph.getTensorMap()))
    {
        return false;
    }

    try
    {
        ConvFwdBiasActivParams params(std::get<0>(nodeAttrs.value()),
                                      std::get<1>(nodeAttrs.value()),
                                      std::get<2>(nodeAttrs.value()),
                                      opGraph.getTensorMap(),
                                      _deterministic);
        HipdnnMiopenSettings executionSettings;
        ConvFwdBiasActivPlan plan(handle, std::move(params), executionSettings, true, false);
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }
}

size_t MiopenConvFwdBiasActivPlanBuilder::getMaxWorkspaceSize(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const HipdnnMiopenSettings& executionSettings) const
{
    const auto [convAttr, biasAttr, activAttr] = getNodeAttrs(opGraph);
    nodeAttrsCheckTensors(convAttr, biasAttr, activAttr, opGraph.getTensorMap());

    ConvFwdBiasActivParams params(
        convAttr, biasAttr, activAttr, opGraph.getTensorMap(), _deterministic);
    ConvFwdBiasActivPlan plan(handle, std::move(params), executionSettings, false, true);
    return plan.getWorkspaceSize(handle);
}

void MiopenConvFwdBiasActivPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const HipdnnMiopenHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] HipdnnMiopenSettings& executionSettings) const
{
}

void MiopenConvFwdBiasActivPlanBuilder::buildPlan(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    HipdnnMiopenContext& executionContext) const
{
    const auto [convAttr, biasAttr, activAttr] = getNodeAttrs(opGraph);
    nodeAttrsCheckTensors(convAttr, biasAttr, activAttr, opGraph.getTensorMap());

    ConvFwdBiasActivParams params(
        convAttr, biasAttr, activAttr, opGraph.getTensorMap(), _deterministic);
    auto plan = std::make_unique<ConvFwdBiasActivPlan>(
        handle, std::move(params), executionContext.executionSettings(), true, true);
    executionContext.setPlan(std::move(plan));
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
    MiopenConvFwdBiasActivPlanBuilder::getCustomKnobs(
        [[maybe_unused]] const HipdnnMiopenHandle& handle,
        [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace miopen_plugin
