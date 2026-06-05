// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <limits>
#include <numeric>
#include <string>

#include <hipblaslt/hipblaslt.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipblasltMatmulPlan.hpp"
#include "HipblasltMatmulPlanBuilder.hpp"

namespace hipblaslt_plugin
{
namespace
{

bool isBias(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attr)
{
    return attr.operation() == hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD;
}

bool isSupportedActivation(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attr)
{
    using PointwiseMode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    switch(attr.operation())
    {
    case PointwiseMode::RELU_FWD:
    case PointwiseMode::GELU_APPROX_TANH_FWD:
    case PointwiseMode::SWISH_FWD:
        return true;
    default:
        return false;
    }
}

std::tuple<const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes&,
           const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes*,
           const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes*>
    getNodeAttrs(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    if(opGraph.nodeCount() < 1 || opGraph.nodeCount() > 3)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Matmul plan builder supports only graphs with 1, 2 or 3 nodes. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    // Expect that the graph is sorted in topological order
    // Expect the first node to be matmul operation
    const auto& matmulNodeWrapper = opGraph.getNodeWrapper(0);
    const auto matmulNodeName = matmulNodeWrapper.name();
    if(matmulNodeWrapper.attributesType()
       != hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MatmulAttributes)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "First node in the graph (" + matmulNodeName + ") must be matmul. Found node of type: "
                + std::string(hipdnn_flatbuffers_sdk::data_objects::toString(
                    matmulNodeWrapper.attributesType())));
    }

    const auto& matmulAttr
        = matmulNodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes>();
    if(opGraph.nodeCount() == 1)
    {
        return {matmulAttr, nullptr, nullptr};
    }

    // Expect the possble second node to be either bias or activation forward
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

    if(isSupportedActivation(secondNodeAttr))
    {
        // The second node is activation.
        // If activation node is already present, then graph must have only 2 nodes
        if(opGraph.nodeCount() != 2)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Activation cannot be followed by another node. Found "
                    + std::to_string(opGraph.nodeCount()) + " nodes in the graph.");
        }

        // The second node is activation
        const auto& activAttr = secondNodeAttr;
        return {matmulAttr, nullptr, &activAttr};
    }

    if(!isBias(secondNodeAttr))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Second node in the graph (" + secondNodeName
                + ") must be either bias addition or supported activation. Found pointwise "
                  "operation: "
                + std::string(
                    hipdnn_flatbuffers_sdk::data_objects::toString(secondNodeAttr.operation())));
    }

    // The second node is bias
    const auto& biasAttr = secondNodeAttr;
    if(opGraph.nodeCount() == 2)
    {
        return {matmulAttr, &biasAttr, nullptr};
    }

    // The third node after bias is activation
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

    if(!isSupportedActivation(thirdNodeAttr))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Third node in the graph (" + thirdNodeName
                + ") must be supported activation. Found pointwise operation: "
                + std::string(
                    hipdnn_flatbuffers_sdk::data_objects::toString(thirdNodeAttr.operation())));
    }

    const auto& activAttr = thirdNodeAttr;
    return {matmulAttr, &biasAttr, &activAttr};
}

void checkNodeAttrsTensors(
    const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes& matmulAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* biasAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* activAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto& aType
        = hipblaslt_utils::findTensorAttributes(tensorMap, matmulAttr.a_tensor_uid()).dataType();
    const auto& bType
        = hipblaslt_utils::findTensorAttributes(tensorMap, matmulAttr.b_tensor_uid()).dataType();
    const auto& cType
        = hipblaslt_utils::findTensorAttributes(tensorMap, matmulAttr.c_tensor_uid()).dataType();

    static constexpr std::array<hipdnn_flatbuffers_sdk::data_objects::DataType, 3> VALID_DATA_TYPES
        = {hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
           hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
           hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16};

    if(std::find(VALID_DATA_TYPES.begin(), VALID_DATA_TYPES.end(), aType) == VALID_DATA_TYPES.end()
       || std::find(VALID_DATA_TYPES.begin(), VALID_DATA_TYPES.end(), bType)
              == VALID_DATA_TYPES.end()
       || std::find(VALID_DATA_TYPES.begin(), VALID_DATA_TYPES.end(), cType)
              == VALID_DATA_TYPES.end())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Matmul node input and output data types must be fp32, fp16 or bf16");
    }

    // Check the connections between the matmul and bias/activation nodes
    if(biasAttr != nullptr)
    {
        if(!biasAttr->in_1_tensor_uid().has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "Bias node must have two input tensors");
        }

        // One of the bias inputs must be the matmul output
        if(biasAttr->in_0_tensor_uid() != matmulAttr.c_tensor_uid()
           && biasAttr->in_1_tensor_uid().value() != matmulAttr.c_tensor_uid())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "One of the bias node inputs must be the matmul output tensor");
        }

        // The existing activation input must be the bias output
        if(activAttr != nullptr && activAttr->in_0_tensor_uid() != biasAttr->out_0_tensor_uid())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Activation node input must be the bias node output tensor");
        }

        const auto& biasInType
            = biasAttr->in_0_tensor_uid() != matmulAttr.c_tensor_uid()
                  ? hipblaslt_utils::findTensorAttributes(tensorMap,
                                                          biasAttr->in_1_tensor_uid().value())
                        .dataType()
                  : hipblaslt_utils::findTensorAttributes(tensorMap, biasAttr->in_0_tensor_uid())
                        .dataType();
        const auto& biasOutType
            = hipblaslt_utils::findTensorAttributes(tensorMap, biasAttr->out_0_tensor_uid())
                  .dataType();

        if(std::find(VALID_DATA_TYPES.begin(), VALID_DATA_TYPES.end(), biasInType)
               == VALID_DATA_TYPES.end()
           || std::find(VALID_DATA_TYPES.begin(), VALID_DATA_TYPES.end(), biasOutType)
                  == VALID_DATA_TYPES.end())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Bias node input and output data types must be fp32, fp16 or bf16");
        }
    }
    else if(activAttr != nullptr)
    {
        // The activation input must be the matmul output
        if(activAttr->in_0_tensor_uid() != matmulAttr.c_tensor_uid())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Activation node input must be the matmul node output tensor");
        }
    }
}

void checkComputeTypes(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                       const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* biasAttr,
                       const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* activAttr)
{
    uint32_t const matmulAttrIdx = 0;
    uint32_t const biasAttrIdx = 1;
    uint32_t const activAttrIdx = (biasAttr != nullptr) ? 2 : 1;

    if(graph.getNode(matmulAttrIdx).compute_data_type()
       != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Matmul node compute data type must be float");
    }

    if(biasAttr != nullptr)
    {
        if(graph.getNode(biasAttrIdx).compute_data_type()
           != graph.getNode(matmulAttrIdx).compute_data_type())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Bias node compute data type must be equal to matmul node compute data type");
        }
    }

    if(activAttr != nullptr)
    {
        if(graph.getNode(activAttrIdx).compute_data_type()
           != graph.getNode(matmulAttrIdx).compute_data_type())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Activation node compute data type must be equal to matmul node compute data type");
        }
    }
}

} // namespace

bool HipblasltMatmulPlanBuilder::isApplicable(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    try
    {
        auto nodeAttrs = getNodeAttrs(opGraph);
        checkNodeAttrsTensors(std::get<0>(nodeAttrs),
                              std::get<1>(nodeAttrs),
                              std::get<2>(nodeAttrs),
                              opGraph.getTensorMap());

        checkComputeTypes(opGraph, std::get<1>(nodeAttrs), std::get<2>(nodeAttrs));

        MatmulParams params(std::get<0>(nodeAttrs),
                            std::get<1>(nodeAttrs),
                            std::get<2>(nodeAttrs),
                            opGraph.getTensorMap());
        MatmulPlan const plan(handle, std::move(params));
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }
}

size_t HipblasltMatmulPlanBuilder::getWorkspaceSize(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    auto nodeAttrs = getNodeAttrs(opGraph);
    checkNodeAttrsTensors(std::get<0>(nodeAttrs),
                          std::get<1>(nodeAttrs),
                          std::get<2>(nodeAttrs),
                          opGraph.getTensorMap());

    MatmulParams params(std::get<0>(nodeAttrs),
                        std::get<1>(nodeAttrs),
                        std::get<2>(nodeAttrs),
                        opGraph.getTensorMap());
    MatmulPlan const plan(handle, std::move(params));
    return plan.getWorkspaceSize(handle);
}

void HipblasltMatmulPlanBuilder::buildPlan(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    auto nodeAttrs = getNodeAttrs(opGraph);
    checkNodeAttrsTensors(std::get<0>(nodeAttrs),
                          std::get<1>(nodeAttrs),
                          std::get<2>(nodeAttrs),
                          opGraph.getTensorMap());

    HIPDNN_PLUGIN_LOG_INFO("Building matmul plan for node: " << opGraph.getNodeWrapper(0).name());

    MatmulParams params(std::get<0>(nodeAttrs),
                        std::get<1>(nodeAttrs),
                        std::get<2>(nodeAttrs),
                        opGraph.getTensorMap());
    auto plan = std::make_unique<MatmulPlan>(handle, std::move(params));
    executionContext.setPlan(std::move(plan));
}

} // namespace hipblaslt_plugin
