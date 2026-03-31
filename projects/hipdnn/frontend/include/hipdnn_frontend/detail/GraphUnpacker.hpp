// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributesVarianceExt.hpp>
#include <hipdnn_frontend/attributes/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/BlockScaleQuantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <hipdnn_frontend/detail/OperationUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/BatchnormBackwardNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNodeVarianceExt.hpp>
#include <hipdnn_frontend/node/BatchnormNode.hpp>
#include <hipdnn_frontend/node/BlockScaleDequantizeNode.hpp>
#include <hipdnn_frontend/node/BlockScaleQuantizeNode.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_frontend/node/CustomOpNode.hpp>
#include <hipdnn_frontend/node/LayerNormNode.hpp>
#include <hipdnn_frontend/node/MatmulNode.hpp>
#include <hipdnn_frontend/node/Node.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>
#include <hipdnn_frontend/node/RMSNormNode.hpp>
#include <hipdnn_frontend/node/ReductionNode.hpp>
#include <hipdnn_frontend/node/SdpaBpropNode.hpp>
#include <hipdnn_frontend/node/SdpaFpropNode.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hipdnn_frontend::detail
{

/// Deserializes a FlatBuffer node's attributes into a frontend node and emplaces it.
template <typename FrontendAttrType, typename FrontendNodeType, typename FbAttrType>
void unpackNodeFromFlatBuffer(
    const hipdnn_data_sdk::data_objects::Node* fbNode,
    const FbAttrType* fbAttr,
    const std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    const graph::GraphAttributes& outGraphAttrs,
    std::vector<std::shared_ptr<graph::INode>>& outNodes)
{
    auto attr = FrontendAttrType::fromFlatBuffer(fbAttr, tensorMap);
    if(fbNode->name() != nullptr)
    {
        attr.set_name(fbNode->name()->str());
    }
    outNodes.emplace_back(std::make_shared<FrontendNodeType>(std::move(attr), outGraphAttrs));
}

/// Builds frontend nodes and graph-level attributes from a parsed FlatBuffer Graph.
/// Each node type is dispatched to its corresponding frontend node class.
[[nodiscard]] inline Error
    unpackGraphFromFlatBuffer(const hipdnn_data_sdk::data_objects::Graph* fbGraph,
                              std::vector<std::shared_ptr<graph::INode>>& outNodes,
                              graph::GraphAttributes& outGraphAttrs,
                              std::optional<int64_t>& outPreferredEngineId)
{
    // Set graph attributes from FlatBuffer
    if(fbGraph->name() != nullptr)
    {
        outGraphAttrs.set_name(fbGraph->name()->c_str());
    }
    outGraphAttrs.set_compute_data_type(fromSdkType(fbGraph->compute_data_type()));
    outGraphAttrs.set_intermediate_data_type(fromSdkType(fbGraph->intermediate_data_type()));
    outGraphAttrs.set_io_data_type(fromSdkType(fbGraph->io_data_type()));

    outPreferredEngineId = fbGraph->preferred_engine_id();

    // Build tensorMap from FlatBuffer tensors
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>> tensorMap;
    if(fbGraph->tensors() != nullptr)
    {
        for(const auto* fbTensor : *fbGraph->tensors())
        {
            auto tensor = graph::TensorAttributes::fromFlatBuffer(fbTensor);
            if(tensor == nullptr)
            {
                return {ErrorCode::INVALID_VALUE, "Failed to deserialize tensor from FlatBuffer"};
            }
            if(!tensor->has_uid())
            {
                return {ErrorCode::INVALID_VALUE, "Tensor in FlatBuffer graph has no UID"};
            }
            tensorMap[tensor->get_uid()] = tensor;
        }
    }

    // Create nodes from FlatBuffer
    if(fbGraph->nodes() != nullptr)
    {
        for(const auto* fbNode : *fbGraph->nodes())
        {
            if(fbNode == nullptr)
            {
                return {ErrorCode::INVALID_VALUE, "Null node in FlatBuffer graph"};
            }

            auto type = fbNode->attributes_type();

            switch(type)
            {
            case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes:
                unpackNodeFromFlatBuffer<graph::BatchnormAttributes, graph::BatchnormNode>(
                    fbNode,
                    fbNode->attributes_as_BatchnormAttributes(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
                unpackNodeFromFlatBuffer<graph::BatchnormBackwardAttributes,
                                         graph::BatchnormBackwardNode>(
                    fbNode,
                    fbNode->attributes_as_BatchnormBackwardAttributes(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
                unpackNodeFromFlatBuffer<graph::BatchnormInferenceAttributes,
                                         graph::BatchnormInferenceNode>(
                    fbNode,
                    fbNode->attributes_as_BatchnormInferenceAttributes(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::
                BatchnormInferenceAttributesVarianceExt:
                unpackNodeFromFlatBuffer<graph::BatchnormInferenceAttributesVarianceExt,
                                         graph::BatchnormInferenceNodeVarianceExt>(
                    fbNode,
                    fbNode->attributes_as_BatchnormInferenceAttributesVarianceExt(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
                unpackNodeFromFlatBuffer<graph::ConvFpropAttributes, graph::ConvolutionFpropNode>(
                    fbNode,
                    fbNode->attributes_as_ConvolutionFwdAttributes(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
                unpackNodeFromFlatBuffer<graph::ConvDgradAttributes, graph::ConvolutionDgradNode>(
                    fbNode,
                    fbNode->attributes_as_ConvolutionBwdAttributes(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
                unpackNodeFromFlatBuffer<graph::ConvWgradAttributes, graph::ConvolutionWgradNode>(
                    fbNode,
                    fbNode->attributes_as_ConvolutionWrwAttributes(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes:
                unpackNodeFromFlatBuffer<graph::PointwiseAttributes, graph::PointwiseNode>(
                    fbNode,
                    fbNode->attributes_as_PointwiseAttributes(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::MatmulAttributes:
                unpackNodeFromFlatBuffer<graph::MatmulAttributes, graph::MatmulNode>(
                    fbNode,
                    fbNode->attributes_as_MatmulAttributes(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::SdpaAttributes:
                unpackNodeFromFlatBuffer<graph::SdpaAttributes, graph::SdpaFpropNode>(
                    fbNode,
                    fbNode->attributes_as_SdpaAttributes(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::LayernormAttributes:
                unpackNodeFromFlatBuffer<graph::LayernormAttributes, graph::LayerNormNode>(
                    fbNode,
                    fbNode->attributes_as_LayernormAttributes(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::RMSNormAttributes:
                unpackNodeFromFlatBuffer<graph::RMSNormAttributes, graph::RMSNormNode>(
                    fbNode,
                    fbNode->attributes_as_RMSNormAttributes(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes:
                unpackNodeFromFlatBuffer<graph::BlockScaleDequantizeAttributes,
                                         graph::BlockScaleDequantizeNode>(
                    fbNode,
                    fbNode->attributes_as_BlockScaleDequantizeAttributes(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::BlockScaleQuantizeAttributes:
                unpackNodeFromFlatBuffer<graph::BlockScaleQuantizeAttributes,
                                         graph::BlockScaleQuantizeNode>(
                    fbNode,
                    fbNode->attributes_as_BlockScaleQuantizeAttributes(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::SdpaBackwardAttributes:
                unpackNodeFromFlatBuffer<graph::SdpaBackwardAttributes, graph::SdpaBpropNode>(
                    fbNode,
                    fbNode->attributes_as_SdpaBackwardAttributes(),
                    tensorMap,
                    outGraphAttrs,
                    outNodes);
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::CustomOpAttributes:
            {
                // CustomOpAttributes does not store compute_data_type in its FlatBuffer table;
                // it must be read from the node-level field.
                auto attr = graph::CustomOpAttributes::fromFlatBuffer(
                    fbNode->attributes_as_CustomOpAttributes(), tensorMap);
                if(fbNode->name() != nullptr)
                {
                    attr.set_name(fbNode->name()->str());
                }
                attr.set_compute_data_type(fromSdkType(fbNode->compute_data_type()));
                outNodes.emplace_back(
                    std::make_shared<graph::CustomOpNode>(std::move(attr), outGraphAttrs));
                break;
            }
            case hipdnn_data_sdk::data_objects::NodeAttributes::ReductionAttributes:
            {
                auto attr = graph::ReductionAttributes::fromFlatBuffer(
                    fbNode->attributes_as_ReductionAttributes(), tensorMap);
                if(fbNode->name() != nullptr)
                {
                    attr.set_name(fbNode->name()->str());
                }
                attr.set_compute_data_type(fromSdkType(fbNode->compute_data_type()));
                outNodes.emplace_back(
                    std::make_shared<graph::ReductionNode>(std::move(attr), outGraphAttrs));
                break;
            }
            default:
                return {ErrorCode::INVALID_VALUE,
                        "Unsupported node type in FlatBuffer deserialization"};
            }
        }
    }

    return {};
}

/// Unpacks a finalized backend OperationGraph descriptor into frontend nodes
/// and graph-level attributes.
///
/// Extracts operations and graph-level data types from a backend descriptor and
/// rebuilds the frontend Graph representation. Tensors are shared across operations
/// via UID-based lookup.
[[nodiscard]] inline Error
    unpackGraphDescriptor(hipdnnBackendDescriptor_t graphDesc,
                          std::vector<std::shared_ptr<graph::INode>>& outNodes,
                          graph::GraphAttributes& outGraphAttrs,
                          std::optional<int64_t>& outPreferredEngineId)
{
    if(graphDesc == nullptr)
    {
        return {ErrorCode::INVALID_VALUE, "Null backend graph descriptor"};
    }

    // Query operation descriptors from the backend graph descriptor
    auto [opDescs, opErr] = getDescriptorAttrDescArray(
        graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_OPS, "operation descriptors from graph");

    if(opErr.is_bad())
    {
        return opErr;
    }

    if(opDescs.empty())
    {
        return {ErrorCode::INVALID_VALUE, "Graph descriptor has no operations"};
    }

    // Tensor map for sharing tensors across operations
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>> tensorMap;

    // Process each operation using the generic unpacker
    for(size_t i = 0; i < opDescs.size(); ++i)
    {
        auto opDesc = opDescs[i].get();
        if(opDesc == nullptr)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Null operation descriptor at index " + std::to_string(i)
                        + " returned from graph descriptor"};
        }

        auto [node, err] = unpackOperation(opDesc, tensorMap, outGraphAttrs);

        if(err.is_bad())
        {
            return {err.code,
                    "Failed to unpack operation " + std::to_string(i) + ": " + err.get_message()};
        }

        outNodes.emplace_back(std::move(node));
    }

    // Query graph-level data types
    auto [computeDataType, computeErr]
        = unpackGraphDataType(graphDesc,
                              HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT,
                              "compute data type from graph descriptor");
    if(computeErr.is_bad())
    {
        return computeErr;
    }
    outGraphAttrs.set_compute_data_type(computeDataType);

    auto [intermediateDataType, intermediateErr]
        = unpackGraphDataType(graphDesc,
                              HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT,
                              "intermediate data type from graph descriptor");
    if(intermediateErr.is_bad())
    {
        return intermediateErr;
    }
    outGraphAttrs.set_intermediate_data_type(intermediateDataType);

    auto [ioDataType, ioErr] = unpackGraphDataType(graphDesc,
                                                   HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT,
                                                   "IO data type from graph descriptor");
    if(ioErr.is_bad())
    {
        return ioErr;
    }
    outGraphAttrs.set_io_data_type(ioDataType);

    // Query preferred engine ID (optional, may not be set)
    int64_t preferredEngineId = 0;
    int64_t actualCount = 0;
    auto status
        = hipdnnBackend()->backendGetAttribute(graphDesc,
                                               HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT,
                                               HIPDNN_TYPE_INT64,
                                               1,
                                               &actualCount,
                                               &preferredEngineId);
    if(status == HIPDNN_STATUS_SUCCESS && actualCount > 0)
    {
        outPreferredEngineId = preferredEngineId;
    }

    // Query graph name (optional, may not be set)
    std::string graphName;
    HIPDNN_CHECK_ERROR(getDescriptorAttrString(
        graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT, graphName, "graph name"));
    if(!graphName.empty())
    {
        outGraphAttrs.set_name(graphName);
    }

    return {};
}

/// Deserializes a backend graph descriptor from binary data and unpacks it into
/// frontend nodes and graph-level attributes. If a handle is provided, the
/// descriptor is finalized for full backend support.
///
/// NOTE: Operation type support depends on NodeFactory and unpackOperation().
/// Unsupported operation types will return an error during unpacking.
[[nodiscard]] inline std::pair<std::unique_ptr<ScopedHipdnnBackendDescriptor>, Error>
    deserializeAndUnpackGraph(hipdnnHandle_t handle,
                              const std::vector<uint8_t>& data,
                              std::vector<std::shared_ptr<graph::INode>>& outNodes,
                              graph::GraphAttributes& outGraphAttrs,
                              std::optional<int64_t>& outPreferredEngineId)
{
    ScopedHipdnnBackendDescriptor graphDesc(data.data(), data.size());
    if(!graphDesc.valid())
    {
        return std::make_pair(
            std::unique_ptr<ScopedHipdnnBackendDescriptor>(nullptr),
            Error{ErrorCode::HIPDNN_BACKEND_ERROR,
                  "Failed to create backend graph descriptor from serialized data"});
    }

    if(handle != nullptr)
    {
        auto setStatus = hipdnnBackend()->backendSetAttribute(graphDesc.get(),
                                                              HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                                              HIPDNN_TYPE_HANDLE,
                                                              1,
                                                              static_cast<const void*>(&handle));
        if(setStatus != HIPDNN_STATUS_SUCCESS)
        {
            std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> backendErrMsg{};
            hipdnnBackend()->getLastErrorString(backendErrMsg.data(), backendErrMsg.size());
            return std::make_pair(
                std::unique_ptr<ScopedHipdnnBackendDescriptor>(nullptr),
                Error{ErrorCode::HIPDNN_BACKEND_ERROR,
                      std::string("Failed to set handle on the graph. Backend error: ")
                          + backendErrMsg.data()});
        }

        auto finStatus = hipdnnBackend()->backendFinalize(graphDesc.get());
        if(finStatus != HIPDNN_STATUS_SUCCESS)
        {
            std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> backendErrMsg{};
            hipdnnBackend()->getLastErrorString(backendErrMsg.data(), backendErrMsg.size());
            return std::make_pair(
                std::unique_ptr<ScopedHipdnnBackendDescriptor>(nullptr),
                Error{ErrorCode::HIPDNN_BACKEND_ERROR,
                      std::string("Failed to finalize backend descriptor for the graph. Backend "
                                  "error: ")
                          + backendErrMsg.data()});
        }
    }

    auto unpackErr
        = unpackGraphDescriptor(graphDesc.get(), outNodes, outGraphAttrs, outPreferredEngineId);
    if(unpackErr.is_bad())
    {
        return std::make_pair(std::unique_ptr<ScopedHipdnnBackendDescriptor>(nullptr), unpackErr);
    }

    return std::make_pair(std::make_unique<ScopedHipdnnBackendDescriptor>(std::move(graphDesc)),
                          Error{});
}

} // namespace hipdnn_frontend::detail
