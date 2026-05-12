// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <HipdnnOperationType.h>
#include <array>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
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
#include <hipdnn_frontend/node/ResampleFwdNode.hpp>
#include <hipdnn_frontend/node/SdpaBwdNode.hpp>
#include <hipdnn_frontend/node/SdpaFwdNode.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

/// Queries the operation type directly via HIPDNN_ATTR_OPERATION_TYPE_EXT.
/// Returns an error on failure — no fallback probing.
[[nodiscard]] inline std::pair<hipdnnOperationType_ext_t, Error>
    queryOperationType(hipdnnBackendDescriptor_t opDesc)
{
    hipdnnOperationType_ext_t typeValue = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t actualCount = 0;
    auto status = hipdnnBackend()->backendGetAttribute(opDesc,
                                                       HIPDNN_ATTR_OPERATION_TYPE_EXT,
                                                       HIPDNN_TYPE_OPERATION_TYPE_EXT,
                                                       1,
                                                       &actualCount,
                                                       &typeValue);
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> backendErrMsg{};
        hipdnnBackend()->getLastErrorString(backendErrMsg.data(), backendErrMsg.size());
        return {HIPDNN_OPERATION_TYPE_NOT_SET_EXT,
                {ErrorCode::HIPDNN_BACKEND_ERROR,
                 std::string("Failed to query HIPDNN_ATTR_OPERATION_TYPE_EXT. Backend error: ")
                     + backendErrMsg.data()}};
    }
    return {typeValue, {}};
}

/// Creates a frontend node of the appropriate type for the given backend operation type.
///
/// @param opType The backend operation type from HIPDNN_ATTR_OPERATION_TYPE_EXT
/// @param graphAttrs Graph-level attributes to associate with the created node
/// @return A shared_ptr to the created INode, or an error if the type is unsupported
[[nodiscard]] inline std::pair<std::shared_ptr<graph::INode>, Error>
    createNodeForType(hipdnnOperationType_ext_t opType, const graph::GraphAttributes& graphAttrs)
{
    switch(opType)
    {
    case HIPDNN_OPERATION_TYPE_BATCHNORM_EXT:
        return {std::make_shared<graph::BatchnormNode>(graph::BatchnormAttributes{}, graphAttrs),
                {}};
    case HIPDNN_OPERATION_TYPE_BATCHNORM_BACKWARD_EXT:
        return {std::make_shared<graph::BatchnormBackwardNode>(graph::BatchnormBackwardAttributes{},
                                                               graphAttrs),
                {}};
    case HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE_EXT:
        return {std::make_shared<graph::BatchnormInferenceNode>(
                    graph::BatchnormInferenceAttributes{}, graphAttrs),
                {}};
    case HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE_VARIANCE_EXT:
        return {std::make_shared<graph::BatchnormInferenceNodeVarianceExt>(
                    graph::BatchnormInferenceAttributesVarianceExt{}, graphAttrs),
                {}};
    case HIPDNN_OPERATION_TYPE_BLOCK_SCALE_DEQUANTIZE_EXT:
        return {std::make_shared<graph::BlockScaleDequantizeNode>(
                    graph::BlockScaleDequantizeAttributes{}, graphAttrs),
                {}};
    case HIPDNN_OPERATION_TYPE_BLOCK_SCALE_QUANTIZE_EXT:
        return {std::make_shared<graph::BlockScaleQuantizeNode>(
                    graph::BlockScaleQuantizeAttributes{}, graphAttrs),
                {}};
    case HIPDNN_OPERATION_TYPE_CONVOLUTION_BACKWARD_DATA_EXT:
        return {
            std::make_shared<graph::ConvolutionDgradNode>(graph::ConvDgradAttributes{}, graphAttrs),
            {}};
    case HIPDNN_OPERATION_TYPE_CONVOLUTION_BACKWARD_WEIGHTS_EXT:
        return {
            std::make_shared<graph::ConvolutionWgradNode>(graph::ConvWgradAttributes{}, graphAttrs),
            {}};
    case HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT:
        return {
            std::make_shared<graph::ConvolutionFpropNode>(graph::ConvFpropAttributes{}, graphAttrs),
            {}};
    case HIPDNN_OPERATION_TYPE_CUSTOM_OP_EXT:
        return {std::make_shared<graph::CustomOpNode>(graph::CustomOpAttributes{}, graphAttrs), {}};
    case HIPDNN_OPERATION_TYPE_LAYERNORM_EXT:
        return {std::make_shared<graph::LayerNormNode>(graph::LayernormAttributes{}, graphAttrs),
                {}};
    case HIPDNN_OPERATION_TYPE_MATMUL_EXT:
        return {std::make_shared<graph::MatmulNode>(graph::MatmulAttributes{}, graphAttrs), {}};
    case HIPDNN_OPERATION_TYPE_POINTWISE_EXT:
        return {std::make_shared<graph::PointwiseNode>(graph::PointwiseAttributes{}, graphAttrs),
                {}};
    case HIPDNN_OPERATION_TYPE_REDUCTION_EXT:
        return {std::make_shared<graph::ReductionNode>(graph::ReductionAttributes{}, graphAttrs),
                {}};
    case HIPDNN_OPERATION_TYPE_RMSNORM_EXT:
        return {std::make_shared<graph::RMSNormNode>(graph::RMSNormAttributes{}, graphAttrs), {}};
    case HIPDNN_OPERATION_TYPE_SDPA_BACKWARD_EXT:
        return {std::make_shared<graph::SdpaBwdNode>(graph::SdpaBackwardAttributes{}, graphAttrs),
                {}};
    case HIPDNN_OPERATION_TYPE_SDPA_FORWARD_EXT:
        return {std::make_shared<graph::SdpaFwdNode>(graph::SdpaAttributes{}, graphAttrs), {}};
    case HIPDNN_OPERATION_TYPE_RESAMPLE_FWD:
        return {
            std::make_shared<graph::ResampleFwdNode>(graph::ResampleFwdAttributes{}, graphAttrs),
            {}};
    default:
        return {nullptr,
                {ErrorCode::HIPDNN_BACKEND_ERROR,
                 "Unsupported operation type for graph lifting (type id: "
                     + std::to_string(static_cast<int>(opType)) + ")"}};
    }
}

/// Unpacks a backend operation descriptor and returns a frontend node.
///
/// Uses a two-phase approach:
/// 1. Query the operation type via HIPDNN_ATTR_OPERATION_TYPE_EXT
/// 2. Create the correct node type and unpack via virtual dispatch
///
/// Errors are fail-fast: any failure during type query or unpacking returns
/// immediately with a clear error message.
///
/// @param opDesc The backend operation descriptor to unpack
/// @param tensorMap A map from tensor UID to TensorAttributes for sharing tensors across operations
/// @param graphAttrs Graph-level attributes to associate with the created node
/// @return A shared_ptr to the created INode, or an error if unpacking fails
[[nodiscard]] inline std::pair<std::shared_ptr<graph::INode>, Error> unpackOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    const graph::GraphAttributes& graphAttrs)
{
    auto [opType, typeErr] = queryOperationType(opDesc);
    if(typeErr.is_bad())
    {
        return {nullptr,
                {typeErr.code,
                 std::string("Failed to determine operation type: ") + typeErr.get_message()}};
    }

    auto [node, nodeErr] = createNodeForType(opType, graphAttrs);
    if(nodeErr.is_bad())
    {
        return {nullptr, nodeErr};
    }

    auto err = node->unpack_from_descriptor(opDesc, tensorMap);
    if(err.is_bad())
    {
        return {nullptr, err};
    }

    return {node, {}};
}

} // namespace hipdnn_frontend::detail
