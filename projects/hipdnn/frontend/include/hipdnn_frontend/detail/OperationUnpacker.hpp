// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <HipdnnOperationType.h>
#include <array>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/Node.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

/// Queries the operation type directly via HIPDNN_ATTR_OPERATION_TYPE_EXT.
/// Returns an error on failure — no fallback probing.
[[nodiscard]] inline std::pair<hipdnnOperationType_t, Error>
    queryOperationType(hipdnnBackendDescriptor_t opDesc)
{
    hipdnnOperationType_t typeValue = HIPDNN_OPERATION_TYPE_NOT_SET;
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
        return {HIPDNN_OPERATION_TYPE_NOT_SET,
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
    createNodeForType(hipdnnOperationType_t opType, const graph::GraphAttributes& graphAttrs)
{
    switch(opType)
    {
    case HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD:
        return {
            std::make_shared<graph::ConvolutionFpropNode>(graph::ConvFpropAttributes{}, graphAttrs),
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
