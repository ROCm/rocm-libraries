// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <hipdnn_frontend/detail/OperationUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/Node.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hipdnn_frontend::detail
{

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

/// Optionally sets a handle and finalizes a graph descriptor, then unpacks it
/// into frontend nodes and graph-level attributes. Takes ownership of the
/// descriptor by value (move-only).
[[nodiscard]] inline std::pair<std::unique_ptr<ScopedHipdnnBackendDescriptor>, Error>
    finalizeAndUnpackGraph(ScopedHipdnnBackendDescriptor graphDesc,
                           hipdnnHandle_t handle,
                           std::vector<std::shared_ptr<graph::INode>>& outNodes,
                           graph::GraphAttributes& outGraphAttrs,
                           std::optional<int64_t>& outPreferredEngineId)
{
    if(handle != nullptr)
    {
        auto setErr = setDescriptorAttrScalar(graphDesc.get(),
                                              HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                              HIPDNN_TYPE_HANDLE,
                                              handle,
                                              "handle on the graph");
        if(setErr.is_bad())
        {
            return std::make_pair(nullptr, std::move(setErr));
        }

        auto finErr = finalizeDescriptor(graphDesc.get(), "graph descriptor");
        if(finErr.is_bad())
        {
            return std::make_pair(nullptr, std::move(finErr));
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

    return finalizeAndUnpackGraph(
        std::move(graphDesc), handle, outNodes, outGraphAttrs, outPreferredEngineId);
}

/// Deserializes a backend graph descriptor from a JSON string and unpacks it into
/// frontend nodes and graph-level attributes. If a handle is provided, the
/// descriptor is finalized for full backend support.
///
/// NOTE: Operation type support depends on NodeFactory and unpackOperation().
/// Unsupported operation types will return an error during unpacking.
[[nodiscard]] inline std::pair<std::unique_ptr<ScopedHipdnnBackendDescriptor>, Error>
    deserializeAndUnpackJsonGraph(hipdnnHandle_t handle,
                                  const std::string& jsonData,
                                  std::vector<std::shared_ptr<graph::INode>>& outNodes,
                                  graph::GraphAttributes& outGraphAttrs,
                                  std::optional<int64_t>& outPreferredEngineId)
{
    ScopedHipdnnBackendDescriptor graphDesc(jsonData.c_str(), jsonData.size());
    if(!graphDesc.valid())
    {
        return std::make_pair(std::unique_ptr<ScopedHipdnnBackendDescriptor>(nullptr),
                              Error{ErrorCode::HIPDNN_BACKEND_ERROR,
                                    "Failed to create backend graph descriptor from JSON data"});
    }

    return finalizeAndUnpackGraph(
        std::move(graphDesc), handle, outNodes, outGraphAttrs, outPreferredEngineId);
}

} // namespace hipdnn_frontend::detail
