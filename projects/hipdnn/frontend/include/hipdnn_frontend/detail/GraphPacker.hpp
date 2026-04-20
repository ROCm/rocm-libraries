// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <HipdnnDataType.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/Node.hpp>

#include <optional>
#include <vector>

namespace hipdnn_frontend::detail
{

/// Assembles a GraphDescriptor from pre-built operation descriptors without
/// setting a handle or finalizing. Unset data types are skipped.
inline Error assembleGraphDescriptor(const std::vector<ScopedHipdnnBackendDescriptor>& operations,
                                     std::optional<hipdnnDataType_t> computeDataType,
                                     std::optional<hipdnnDataType_t> intermediateDataType,
                                     std::optional<hipdnnDataType_t> ioDataType,
                                     const std::optional<int64_t>& preferredEngineId,
                                     const std::string& name,
                                     std::unique_ptr<ScopedHipdnnBackendDescriptor>& outGraphDesc)
{
    ScopedHipdnnBackendDescriptor graphDesc(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    if(!graphDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create GraphDescriptor"};
    }

    // Set operations on graph
    std::vector<hipdnnBackendDescriptor_t> opDescPtrs;
    opDescPtrs.reserve(operations.size());
    for(const auto& desc : operations)
    {
        opDescPtrs.push_back(desc.get());
    }
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(graphDesc.get(),
                                             HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             static_cast<int64_t>(opDescPtrs.size()),
                                             static_cast<const void*>(opDescPtrs.data())),
        "Failed to set operations on GraphDescriptor");

    // Set graph-level data types (skip if unset)
    if(computeDataType.has_value())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(graphDesc.get(),
                                                   HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT,
                                                   HIPDNN_TYPE_DATA_TYPE,
                                                   *computeDataType,
                                                   "compute data type on GraphDescriptor"));
    }

    if(intermediateDataType.has_value())
    {
        HIPDNN_CHECK_ERROR(
            setDescriptorAttrScalar(graphDesc.get(),
                                    HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT,
                                    HIPDNN_TYPE_DATA_TYPE,
                                    *intermediateDataType,
                                    "intermediate data type on GraphDescriptor"));
    }

    if(ioDataType.has_value())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(graphDesc.get(),
                                                   HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT,
                                                   HIPDNN_TYPE_DATA_TYPE,
                                                   *ioDataType,
                                                   "io data type on GraphDescriptor"));
    }

    // Set preferred engine ID if specified
    if(preferredEngineId.has_value())
    {
        auto engineId = preferredEngineId.value();
        HIPDNN_CHECK_ERROR(
            setDescriptorAttrScalar(graphDesc.get(),
                                    HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT,
                                    HIPDNN_TYPE_INT64,
                                    engineId,
                                    "preferred engine ID on GraphDescriptor"));
    }

    // Set graph name if non-empty
    if(!name.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            graphDesc.get(), HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT, name, "graph name"));
    }

    outGraphDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(std::move(graphDesc));
    return {};
}

/// Overload that also sets a handle, finalizes, and makes the descriptor
/// ready for engine selection and execution.
inline Error assembleGraphDescriptor(const std::vector<ScopedHipdnnBackendDescriptor>& operations,
                                     hipdnnHandle_t handle,
                                     std::optional<hipdnnDataType_t> computeDataType,
                                     std::optional<hipdnnDataType_t> intermediateDataType,
                                     std::optional<hipdnnDataType_t> ioDataType,
                                     const std::optional<int64_t>& preferredEngineId,
                                     const std::string& name,
                                     std::unique_ptr<ScopedHipdnnBackendDescriptor>& outGraphDesc)
{
    HIPDNN_CHECK_ERROR(assembleGraphDescriptor(operations,
                                               computeDataType,
                                               intermediateDataType,
                                               ioDataType,
                                               preferredEngineId,
                                               name,
                                               outGraphDesc));

    // Set handle then finalize for full backend support
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(outGraphDesc->get(),
                                             HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                             HIPDNN_TYPE_HANDLE,
                                             1,
                                             static_cast<const void*>(&handle)),
        "Failed to set handle on GraphDescriptor");

    HIPDNN_CHECK_ERROR(finalizeDescriptor(outGraphDesc->get(), "GraphDescriptor"));

    return {};
}

} // namespace hipdnn_frontend::detail
