// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GraphDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "FlatbufferUtilities.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"

namespace hipdnn_backend
{

void GraphDescriptor::finalize()
{
    THROW_IF_NULL(_graph, HIPDNN_STATUS_BAD_PARAM, "GraphDescriptor::finalize: graph is null");
    THROW_IF_NULL(_handle, HIPDNN_STATUS_BAD_PARAM, "GraphDescriptor::finalize: handle is null");
    HipdnnBackendDescriptorImpl<GraphDescriptor>::finalize();
}

void GraphDescriptor::setHandle(hipdnnBackendAttributeType_t attributeType,
                                int64_t elementCount,
                                const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_HANDLE,
                HIPDNN_STATUS_BAD_PARAM,
                "GraphDescriptor failed to set handle: Invalid attribute type.");
    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "GraphDescriptor failed to set handle: Invalid element count.");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "GraphDescriptor failed to set handle: Null pointer.");

    hipdnnHandle_t handle = *static_cast<const hipdnnHandle_t*>(arrayOfElements);

    THROW_IF_NULL(handle,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "GraphDescriptor failed to set handle: Handle is null.");

    _handle = handle;
}

void GraphDescriptor::getAttribute([[maybe_unused]] hipdnnBackendAttributeName_t attributeName,
                                   [[maybe_unused]] hipdnnBackendAttributeType_t attributeType,
                                   [[maybe_unused]] int64_t requestedElementCount,
                                   [[maybe_unused]] int64_t* elementCount,
                                   [[maybe_unused]] void* arrayOfElements) const
{
    throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                          "GraphDescriptor::getAttribute: not supported");
}

void GraphDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                   hipdnnBackendAttributeType_t attributeType,
                                   int64_t elementCount,
                                   const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "GraphDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATIONGRAPH_HANDLE:
        setHandle(attributeType, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("GraphDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }

    if(attributeName != HIPDNN_ATTR_OPERATIONGRAPH_HANDLE)
    {
        // Clear the serialized graph when the graph is modified
        _graphSerializedBuffer = flatbuffers::DetachedBuffer();
    }
}

void GraphDescriptor::deserializeGraph(const uint8_t* serializedGraph, size_t graphByteSize)
{
    THROW_IF_NULL(serializedGraph,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "GraphDescriptor::deserializeGraph: serializedGraph is null");
    THROW_IF_TRUE(graphByteSize == 0,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GraphDescriptor::deserializeGraph: graphByteSize is 0");

    // TODO: Consider skipping validation entirely, or maybe add an API option to skip it for schema extension cases.
    flatbuffer_utilities::convertSerializedGraphToGraph(serializedGraph, graphByteSize, _graph);
}

hipdnnPluginConstData_t GraphDescriptor::getSerializedGraph() const
{
    if(_graphSerializedBuffer.size() == 0)
    {
        THROW_IF_NULL(_graph,
                      HIPDNN_STATUS_INTERNAL_ERROR,
                      "GraphDescriptor::getSerializedGraph: graph is null");

        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(hipdnn_sdk::data_objects::Graph::Pack(builder, _graph.get()));

        _graphSerializedBuffer = builder.Release();
    }

    return {_graphSerializedBuffer.data(), _graphSerializedBuffer.size()};
}

hipdnnBackendDescriptorType_t GraphDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR;
}

hipdnnHandle_t GraphDescriptor::getHandle() const
{
    THROW_IF_NULL(_handle, HIPDNN_STATUS_BAD_PARAM, "GraphDescriptor::getHandle: handle is null");
    return _handle;
}

} // namespace hipdnn_backend
