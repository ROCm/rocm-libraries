// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GraphDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "DataTypeConversion.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "FlatbufferUtilities.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"

#include <unordered_map>

namespace hipdnn_backend
{

void GraphDescriptor::finalize()
{
    THROW_IF_NULL(_handle, HIPDNN_STATUS_BAD_PARAM, "GraphDescriptor::finalize: handle is null");

    // If operations were set, build graph from them
    if(!_operations.empty())
    {
        buildGraphFromOperations();
    }

    THROW_IF_NULL(_graph, HIPDNN_STATUS_BAD_PARAM, "GraphDescriptor::finalize: graph is null");
    HipdnnBackendDescriptorImpl<GraphDescriptor>::finalize();
}

void GraphDescriptor::buildGraphFromOperations()
{
    _graph = std::make_unique<hipdnn_data_sdk::data_objects::GraphT>();

    // Apply graph-level attributes
    _graph->compute_data_type = _computeDataType;
    _graph->intermediate_data_type = _intermediateDataType;
    _graph->io_data_type = _ioDataType;
    _graph->preferred_engine_id = _preferredEngineId;

    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> seenTensors;

    for(const auto& op : _operations)
    {
        // Collect unique tensors (deduplicated by UID)
        for(const auto& tensorDesc : op->getTensorDescriptors())
        {
            auto uid = tensorDesc->getData().uid;
            auto it = seenTensors.find(uid);
            if(it != seenTensors.end())
            {
                THROW_IF_FALSE(it->second.get() == tensorDesc.get(),
                               HIPDNN_STATUS_BAD_PARAM,
                               "GraphDescriptor::buildGraphFromOperations: Tensor UID "
                                   + std::to_string(uid)
                                   + " used with different descriptor objects");
            }
            else
            {
                seenTensors[uid] = tensorDesc;
                _graph->tensors.push_back(
                    std::make_unique<hipdnn_data_sdk::data_objects::TensorAttributesT>(
                        tensorDesc->getData()));
            }
        }

        // Build node from operation
        _graph->nodes.push_back(op->buildNode());
    }

    // TODO: Keep _operations instead of clearing, and add a getAttribute path
    // for HIPDNN_ATTR_OPERATIONGRAPH_OPS to allow retrieving the operations.
    // This will enable rebuilding graphs in the frontend from a serialized
    // backend graph.
    _operations.clear();
}

void GraphDescriptor::setDataType(hipdnnBackendAttributeName_t attributeName,
                                  hipdnnBackendAttributeType_t attributeType,
                                  int64_t elementCount,
                                  const void* arrayOfElements)
{
    const char* errorPrefix = "GraphDescriptor::setDataType";

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT:
        hipdnn_backend::setDataType(
            _computeDataType, attributeType, elementCount, arrayOfElements, errorPrefix);
        break;
    case HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT:
        hipdnn_backend::setDataType(
            _intermediateDataType, attributeType, elementCount, arrayOfElements, errorPrefix);
        break;
    case HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT:
        hipdnn_backend::setDataType(
            _ioDataType, attributeType, elementCount, arrayOfElements, errorPrefix);
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "GraphDescriptor::setDataType: Unsupported attribute name "
                                  + std::string(hipdnnGetAttributeNameString(attributeName)));
    }
}

void GraphDescriptor::setPreferredEngineId(hipdnnBackendAttributeType_t attributeType,
                                           int64_t elementCount,
                                           const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_INT64,
                HIPDNN_STATUS_BAD_PARAM,
                "GraphDescriptor::setPreferredEngineId: Invalid attribute type.");
    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "GraphDescriptor::setPreferredEngineId: Invalid element count.");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "GraphDescriptor::setPreferredEngineId: Null pointer.");

    _preferredEngineId = *static_cast<const int64_t*>(arrayOfElements);
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

void GraphDescriptor::setOperations(hipdnnBackendAttributeType_t attributeType,
                                    int64_t elementCount,
                                    const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "GraphDescriptor::setOperations: attributeType mismatch");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "GraphDescriptor::setOperations: arrayOfElements is null");

    auto descriptors = static_cast<HipdnnBackendDescriptor* const*>(arrayOfElements);

    // Validate all descriptors into a temporary vector before modifying state,
    // so that a validation failure doesn't leave _operations in a partial state.
    std::vector<std::shared_ptr<IGraphOperation>> newOperations;
    newOperations.reserve(static_cast<size_t>(elementCount));

    for(int64_t i = 0; i < elementCount; ++i)
    {
        THROW_IF_NULL(descriptors[i],
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "GraphDescriptor::setOperations: descriptor is null");
        THROW_IF_FALSE(descriptors[i]->isFinalized(),
                       HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                       "GraphDescriptor::setOperations: Operation descriptor not finalized");

        auto graphOp = descriptors[i]->tryAsInterface<IGraphOperation>();
        THROW_IF_NULL(graphOp,
                      HIPDNN_STATUS_NOT_SUPPORTED,
                      "GraphDescriptor::setOperations: Descriptor does not implement "
                      "IGraphOperation");
        newOperations.push_back(graphOp);
    }

    _operations = std::move(newOperations);
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
    case HIPDNN_ATTR_OPERATIONGRAPH_OPS:
        setOperations(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT:
    case HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT:
    case HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT:
        setDataType(attributeName, attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT:
        setPreferredEngineId(attributeType, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("GraphDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
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
    // TODO: Support serializing a graph without finalization, and deserializing
    // without a handle set.
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "GraphDescriptor::getSerializedGraph: graph is not finalized");

    if(_graphSerializedBuffer.size() == 0)
    {
        THROW_IF_NULL(_graph,
                      HIPDNN_STATUS_INTERNAL_ERROR,
                      "GraphDescriptor::getSerializedGraph: graph is null");

        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(hipdnn_data_sdk::data_objects::Graph::Pack(builder, _graph.get()));

        _graphSerializedBuffer = builder.Release();

        flatbuffers::Verifier verifier(_graphSerializedBuffer.data(),
                                       _graphSerializedBuffer.size());
        if(!hipdnn_data_sdk::data_objects::VerifyGraphBuffer(verifier))
        {
            _graphSerializedBuffer = flatbuffers::DetachedBuffer();
            throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                                  "GraphDescriptor::getSerializedGraph: serialized graph "
                                  "failed verification");
        }
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

std::string GraphDescriptor::toString() const
{
    std::string str = "GraphDescriptor: {handle=";
    str += _handle != nullptr ? fmt::format("{:p}", static_cast<const void*>(_handle)) : "null";
    str += ", serializedGraphSize="
           + std::to_string(_graphSerializedBuffer.size() > 0 ? _graphSerializedBuffer.size() : 0);
    str += "}";
    return str;
}

} // namespace hipdnn_backend
