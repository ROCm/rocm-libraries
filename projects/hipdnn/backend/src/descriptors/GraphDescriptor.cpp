// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GraphDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "DataTypeConversion.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "FlatbufferUtilities.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "NodeFactory.hpp"

#include <logging/GraphLogger.hpp>
#include <unordered_map>

namespace hipdnn_backend
{

void GraphDescriptor::finalize()
{
    THROW_IF_NULL(_handle, HIPDNN_STATUS_BAD_PARAM, "GraphDescriptor::finalize: handle is null");

    if(_graphSerializedBuffer.size() > 0)
    {
        // Serialized buffer already cached (from deserializeGraph or previous finalize)
    }
    else if(!_operations.empty())
    {
        // C-API flow: build graph from operation descriptors, then serialize
        auto graph = buildGraphFromOperations();
        THROW_IF_NULL(graph, HIPDNN_STATUS_BAD_PARAM, "GraphDescriptor::finalize: graph is null");

        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(hipdnn_data_sdk::data_objects::Graph::Pack(builder, graph.get()));
        _graphSerializedBuffer = builder.Release();
    }
    else
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "GraphDescriptor::finalize: no operations set and no graph "
                              "deserialized");
    }

    // Verify the serialized buffer
    flatbuffers::Verifier verifier(_graphSerializedBuffer.data(), _graphSerializedBuffer.size());
    THROW_IF_FALSE(hipdnn_data_sdk::data_objects::VerifyGraphBuffer(verifier),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "GraphDescriptor::finalize: serialized graph failed verification");

    HipdnnBackendDescriptorImpl<GraphDescriptor>::finalize();

    if(logging::GraphLogger::isEnabled())
    {
        auto serialized = getSerializedGraph();
        logging::GraphLogger::logGraph(static_cast<const uint8_t*>(serialized.ptr),
                                       serialized.size);
    }
}

std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> GraphDescriptor::buildGraphFromOperations()
{
    auto graph = std::make_unique<hipdnn_data_sdk::data_objects::GraphT>();

    // Apply graph-level attributes
    graph->compute_data_type = _computeDataType;
    graph->intermediate_data_type = _intermediateDataType;
    graph->io_data_type = _ioDataType;
    graph->preferred_engine_id = _preferredEngineId;
    graph->name = _name;

    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> seenTensors;

    for(const auto& desc : _operations)
    {
        auto* op = desc->asGraphOperation();

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
                graph->tensors.push_back(
                    std::make_unique<hipdnn_data_sdk::data_objects::TensorAttributesT>(
                        tensorDesc->getData()));
            }
        }

        // Build node from operation
        graph->nodes.push_back(op->buildNode());
    }

    return graph;
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
    flatbuffers::Optional<int64_t> flatbufferOptional = flatbuffers::nullopt;
    setOptionalScalar<HIPDNN_TYPE_INT64>(flatbufferOptional,
                                         attributeType,
                                         elementCount,
                                         arrayOfElements,
                                         "GraphDescriptor::setPreferredEngineId");
    _preferredEngineId = flatbufferOptional;
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

void GraphDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                   hipdnnBackendAttributeType_t attributeType,
                                   int64_t requestedElementCount,
                                   int64_t* elementCount,
                                   void* arrayOfElements) const
{
    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATIONGRAPH_OPS:
        getOperations(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "GraphDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT:
        getDataType(_intermediateDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "GraphDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT:
        getDataType(_ioDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "GraphDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT:
        getPreferredEngineId(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "GraphDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("GraphDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void GraphDescriptor::getOperations(hipdnnBackendAttributeType_t attributeType,
                                    int64_t requestedElementCount,
                                    int64_t* elementCount,
                                    void* arrayOfElements) const
{
    checkGetArgs(HIPDNN_TYPE_BACKEND_DESCRIPTOR, attributeType, "GraphDescriptor::getAttribute()");

    // Lazy unpack: if the serialized buffer is populated and _operations is empty,
    // re-parse the buffer and unpack nodes into operations.
    // Build into a temporary vector so _operations is only populated on full success.
    if(_graphSerializedBuffer.size() > 0 && _operations.empty())
    {
        auto graphT = hipdnn_data_sdk::data_objects::UnPackGraph(_graphSerializedBuffer.data());
        auto tensorMap = NodeFactory::buildTensorMap(graphT->tensors);
        std::vector<std::shared_ptr<IBackendDescriptor>> unpacked;
        unpacked.reserve(graphT->nodes.size());
        for(const auto& nodeT : graphT->nodes)
        {
            unpacked.push_back(NodeFactory::createOperationFromNode(*nodeT, tensorMap));
        }
        _operations = std::move(unpacked);
    }

    auto count = static_cast<int64_t>(_operations.size());

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "GraphDescriptor::getAttribute(): elementCount is null");
        *elementCount = count;
        return;
    }

    THROW_IF_FALSE(requestedElementCount >= count,
                   HIPDNN_STATUS_BAD_PARAM,
                   "GraphDescriptor::getAttribute(): requestedElementCount < operation count");

    if(elementCount != nullptr)
    {
        *elementCount = count;
    }

    auto outputArray = static_cast<HipdnnBackendDescriptor**>(arrayOfElements);

    // Build into a local vector so no pointers leak if a later iteration fails.
    std::vector<HipdnnBackendDescriptor*> packed;
    packed.reserve(_operations.size());
    try
    {
        for(const auto& operation : _operations)
        {
            packed.push_back(HipdnnBackendDescriptor::packDescriptor(operation));
        }
    }
    catch(...)
    {
        for(auto* p : packed)
        {
            delete p;
        }
        throw;
    }

    std::copy(packed.begin(), packed.end(), outputArray);
}

void GraphDescriptor::getPreferredEngineId(hipdnnBackendAttributeType_t attributeType,
                                           int64_t requestedElementCount,
                                           int64_t* elementCount,
                                           void* arrayOfElements) const
{
    auto flatbufferOptional = _preferredEngineId;
    getOptionalScalar<HIPDNN_TYPE_INT64>(flatbufferOptional,
                                         attributeType,
                                         requestedElementCount,
                                         elementCount,
                                         arrayOfElements,
                                         "GraphDescriptor::getAttribute()");
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

    // The FlatBuffer and C-API flows are mutually exclusive. Setting operations on a
    // descriptor that was populated via deserializeGraph() is not supported.
    THROW_IF_TRUE(_graphSerializedBuffer.size() > 0,
                  HIPDNN_STATUS_NOT_SUPPORTED,
                  "GraphDescriptor::setOperations: cannot set operations on a graph populated "
                  "via deserializeGraph(). The FlatBuffer and C-API flows are mutually "
                  "exclusive.");

    auto descriptors = static_cast<HipdnnBackendDescriptor* const*>(arrayOfElements);

    // Validate all descriptors into a temporary vector before modifying state,
    // so that a validation failure doesn't leave _operations in a partial state.
    std::vector<std::shared_ptr<IBackendDescriptor>> newOperations;
    newOperations.reserve(static_cast<size_t>(elementCount));

    for(int64_t i = 0; i < elementCount; ++i)
    {
        THROW_IF_NULL(descriptors[i],
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "GraphDescriptor::setOperations: descriptor is null");
        THROW_IF_FALSE(descriptors[i]->isFinalized(),
                       HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                       "GraphDescriptor::setOperations: Operation descriptor not finalized");

        // Validate that the descriptor implements IGraphOperation before storing.
        THROW_IF_NULL(descriptors[i]->tryAsGraphOperation(),
                      HIPDNN_STATUS_NOT_SUPPORTED,
                      "GraphDescriptor::setOperations: Descriptor does not implement "
                      "IGraphOperation");
        newOperations.push_back(descriptors[i]->getImpl());
    }

    // Accumulate operations (multiple setAttribute calls append to existing operations)
    _operations.insert(_operations.end(),
                       std::make_move_iterator(newOperations.begin()),
                       std::make_move_iterator(newOperations.end()));
}

void GraphDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                   hipdnnBackendAttributeType_t attributeType,
                                   int64_t elementCount,
                                   const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "GraphDescriptor::setAttribute() failed: Already finalized.");

    bool invalidate = true;

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATIONGRAPH_HANDLE:
        setHandle(attributeType, elementCount, arrayOfElements);
        invalidate = false;
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
    case HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT:
        setString(
            _name, attributeType, elementCount, arrayOfElements, "GraphDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("GraphDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }

    // Attribute mutations invalidate the cached serialized graph.
    // Non-mutating attributes (e.g., HANDLE) set invalidate = false.
    if(invalidate)
    {
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
    THROW_IF_FALSE(_operations.empty(),
                   HIPDNN_STATUS_NOT_SUPPORTED,
                   "GraphDescriptor::deserializeGraph: cannot deserialize into a graph with "
                   "existing operations. The FlatBuffer and C-API flows are mutually exclusive.");

    // Parse FlatBuffer into a local GraphT to extract attributes, then re-serialize into
    // _graphSerializedBuffer. When getOperations() is called, the buffer is re-parsed on demand.
    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> graph;
    flatbuffer_utilities::convertSerializedGraphToGraph(serializedGraph, graphByteSize, graph);

    // Extract graph-level attributes
    _computeDataType = graph->compute_data_type;
    _intermediateDataType = graph->intermediate_data_type;
    _ioDataType = graph->io_data_type;
    _preferredEngineId = graph->preferred_engine_id;
    _name = graph->name;

    // Cache the serialized bytes for getSerializedGraph() by re-serializing from the parsed GraphT
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_data_sdk::data_objects::Graph::Pack(builder, graph.get()));
    _graphSerializedBuffer = builder.Release();
}

hipdnnPluginConstData_t GraphDescriptor::getSerializedGraph() const
{
    // For the FlatBuffer flow, the serialized buffer is populated during deserializeGraph()
    // and can be returned without requiring finalize(). This enables lazy serialization for
    // graphs that were deserialized and not modified.
    if(_graphSerializedBuffer.size() > 0)
    {
        return {_graphSerializedBuffer.data(), _graphSerializedBuffer.size()};
    }

    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "GraphDescriptor::getSerializedGraph: graph is not finalized");

    // After finalization, the buffer should always be populated
    THROW_IF_TRUE(_graphSerializedBuffer.size() == 0,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "GraphDescriptor::getSerializedGraph: serialized buffer is empty after "
                  "finalization");

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
    str += ", name=" + (_name.empty() ? std::string("(empty)") : _name);
    str += ", serializedGraphSize=" + std::to_string(_graphSerializedBuffer.size());
    str += "}";
    return str;
}

} // namespace hipdnn_backend
