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

#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <logging/GraphLogger.hpp>
#include <nlohmann/json.hpp>
#include <unordered_map>

namespace hipdnn_backend
{

void GraphDescriptor::invalidateCache()
{
    _graphSerializedBuffer = flatbuffers::DetachedBuffer();
}

void GraphDescriptor::finalize()
{
    THROW_IF_NULL(_handle, HIPDNN_STATUS_BAD_PARAM, "GraphDescriptor::finalize: handle is null");

    THROW_IF_TRUE(_operations.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "GraphDescriptor::finalize: no operations set");

    invalidateCache();
    buildSerializedGraph();

    HipdnnBackendDescriptorImpl<GraphDescriptor>::finalize();

    if(logging::GraphLogger::isEnabled())
    {
        auto serialized = getSerializedGraph();
        logging::GraphLogger::logGraph(static_cast<const uint8_t*>(serialized.ptr),
                                       serialized.size);
    }
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT>
    GraphDescriptor::buildGraphFromOperations() const
{
    auto graph = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::GraphT>();

    // Apply graph-level attributes
    graph->compute_data_type = _computeDataType;
    graph->intermediate_data_type = _intermediateDataType;
    graph->io_data_type = _ioDataType;
    graph->preferred_engine_id = _preferredEngineId;
    graph->is_override_shape_enabled = _isOverrideShapeEnabled;
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
                    std::make_unique<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>(
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
    case HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT:
        getScalar(_isOverrideShapeEnabled,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "GraphDescriptor::getAttribute()");
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

    HipdnnBackendDescriptor::packDescriptorArray(
        _operations, static_cast<HipdnnBackendDescriptor**>(arrayOfElements));
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
    case HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT:
        setScalar(_isOverrideShapeEnabled,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "GraphDescriptor::setAttribute()");
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
        invalidateCache();
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

    invalidateCache();

    // Parse FlatBuffer and eagerly unpack into _operations
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT> graph;
    flatbuffer_utilities::convertSerializedGraphToGraph(serializedGraph, graphByteSize, graph);

    // Extract graph-level attributes
    _computeDataType = graph->compute_data_type;
    _intermediateDataType = graph->intermediate_data_type;
    _ioDataType = graph->io_data_type;
    _preferredEngineId = graph->preferred_engine_id;
    _isOverrideShapeEnabled = graph->is_override_shape_enabled;
    _name = graph->name;

    // Populate _operations from the deserialized graph nodes
    auto tensorMap = NodeFactory::buildTensorMap(graph->tensors);
    std::vector<std::shared_ptr<IBackendDescriptor>> unpacked;
    unpacked.reserve(graph->nodes.size());
    for(const auto& nodeT : graph->nodes)
    {
        unpacked.push_back(NodeFactory::createOperationFromNode(*nodeT, tensorMap));
    }
    _operations = std::move(unpacked);
}

void GraphDescriptor::buildSerializedGraph()
{
    if(_graphSerializedBuffer.size() > 0)
    {
        return;
    }

    auto graph = buildGraphFromOperations();
    THROW_IF_NULL(graph,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "GraphDescriptor::buildSerializedGraph: graph is null");

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::Graph::Pack(builder, graph.get()));
    _graphSerializedBuffer = builder.Release();

    flatbuffers::Verifier verifier(_graphSerializedBuffer.data(), _graphSerializedBuffer.size());
    THROW_IF_FALSE(hipdnn_flatbuffers_sdk::data_objects::VerifyGraphBuffer(verifier),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "GraphDescriptor::buildSerializedGraph: serialized graph failed verification");
}

hipdnnPluginConstData_t GraphDescriptor::getSerializedGraph() const
{
    THROW_IF_TRUE(_graphSerializedBuffer.size() == 0,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GraphDescriptor::getSerializedGraph: serialized buffer not populated. "
                  "Call finalize() or buildSerializedGraph() first.");

    return {_graphSerializedBuffer.data(), _graphSerializedBuffer.size()};
}

std::string GraphDescriptor::getSerializedJsonGraph() const
{
    // JSON is reconstructed from the cached binary buffer on each call
    // (binary -> FlatBuffer unpack -> JSON serialize). The JSON representation
    // is not cached because this path is not performance-critical.
    auto data = getSerializedGraph();

    auto* graph = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Graph>(
        static_cast<const uint8_t*>(data.ptr));
    const nlohmann::json j = *graph;
    return j.dump();
}

void GraphDescriptor::createFromJsonGraph(GraphDescriptor& desc,
                                          const char* jsonGraph,
                                          size_t jsonByteSize)
{
    THROW_IF_NULL(jsonGraph,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "GraphDescriptor::createFromJsonGraph: jsonGraph is null");
    THROW_IF_TRUE(jsonByteSize == 0,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GraphDescriptor::createFromJsonGraph: jsonByteSize is 0");

    try
    {
        auto j = nlohmann::json::parse(jsonGraph, jsonGraph + jsonByteSize);

        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(
            hipdnn_flatbuffers_sdk::json::to<hipdnn_flatbuffers_sdk::data_objects::Graph>(builder,
                                                                                          j));

        auto buf = builder.Release();
        desc.deserializeGraph(buf.data(), buf.size());
    }
    catch(const nlohmann::json::parse_error& e)
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              std::string("GraphDescriptor::createFromJsonGraph: invalid JSON: ")
                                  + e.what());
    }
    catch(const HipdnnException&)
    {
        throw;
    }
    catch(const std::exception& e)
    {
        throw HipdnnException(
            HIPDNN_STATUS_BAD_PARAM,
            std::string("GraphDescriptor::createFromJsonGraph: invalid graph data: ") + e.what());
    }
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

bool GraphDescriptor::isOverrideShapeEnabled() const
{
    return _isOverrideShapeEnabled;
}

std::string GraphDescriptor::toString() const
{
    std::string str = "GraphDescriptor: {handle=";
    str += _handle != nullptr ? fmt::format("{:p}", static_cast<const void*>(_handle)) : "null";
    str += ", name=" + (_name.empty() ? std::string("(empty)") : _name);
    str += ", serializedGraphSize=" + std::to_string(_graphSerializedBuffer.size());
    str += '}';
    return str;
}

} // namespace hipdnn_backend
