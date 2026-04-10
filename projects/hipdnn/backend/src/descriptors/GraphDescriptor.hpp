// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "IGraphOperation.hpp"
#include <flatbuffers/detached_buffer.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hipdnn_backend
{

class GraphDescriptor : public HipdnnBackendDescriptorImpl<GraphDescriptor>
{
private:
    // _operations is the sole source of truth for the graph's operation list.
    // Two entry paths populate _operations:
    //   1. C-API path: populated via setAttribute(OPS) which calls setOperations().
    //   2. FlatBuffer path: populated via deserializeGraph() from a serialized FlatBuffer.
    //      The buffer is eagerly unpacked into _operations via NodeFactory.
    //
    // _graphSerializedBuffer is a cache derived from _operations. It is populated by:
    //   - finalize() — validates handle + non-empty operations, then serializes.
    //   - buildSerializedGraph() — serializes from operations without requiring finalization.
    // Mutations (e.g., setAttribute) invalidate the cache via invalidateCache().
    // getSerializedGraph() is a const getter that returns the cached buffer or throws.

    hipdnnHandle_t _handle = nullptr;

    // Cached serialized graph buffer. Populated by finalize() or buildSerializedGraph().
    // Cleared by invalidateCache() when operations are mutated.
    flatbuffers::DetachedBuffer _graphSerializedBuffer;

    // Source of truth for the graph's operations. Populated via setOperations() (C-API flow)
    // or eagerly from deserializeGraph() (FlatBuffer flow).
    // Stored as IBackendDescriptor so getOperations() can pack them without cross-casting.
    // All entries are validated to implement IGraphOperation at insertion time.
    std::vector<std::shared_ptr<IBackendDescriptor>> _operations;

    // Graph-level attributes set via setAttribute (applied during buildGraphFromOperations)
    hipdnn_flatbuffers_sdk::data_objects::DataType _computeDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    hipdnn_flatbuffers_sdk::data_objects::DataType _intermediateDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    hipdnn_flatbuffers_sdk::data_objects::DataType _ioDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;

    // Preferred engine ID, empty when unset.
    std::optional<int64_t> _preferredEngineId = std::nullopt;

    // Optional human-readable name for the graph, empty when unset.
    std::string _name;

    void setHandle(hipdnnBackendAttributeType_t attributeType,
                   int64_t elementCount,
                   const void* arrayOfElements);

    void setOperations(hipdnnBackendAttributeType_t attributeType,
                       int64_t elementCount,
                       const void* arrayOfElements);

    void setDataType(hipdnnBackendAttributeName_t attributeName,
                     hipdnnBackendAttributeType_t attributeType,
                     int64_t elementCount,
                     const void* arrayOfElements);

    void setPreferredEngineId(hipdnnBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              const void* arrayOfElements);

    /// Returns operation descriptors via packDescriptor(). Caller owns the returned pointers.
    void getOperations(hipdnnBackendAttributeType_t attributeType,
                       int64_t requestedElementCount,
                       int64_t* elementCount,
                       void* arrayOfElements) const;

    void getPreferredEngineId(hipdnnBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements) const;

    // Build GraphT from operation descriptors and return it
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT> buildGraphFromOperations() const;

    // Clears the cached serialized buffer.
    // Called when _operations is mutated to keep the cache consistent.
    void invalidateCache();

public:
    void finalize() override;

    /// Gets a graph attribute by name.
    ///
    /// When attributeName is HIPDNN_ATTR_OPERATIONGRAPH_OPS and attributeType is
    /// HIPDNN_TYPE_BACKEND_DESCRIPTOR, the returned descriptors are newly allocated
    /// via packDescriptor(). Ownership transfers to the caller, who must delete them.
    /// @see BackendDescriptor.hpp packDescriptor() for the ownership convention.
    void getAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) const override;

    void setAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements) override;

    void deserializeGraph(const uint8_t* serializedGraph, size_t graphByteSize);

    /// Builds and caches the serialized graph buffer from operations.
    /// No-op if the buffer is already populated (e.g. by finalize()).
    void buildSerializedGraph();

    /// Returns the cached serialized graph buffer.
    /// Throws HIPDNN_STATUS_BAD_PARAM if the buffer has not been populated
    /// by finalize() or buildSerializedGraph().
    virtual hipdnnPluginConstData_t getSerializedGraph() const;

    /// Returns the graph as a JSON string, computed from the serialized buffer.
    /// Requires the buffer to be populated by finalize() or buildSerializedGraph().
    virtual std::string getSerializedJsonGraph() const;

    /// Creates a GraphDescriptor from a JSON graph string.
    /// The JSON is converted to FlatBuffer binary and deserialized.
    /// Throws HIPDNN_STATUS_BAD_PARAM if the JSON is malformed.
    static void
        createFromJsonGraph(GraphDescriptor& desc, const char* jsonGraph, size_t jsonByteSize);

    virtual hipdnnHandle_t getHandle() const;

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;
};
}
