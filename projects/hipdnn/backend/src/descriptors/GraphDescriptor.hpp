// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "IGraphOperation.hpp"
#include <flatbuffers/detached_buffer.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <memory>
#include <optional>
#include <vector>

namespace hipdnn_backend
{

class GraphDescriptor : public HipdnnBackendDescriptorImpl<GraphDescriptor>
{
private:
    // GraphDescriptor supports two mutually exclusive entry paths:
    //   1. FlatBuffer path: populated via deserializeGraph() from a serialized FlatBuffer.
    //      The serialized bytes are cached in _graphSerializedBuffer. When getOperations()
    //      is called, the buffer is lazily re-parsed and nodes are unpacked into _operations
    //      via NodeFactory. Node types without IGraphOperation implementations (e.g., Pointwise)
    //      cause NodeFactory to throw.
    //   2. C-API path: populated via setAttribute(OPS) which calls setOperations().
    //      Individual operation descriptors are accumulated in _operations.
    //      finalize() builds a GraphT from _operations via buildGraphFromOperations().
    // These paths are mutually exclusive: once one path is active, attempting to use the other
    // throws HIPDNN_STATUS_NOT_SUPPORTED. This prevents subtle bugs from mixing deserialized
    // state with C-API-provided operations.
    // getOperations() assumes single-threaded access (feature-flag guarded).
    // getOperations() does not require finalization, to support the deserialization
    // lifting path for unfinalized descriptors.

    hipdnnHandle_t _handle = nullptr;

    // Mutable because finalize() may populate this from a const-observable perspective
    // (the serialized buffer is a cache of the logical graph state).
    mutable flatbuffers::DetachedBuffer _graphSerializedBuffer;

    // Populated via setOperations() (C-API flow) or lazily from _graphSerializedBuffer (FlatBuffer flow).
    // Stored as IBackendDescriptor so getOperations() can pack them without cross-casting.
    // All entries are validated to implement IGraphOperation at insertion time.
    mutable std::vector<std::shared_ptr<IBackendDescriptor>> _operations;

    // Graph-level attributes set via setAttribute (applied during buildGraphFromOperations)
    hipdnn_data_sdk::data_objects::DataType _computeDataType
        = hipdnn_data_sdk::data_objects::DataType::UNSET;
    hipdnn_data_sdk::data_objects::DataType _intermediateDataType
        = hipdnn_data_sdk::data_objects::DataType::UNSET;
    hipdnn_data_sdk::data_objects::DataType _ioDataType
        = hipdnn_data_sdk::data_objects::DataType::UNSET;

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
    /// For FlatBuffer-flow descriptors, lazily unpacks from _graphSerializedBuffer into _operations on first call.
    void getOperations(hipdnnBackendAttributeType_t attributeType,
                       int64_t requestedElementCount,
                       int64_t* elementCount,
                       void* arrayOfElements) const;

    void getPreferredEngineId(hipdnnBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements) const;

    // Build GraphT from operation descriptors and return it (C-API flow)
    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> buildGraphFromOperations();

public:
    void finalize() override;

    /// Gets a graph attribute by name.
    ///
    /// When attributeName is HIPDNN_ATTR_OPERATIONGRAPH_OPS and attributeType is
    /// HIPDNN_TYPE_BACKEND_DESCRIPTOR, the returned descriptors are newly allocated
    /// via packDescriptor(). Ownership transfers to the caller, who must delete them.
    /// @see BackendDescriptor.hpp packDescriptor() for the ownership convention.
    ///
    /// For FlatBuffer-flow descriptors, the first OPS query lazily unpacks from _graphSerializedBuffer
    /// into _operations via NodeFactory. Unsupported node types throw HIPDNN_STATUS_NOT_SUPPORTED.
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

    virtual hipdnnPluginConstData_t getSerializedGraph() const;
    virtual hipdnnHandle_t getHandle() const;

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;
};
}
