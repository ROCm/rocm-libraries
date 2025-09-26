// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendDescriptor.hpp"
#include <flatbuffers/detached_buffer.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <memory>
#include <vector>

namespace hipdnn_backend
{

class GraphDescriptor : public HipdnnBackendDescriptorImpl<GraphDescriptor>
{
private:
    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> _graph;
    hipdnnHandle_t _handle = nullptr;
    mutable flatbuffers::DetachedBuffer _graphSerializedBuffer;

    void setHandle(hipdnnBackendAttributeType_t attributeType,
                   int64_t elementCount,
                   const void* arrayOfElements);

public:
    void finalize() override;

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
};
}
