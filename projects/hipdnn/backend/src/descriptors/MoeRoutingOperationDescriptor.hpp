// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "IGraphOperation.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_routing_attributes_generated.h>
#include <unordered_map>

namespace hipdnn_backend
{

class MoeRoutingOperationDescriptor
    : public HipdnnBackendDescriptorImpl<MoeRoutingOperationDescriptor>,
      public IGraphOperation
{
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

    // Direct access to the underlying T struct for OperationGraphBuilder
    const hipdnn_flatbuffers_sdk::data_objects::MoeRoutingAttributesT& getData() const
    {
        return _data;
    }

    // Access to tensor descriptor references for graph building
    std::shared_ptr<TensorDescriptor> getTopkIdsDesc() const
    {
        return _topkIdsDesc;
    }
    std::shared_ptr<TensorDescriptor> getFirstTokenOffsetDesc() const
    {
        return _firstTokenOffsetDesc;
    }
    std::shared_ptr<TensorDescriptor> getTokenIndexDesc() const
    {
        return _tokenIndexDesc;
    }
    std::shared_ptr<TensorDescriptor> getTokenKsDesc() const
    {
        return _tokenKsDesc;
    }

    // IGraphOperation interface
    std::vector<std::shared_ptr<TensorDescriptor>> getTensorDescriptors() const override;
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT> buildNode() const override;

    // Creates a finalized MoeRoutingOperationDescriptor directly from a FlatBuffer NodeT.
    // Casts nodeT.attributes to MoeRoutingAttributes internally, then directly assigns
    // the data struct, looks up tensor descriptors from the tensor map, and calls finalize().
    static std::shared_ptr<MoeRoutingOperationDescriptor>
        fromNode(const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
                 const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap);

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    hipdnn_flatbuffers_sdk::data_objects::MoeRoutingAttributesT _data;

    // Store tensor descriptor references for validation and graph building
    std::shared_ptr<TensorDescriptor> _topkIdsDesc;
    std::shared_ptr<TensorDescriptor> _firstTokenOffsetDesc;
    std::shared_ptr<TensorDescriptor> _tokenIndexDesc;
    std::shared_ptr<TensorDescriptor> _tokenKsDesc;

    std::string _name;
};

} // namespace hipdnn_backend
