// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "IGraphOperation.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/paged_cache_store_attributes_generated.h>
#include <unordered_map>

namespace hipdnn_backend
{

class PagedCacheStoreOperationDescriptor
    : public HipdnnBackendDescriptorImpl<PagedCacheStoreOperationDescriptor>,
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
    const hipdnn_flatbuffers_sdk::data_objects::PagedCacheStoreAttributesT& getData() const
    {
        return _data;
    }

    // Access to tensor descriptor references for graph building
    std::shared_ptr<TensorDescriptor> getXDesc() const
    {
        return _xDesc;
    }
    std::shared_ptr<TensorDescriptor> getSlotMappingDesc() const
    {
        return _slotMappingDesc;
    }
    std::shared_ptr<TensorDescriptor> getContainerDesc() const
    {
        return _containerDesc;
    }
    std::shared_ptr<TensorDescriptor> getScaleDesc() const
    {
        return _scaleDesc;
    }

    // IGraphOperation interface
    std::vector<std::shared_ptr<TensorDescriptor>> getTensorDescriptors() const override;
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT> buildNode() const override;

    // Creates a finalized PagedCacheStoreOperationDescriptor directly from a FlatBuffer NodeT.
    // Casts nodeT.attributes to PagedCacheStoreAttributes internally, then directly assigns
    // the data struct, looks up tensor descriptors from the tensor map, and calls finalize().
    static std::shared_ptr<PagedCacheStoreOperationDescriptor>
        fromNode(const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
                 const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap);

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    hipdnn_flatbuffers_sdk::data_objects::PagedCacheStoreAttributesT _data;

    // Store tensor descriptor references for validation and graph building
    std::shared_ptr<TensorDescriptor> _xDesc;
    std::shared_ptr<TensorDescriptor> _slotMappingDesc;
    std::shared_ptr<TensorDescriptor> _containerDesc;
    std::shared_ptr<TensorDescriptor> _scaleDesc;

    std::string _name;
};

} // namespace hipdnn_backend
