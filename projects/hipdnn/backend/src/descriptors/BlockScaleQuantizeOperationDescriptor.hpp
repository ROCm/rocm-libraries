// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "IGraphOperation.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_quantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <memory>
#include <unordered_map>

namespace hipdnn_backend
{

class BlockScaleQuantizeOperationDescriptor
    : public HipdnnBackendDescriptorImpl<BlockScaleQuantizeOperationDescriptor>,
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
    const hipdnn_flatbuffers_sdk::data_objects::BlockScaleQuantizeAttributesT& getData() const
    {
        return _data;
    }

    // Access to tensor descriptor references for graph building
    std::shared_ptr<TensorDescriptor> getXDesc() const
    {
        return _xDesc;
    }
    std::shared_ptr<TensorDescriptor> getYDesc() const
    {
        return _yDesc;
    }
    std::shared_ptr<TensorDescriptor> getScaleDesc() const
    {
        return _scaleDesc;
    }

    // Get compute data type for the operation (used when building graph nodes)
    hipdnn_flatbuffers_sdk::data_objects::DataType getComputeDataType() const
    {
        return _computeDataType;
    }

    // IGraphOperation interface
    std::vector<std::shared_ptr<TensorDescriptor>> getTensorDescriptors() const override;
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT> buildNode() const override;

    // Creates a finalized BlockScaleQuantizeOperationDescriptor directly from a FlatBuffer NodeT.
    // Casts nodeT.attributes to BlockScaleQuantizeAttributesT internally, then directly assigns
    // the data struct, looks up tensor descriptors from the tensor map, and calls finalize().
    static std::shared_ptr<BlockScaleQuantizeOperationDescriptor>
        fromNode(const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
                 const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap);

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    hipdnn_flatbuffers_sdk::data_objects::BlockScaleQuantizeAttributesT _data;

    std::string _name;

    // Store tensor descriptor references for validation and graph building
    std::shared_ptr<TensorDescriptor> _xDesc;
    std::shared_ptr<TensorDescriptor> _yDesc;
    std::shared_ptr<TensorDescriptor> _scaleDesc;

    // Compute data type for this operation (stored at node level in graph)
    hipdnn_flatbuffers_sdk::data_objects::DataType _computeDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
};

} // namespace hipdnn_backend
