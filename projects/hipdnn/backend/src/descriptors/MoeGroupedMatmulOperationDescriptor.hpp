// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "IGraphOperation.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_grouped_matmul_attributes_generated.h>
#include <unordered_map>

namespace hipdnn_backend
{

class MoeGroupedMatmulOperationDescriptor
    : public HipdnnBackendDescriptorImpl<MoeGroupedMatmulOperationDescriptor>,
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
    const hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulAttributesT& getData() const
    {
        return _data;
    }

    // Access to tensor descriptor references for graph building
    std::shared_ptr<TensorDescriptor> getTokenDesc() const
    {
        return _tokenDesc;
    }
    std::shared_ptr<TensorDescriptor> getWeightDesc() const
    {
        return _weightDesc;
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
    std::shared_ptr<TensorDescriptor> getOutputDesc() const
    {
        return _outputDesc;
    }

    // Get compute data type for the operation (used when building graph nodes)
    hipdnn_flatbuffers_sdk::data_objects::DataType getComputeDataType() const
    {
        return _computeDataType;
    }

    // IGraphOperation interface
    std::vector<std::shared_ptr<TensorDescriptor>> getTensorDescriptors() const override;
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT> buildNode() const override;

    // Creates a finalized MoeGroupedMatmulOperationDescriptor directly from a FlatBuffer NodeT.
    // Casts nodeT.attributes to MoeGroupedMatmulAttributes internally, then directly assigns
    // the data struct, looks up tensor descriptors from the tensor map, and calls finalize().
    static std::shared_ptr<MoeGroupedMatmulOperationDescriptor>
        fromNode(const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
                 const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap);

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulAttributesT _data;

    // Store tensor descriptor references for validation and graph building
    std::shared_ptr<TensorDescriptor> _tokenDesc;
    std::shared_ptr<TensorDescriptor> _weightDesc;
    std::shared_ptr<TensorDescriptor> _firstTokenOffsetDesc;
    std::shared_ptr<TensorDescriptor> _tokenIndexDesc;
    std::shared_ptr<TensorDescriptor> _tokenKsDesc;
    std::shared_ptr<TensorDescriptor> _outputDesc;

    // Compute data type for this operation (stored at node level in graph)
    hipdnn_flatbuffers_sdk::data_objects::DataType _computeDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;

    std::string _name;
};

} // namespace hipdnn_backend
