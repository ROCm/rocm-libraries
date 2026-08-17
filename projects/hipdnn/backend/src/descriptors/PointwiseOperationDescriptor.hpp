// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "IGraphOperation.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <unordered_map>

namespace hipdnn_backend
{

class PointwiseOperationDescriptor
    : public HipdnnBackendDescriptorImpl<PointwiseOperationDescriptor>,
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
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributesT& getData() const
    {
        return _data;
    }

    // Access to tensor descriptor references for graph building
    std::shared_ptr<TensorDescriptor> getIn0Desc() const
    {
        return _in0Desc;
    }
    std::shared_ptr<TensorDescriptor> getOut0Desc() const
    {
        return _out0Desc;
    }
    std::shared_ptr<TensorDescriptor> getIn1Desc() const
    {
        return _in1Desc;
    }
    std::shared_ptr<TensorDescriptor> getIn2Desc() const
    {
        return _in2Desc;
    }

    // Get compute data type for the operation (used when building graph nodes)
    hipdnn_flatbuffers_sdk::data_objects::DataType getComputeDataType() const
    {
        return _computeDataType;
    }

    // IGraphOperation interface
    std::vector<std::shared_ptr<TensorDescriptor>> getTensorDescriptors() const override;
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT> buildNode() const override;

    static std::shared_ptr<PointwiseOperationDescriptor>
        fromNode(const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
                 const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap);

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributesT _data;
    std::string _name;

    // Store tensor descriptor references for validation and graph building
    std::shared_ptr<TensorDescriptor> _in0Desc;
    std::shared_ptr<TensorDescriptor> _out0Desc;
    std::shared_ptr<TensorDescriptor> _in1Desc;
    std::shared_ptr<TensorDescriptor> _in2Desc;

    // Compute data type for this operation (stored at node level in graph)
    hipdnn_flatbuffers_sdk::data_objects::DataType _computeDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
};

} // namespace hipdnn_backend
