// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "IGraphOperation.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_data_sdk/data_objects/convolution_fwd_attributes_generated.h>

namespace hipdnn_backend
{

class ConvolutionFwdOperationDescriptor
    : public HipdnnBackendDescriptorImpl<ConvolutionFwdOperationDescriptor>,
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
    const hipdnn_data_sdk::data_objects::ConvolutionFwdAttributesT& getData() const
    {
        return _data;
    }

    // Access to tensor descriptor references for graph building
    std::shared_ptr<TensorDescriptor> getXDesc() const
    {
        return _xDesc;
    }
    std::shared_ptr<TensorDescriptor> getWDesc() const
    {
        return _wDesc;
    }
    std::shared_ptr<TensorDescriptor> getYDesc() const
    {
        return _yDesc;
    }

    // Get compute data type for the operation (used when building graph nodes)
    hipdnn_data_sdk::data_objects::DataType getComputeDataType() const
    {
        return _computeDataType;
    }

    // IGraphOperation interface
    std::vector<std::shared_ptr<TensorDescriptor>> getTensorDescriptors() const override;
    std::unique_ptr<hipdnn_data_sdk::data_objects::NodeT> buildNode() const override;

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    hipdnn_data_sdk::data_objects::ConvolutionFwdAttributesT _data;

    // Store tensor descriptor references for validation and graph building
    std::shared_ptr<TensorDescriptor> _xDesc;
    std::shared_ptr<TensorDescriptor> _wDesc;
    std::shared_ptr<TensorDescriptor> _yDesc;

    // Compute data type for this operation (stored at node level in graph)
    hipdnn_data_sdk::data_objects::DataType _computeDataType
        = hipdnn_data_sdk::data_objects::DataType::UNSET;
};

} // namespace hipdnn_backend
