// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "IGraphOperation.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_data_sdk/data_objects/matmul_attributes_generated.h>

namespace hipdnn_backend
{

class MatmulOperationDescriptor : public HipdnnBackendDescriptorImpl<MatmulOperationDescriptor>,
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
    const hipdnn_data_sdk::data_objects::MatmulAttributesT& getData() const
    {
        return _data;
    }

    // Access to tensor descriptor references for graph building
    std::shared_ptr<TensorDescriptor> getADesc() const
    {
        return _aDesc;
    }
    std::shared_ptr<TensorDescriptor> getBDesc() const
    {
        return _bDesc;
    }
    std::shared_ptr<TensorDescriptor> getCDesc() const
    {
        return _cDesc;
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
    hipdnn_data_sdk::data_objects::MatmulAttributesT _data;

    // Store tensor descriptor references for validation and graph building
    std::shared_ptr<TensorDescriptor> _aDesc;
    std::shared_ptr<TensorDescriptor> _bDesc;
    std::shared_ptr<TensorDescriptor> _cDesc;

    // Compute data type for this operation (stored at node level in graph)
    hipdnn_data_sdk::data_objects::DataType _computeDataType
        = hipdnn_data_sdk::data_objects::DataType::UNSET;
};

} // namespace hipdnn_backend
