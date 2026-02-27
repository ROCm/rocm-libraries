// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

namespace hipdnn_backend
{

class TensorDescriptor : public HipdnnBackendDescriptorImpl<TensorDescriptor>
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
    const hipdnn_data_sdk::data_objects::TensorAttributesT& getData() const
    {
        return _data;
    }

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    void setName(hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElements);
    void getName(hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements) const;

    void setTensorValue(hipdnnBackendAttributeType_t attributeType,
                        int64_t elementCount,
                        const void* arrayOfElements);
    void getTensorValue(hipdnnBackendAttributeType_t attributeType,
                        int64_t requestedElementCount,
                        int64_t* elementCount,
                        void* arrayOfElements) const;

    hipdnn_data_sdk::data_objects::TensorAttributesT _data;
};

} // namespace hipdnn_backend
