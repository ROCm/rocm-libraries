// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include "BackendDescriptor.hpp"
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_backend
{

class VariantDescriptor : public HipdnnBackendDescriptorImpl<VariantDescriptor>
{
private:
    std::vector<const void*> _dataPointers;
    std::vector<int64_t> _uniqueIds;
    void* _workspace = nullptr;

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

    // throws if the variant descriptor is not finalized
    virtual void* getWorkspace() const;
    virtual const std::vector<const void*>& getDataPointers() const;
    virtual const std::vector<int64_t>& getTensorIds() const;

    static hipdnnBackendDescriptorType_t getStaticType();
};
}
