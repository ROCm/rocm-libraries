// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include "BackendDescriptor.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_backend
{

class VariantDescriptor : public HipdnnBackendDescriptorImpl<VariantDescriptor>
{
private:
    std::vector<const void*> _dataPointers;
    std::vector<int64_t> _uniqueIds;
    void* _workspace = nullptr;

    // Per-execute override storage. Shape and stride vectors are flat-concatenated
    // in UID order and sliced at dispatch using _overrideLengths.
    std::vector<int64_t> _overrideUniqueIds;
    std::vector<int64_t> _overrideShapes;
    std::vector<int64_t> _overrideStrides;
    std::vector<int64_t> _overrideLengths;

    bool hasOverrideAttributes() const;

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

    /// Per-execute override-tensor selectors. Empty when no overrides were
    /// supplied. Throws if the descriptor is not finalized.
    virtual const std::vector<int64_t>& getOverrideUniqueIds() const;

    /// Flat concatenation of override shape vectors in getOverrideUniqueIds()
    /// order. Sliced via getOverrideLengths(). Throws if the descriptor is
    /// not finalized.
    virtual const std::vector<int64_t>& getOverrideShapes() const;

    /// Flat concatenation of override stride vectors in getOverrideUniqueIds()
    /// order. Sliced via getOverrideLengths(). Throws if the descriptor is
    /// not finalized.
    virtual const std::vector<int64_t>& getOverrideStrides() const;

    /// Per-UID rank of the override shape/stride vectors. Stored as int64_t
    /// in the variant pack and narrowed to uint32_t at the plugin SDK boundary.
    /// Throws if not finalized.
    virtual const std::vector<int64_t>& getOverrideLengths() const;

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;
};
}
