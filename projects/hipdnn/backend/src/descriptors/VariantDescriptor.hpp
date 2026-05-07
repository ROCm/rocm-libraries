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

    // Per-execute override storage (RFC 0008). These four vectors are populated
    // via the HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_* attributes and consumed at dispatch
    // time. They are intentionally NOT serialized: a variant pack is ephemeral
    // per-execute state, never written to a flatbuffer (RFC §11).
    //
    // Indexing model:
    //   - _overrideUniqueIds[i] selects which graph tensor entry i refers to.
    //   - _overrideLengths[i] is the rank of that tensor's override vectors.
    //   - _overrideShapes / _overrideStrides are the concatenation of each tensor's
    //     shape / stride vector in UID order, sliced at dispatch via prefix sum of
    //     _overrideLengths.
    //
    // _overrideLengths is stored as int64_t (D1 contract) so the variant pack stays
    // type-uniform with the C-API surface; narrowing to uint32_t happens at the SDK
    // dispatch boundary in Stream B.
    std::vector<int64_t> _overrideUniqueIds;
    std::vector<int64_t> _overrideShapes;
    std::vector<int64_t> _overrideStrides;
    std::vector<int64_t> _overrideLengths;

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
    /// in the variant pack (D1 contract); narrowed to uint32_t at the SDK
    /// dispatch boundary. Throws if not finalized.
    virtual const std::vector<int64_t>& getOverrideLengths() const;

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;
};
}
