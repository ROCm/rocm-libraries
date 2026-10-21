// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <hipdnn_compatibility/cudnn/detail/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/v8/error.h>
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
{

class VariantPackRecord final : public RecordOf<CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR>
{
public:
    explicit VariantPackRecord(::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor variantPack)
        : _variantPack(std::move(variantPack))
    {
    }

    hipdnnBackendDescriptor_t hipdnnVariantPack() const noexcept
    {
        return _variantPack.get();
    }

private:
    ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor _variantPack;
};

// hipDNN rejects an empty array with a null-pointer error that does not name
// the caller's array.
inline void requireVariantPackArray(const void* data, std::int64_t count, std::string_view label)
{
    if(count < 0)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(label) + ": element count " + std::to_string(count)
                            + " is negative");
    }
    if(count == 0)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM, std::string(label) + " is required (non-empty)");
    }
    if(data == nullptr)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM, std::string(label) + ": array is null");
    }
}

// Names the caller's arrays in variant-pack error messages.
struct VariantPackLabels
{
    std::string_view dataPointers;
    std::string_view uniqueIds;
};

// Builds and finalizes the hipDNN VARIANT_PACK once, so execute is a single
// backend call. populateBaseVariantPackDescriptor is not reused: it takes a UID
// map, which would silently merge duplicate UIDs and copy both arrays again.
inline std::shared_ptr<const VariantPackRecord> buildVariantPack(void* const* dataPointers,
                                                                 std::int64_t pointerCount,
                                                                 const std::int64_t* uniqueIds,
                                                                 std::int64_t uidCount,
                                                                 void* workspace,
                                                                 const VariantPackLabels& labels)
{
    using ::hipdnn_frontend::detail::hipdnnBackend;

    requireVariantPackArray(
        static_cast<const void*>(dataPointers), pointerCount, labels.dataPointers);
    requireVariantPackArray(uniqueIds, uidCount, labels.uniqueIds);
    if(pointerCount != uidCount)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(labels.dataPointers) + " has " + std::to_string(pointerCount)
                            + " elements but " + std::string(labels.uniqueIds) + " has "
                            + std::to_string(uidCount));
    }

    ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor variantPack = createHipdnnDescriptor(
        HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, "creating the hipDNN variant pack");
    checkHipdnn(hipdnnBackend()->backendSetAttribute(variantPack.get(),
                                                     HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                     HIPDNN_TYPE_VOID_PTR,
                                                     pointerCount,
                                                     static_cast<const void*>(dataPointers)),
                "setting the hipDNN variant pack data pointers");
    checkHipdnn(hipdnnBackend()->backendSetAttribute(variantPack.get(),
                                                     HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                                     HIPDNN_TYPE_INT64,
                                                     uidCount,
                                                     uniqueIds),
                "setting the hipDNN variant pack unique IDs");
    checkHipdnn(hipdnnBackend()->backendSetAttribute(variantPack.get(),
                                                     HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                                     HIPDNN_TYPE_VOID_PTR,
                                                     1,
                                                     static_cast<const void*>(&workspace)),
                "setting the hipDNN variant pack workspace");
    checkHipdnn(hipdnnBackend()->backendFinalize(variantPack.get()),
                "finalizing the hipDNN variant pack");
    return std::make_shared<const VariantPackRecord>(std::move(variantPack));
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
