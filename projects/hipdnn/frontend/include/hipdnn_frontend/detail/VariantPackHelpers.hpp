// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::detail
{

inline Error populateBaseVariantPackDescriptor(ScopedHipdnnBackendDescriptor& variantPackDesc,
                                               std::unordered_map<int64_t, void*>& variantPack,
                                               void* workspace)
{
    std::vector<int64_t> variantPackKeys;
    std::vector<void*> variantPackValues;
    variantPackKeys.reserve(variantPack.size());
    variantPackValues.reserve(variantPack.size());
    for(const auto& [key, value] : variantPack)
    {
        variantPackKeys.push_back(key);
        variantPackValues.push_back(value);
    }

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(variantPackDesc.get(),
                                             HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                             HIPDNN_TYPE_VOID_PTR,
                                             static_cast<int64_t>(variantPackValues.size()),
                                             static_cast<const void*>(variantPackValues.data())),
        "failed to set the variant pack data pointers.");

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(variantPackDesc.get(),
                                             HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                             HIPDNN_TYPE_INT64,
                                             static_cast<int64_t>(variantPackKeys.size()),
                                             variantPackKeys.data()),
        "failed to set the variant pack unique ids.");

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(variantPackDesc.get(),
                                             HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                             HIPDNN_TYPE_VOID_PTR,
                                             1,
                                             static_cast<const void*>(&workspace)),
        "failed to set the variant pack workspace.");

    return {};
}

inline Error
    populateOverrideVariantPackDescriptor(ScopedHipdnnBackendDescriptor& variantPackDesc,
                                          const std::vector<int64_t>& overrideUids,
                                          const std::vector<std::vector<int64_t>>& overrideShapes,
                                          const std::vector<std::vector<int64_t>>& overrideStrides)
{
    std::vector<int64_t> overrideLengths;
    overrideLengths.reserve(overrideUids.size());
    size_t totalElements = 0;
    for(const auto& shape : overrideShapes)
    {
        totalElements += shape.size();
        overrideLengths.push_back(static_cast<int64_t>(shape.size()));
    }

    std::vector<int64_t> flatShapes;
    std::vector<int64_t> flatStrides;
    flatShapes.reserve(totalElements);
    flatStrides.reserve(totalElements);
    for(size_t i = 0; i < overrideUids.size(); ++i)
    {
        flatShapes.insert(flatShapes.end(), overrideShapes[i].begin(), overrideShapes[i].end());
        flatStrides.insert(flatStrides.end(), overrideStrides[i].begin(), overrideStrides[i].end());
    }

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(variantPackDesc.get(),
                                             HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS_EXT,
                                             HIPDNN_TYPE_INT64,
                                             static_cast<int64_t>(overrideUids.size()),
                                             static_cast<const void*>(overrideUids.data())),
        "failed to set HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS_EXT.");

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(variantPackDesc.get(),
                                             HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS_EXT,
                                             HIPDNN_TYPE_INT64,
                                             static_cast<int64_t>(overrideLengths.size()),
                                             static_cast<const void*>(overrideLengths.data())),
        "failed to set HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS_EXT.");

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(variantPackDesc.get(),
                                             HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES_EXT,
                                             HIPDNN_TYPE_INT64,
                                             static_cast<int64_t>(flatShapes.size()),
                                             static_cast<const void*>(flatShapes.data())),
        "failed to set HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES_EXT.");

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(variantPackDesc.get(),
                                             HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES_EXT,
                                             HIPDNN_TYPE_INT64,
                                             static_cast<int64_t>(flatStrides.size()),
                                             static_cast<const void*>(flatStrides.data())),
        "failed to set HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES_EXT.");

    return {};
}

} // namespace hipdnn_frontend::detail
