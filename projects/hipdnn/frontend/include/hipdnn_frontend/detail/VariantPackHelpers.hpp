// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

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

} // namespace hipdnn_frontend::detail
