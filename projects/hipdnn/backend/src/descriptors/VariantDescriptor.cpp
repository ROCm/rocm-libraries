// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "VariantDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "FlatbufferUtilities.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"

#include <numeric>
#include <unordered_set>

namespace hipdnn_backend
{

void VariantDescriptor::finalize()
{
    THROW_IF_NE(_dataPointers.size(),
                _uniqueIds.size(),
                HIPDNN_STATUS_BAD_PARAM,
                "Data pointers and unique ids don't match");
    THROW_IF_TRUE(
        _dataPointers.empty(), HIPDNN_STATUS_BAD_PARAM, "Data pointers and unique ids are empty");

    // Validate the override-tensor invariants at finalize time so malformed
    // variant packs are rejected before dispatch. The dispatch path keeps its
    // own checks as defense-in-depth. Skip entirely when no overrides were
    // supplied (legacy variant packs).
    const bool hasAnyOverride = !_overrideUniqueIds.empty() || !_overrideLengths.empty()
                                || !_overrideShapes.empty() || !_overrideStrides.empty();
    if(hasAnyOverride)
    {
        THROW_IF_NE(_overrideUniqueIds.size(),
                    _overrideLengths.size(),
                    HIPDNN_STATUS_BAD_PARAM,
                    "VariantDescriptor::finalize() failed: OVERRIDE_UNIQUE_IDS and "
                    "OVERRIDE_LENGTHS must have the same size");

        // Sum of per-tensor ranks must equal the flat shape and stride lengths.
        const auto rankSum = std::accumulate(
            _overrideLengths.begin(), _overrideLengths.end(), static_cast<int64_t>(0));
        THROW_IF_NE(static_cast<int64_t>(_overrideShapes.size()),
                    rankSum,
                    HIPDNN_STATUS_BAD_PARAM,
                    "VariantDescriptor::finalize() failed: OVERRIDE_SHAPES total length does "
                    "not match the sum of OVERRIDE_LENGTHS");
        THROW_IF_NE(static_cast<int64_t>(_overrideStrides.size()),
                    rankSum,
                    HIPDNN_STATUS_BAD_PARAM,
                    "VariantDescriptor::finalize() failed: OVERRIDE_STRIDES total length does "
                    "not match the sum of OVERRIDE_LENGTHS");

        // Each override unique-id must refer to a tensor that actually appears
        // in the variant pack's UID list.
        const std::unordered_set<int64_t> uniqueIdSet(_uniqueIds.begin(), _uniqueIds.end());
        for(const auto overrideId : _overrideUniqueIds)
        {
            THROW_IF_TRUE(uniqueIdSet.find(overrideId) == uniqueIdSet.end(),
                          HIPDNN_STATUS_BAD_PARAM,
                          "VariantDescriptor::finalize() failed: OVERRIDE_UNIQUE_IDS entry "
                              + std::to_string(overrideId)
                              + " is not present in VARIANT_PACK_UNIQUE_IDS");
        }
    }

    HipdnnBackendDescriptorImpl<VariantDescriptor>::finalize();
}

void VariantDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                     hipdnnBackendAttributeType_t attributeType,
                                     int64_t requestedElementCount,
                                     int64_t* elementCount,
                                     void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "VariantDescriptor::getAttribute() failed: Not finalized.");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "VariantDescriptor::getAttribute(): arrayOfElements is null");

    switch(attributeName)
    {
    case HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_VOID_PTR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "VariantDescriptor::getAttribute(): attributeType is not "
                       "HIPDNN_TYPE_VOID_PTR for DATA_POINTERS");
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "VariantDescriptor::getAttribute(): elementCount is null");
        *elementCount
            = std::min<int64_t>(requestedElementCount, static_cast<int64_t>(_dataPointers.size()));
        for(size_t i = 0; i < static_cast<size_t>(*elementCount); ++i)
        {
            static_cast<void**>(arrayOfElements)[i] = const_cast<void*>(_dataPointers[i]);
        }
        break;

    case HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS:
        getInt64Vector(_uniqueIds,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "VariantDescriptor::getAttribute()");
        break;

    case HIPDNN_ATTR_VARIANT_PACK_WORKSPACE:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_VOID_PTR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "VariantDescriptor::getAttribute(): attributeType is not "
                       "HIPDNN_TYPE_VOID_PTR for WORKSPACE");
        THROW_IF_FALSE(requestedElementCount == 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "VariantDescriptor::getAttribute(): requestedElementCount "
                       "is not 1 for WORKSPACE");
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }

        *static_cast<void**>(arrayOfElements) = _workspace;
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS:
        getInt64Vector(_overrideUniqueIds,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "VariantDescriptor::getAttribute()");
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES:
        getInt64Vector(_overrideShapes,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "VariantDescriptor::getAttribute()");
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES:
        getInt64Vector(_overrideStrides,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "VariantDescriptor::getAttribute()");
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS:
        // D1 contract: stored as int64_t in the variant pack. The narrowing to
        // uint32_t happens at the SDK dispatch boundary in Stream B, NOT here.
        getInt64Vector(_overrideLengths,
                       attributeType,
                       requestedElementCount,
                       elementCount,
                       arrayOfElements,
                       "VariantDescriptor::getAttribute()");
        break;

    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "VariantDescriptor::getAttribute: attributeName not supported");
    }
}

void VariantDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                     hipdnnBackendAttributeType_t attributeType,
                                     int64_t elementCount,
                                     const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "VariantDescriptor::setAttribute() failed: Already finalized.");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "VariantDescriptor::setAttribute(): arrayOfElements is null");

    switch(attributeName)
    {
    case HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_VOID_PTR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "VariantDescriptor::setAttribute(): attributeType is not "
                       "HIPDNN_TYPE_VOID_PTR for DATA_POINTERS");
        _dataPointers.assign(static_cast<const void* const*>(arrayOfElements),
                             static_cast<const void* const*>(arrayOfElements) + elementCount);
        break;

    case HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS:
        setInt64Vector(_uniqueIds,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "VariantDescriptor::setAttribute()");
        break;

    case HIPDNN_ATTR_VARIANT_PACK_WORKSPACE:
        THROW_IF_FALSE(attributeType == HIPDNN_TYPE_VOID_PTR,
                       HIPDNN_STATUS_BAD_PARAM,
                       "VariantDescriptor::setAttribute(): attributeType is not "
                       "HIPDNN_TYPE_VOID_PTR for WORKSPACE");
        THROW_IF_FALSE(elementCount == 1,
                       HIPDNN_STATUS_BAD_PARAM,
                       "VariantDescriptor::setAttribute(): elementCount is not 1 for WORKSPACE");

        _workspace = *static_cast<void* const*>(arrayOfElements);
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS:
        setInt64Vector(_overrideUniqueIds,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "VariantDescriptor::setAttribute()");
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES:
        setInt64Vector(_overrideShapes,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "VariantDescriptor::setAttribute()");
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES:
        setInt64Vector(_overrideStrides,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "VariantDescriptor::setAttribute()");
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS:
        // D1 contract: stored as int64_t in the variant pack. The narrowing to
        // uint32_t happens at the SDK dispatch boundary in Stream B, NOT here.
        setInt64Vector(_overrideLengths,
                       attributeType,
                       elementCount,
                       arrayOfElements,
                       "VariantDescriptor::setAttribute()");
        break;

    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "VariantDescriptor::setAttribute: attributeName not supported");
    }
}

void* VariantDescriptor::getWorkspace() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "VariantDescriptor::getWorkspace() failed: Not finalized.");
    return _workspace;
}

const std::vector<const void*>& VariantDescriptor::getDataPointers() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "VariantDescriptor::getDataPointers() failed: Not finalized.");
    return _dataPointers;
}

const std::vector<int64_t>& VariantDescriptor::getTensorIds() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "VariantDescriptor::getTensorIds() failed: Not finalized.");
    return _uniqueIds;
}

const std::vector<int64_t>& VariantDescriptor::getOverrideUniqueIds() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "VariantDescriptor::getOverrideUniqueIds() failed: Not finalized.");
    return _overrideUniqueIds;
}

const std::vector<int64_t>& VariantDescriptor::getOverrideShapes() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "VariantDescriptor::getOverrideShapes() failed: Not finalized.");
    return _overrideShapes;
}

const std::vector<int64_t>& VariantDescriptor::getOverrideStrides() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "VariantDescriptor::getOverrideStrides() failed: Not finalized.");
    return _overrideStrides;
}

const std::vector<int64_t>& VariantDescriptor::getOverrideLengths() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "VariantDescriptor::getOverrideLengths() failed: Not finalized.");
    return _overrideLengths;
}

hipdnnBackendDescriptorType_t VariantDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR;
}

std::string VariantDescriptor::toString() const
{
    std::string str = "VariantDescriptor: {numDataPointers=" + std::to_string(_dataPointers.size());
    str += ", numUniqueIds=" + std::to_string(_uniqueIds.size());
    str += _workspace != nullptr ? ", workspace=" + fmt::format("{:p}", _workspace)
                                 : ", workspace=null";
    // Only emit override field counts when at least one is non-empty so
    // legacy variant-pack log lines stay unchanged.
    if(!_overrideUniqueIds.empty() || !_overrideLengths.empty() || !_overrideShapes.empty()
       || !_overrideStrides.empty())
    {
        str += ", overrideUniqueIds=" + std::to_string(_overrideUniqueIds.size());
        str += ", overrideLengths=" + std::to_string(_overrideLengths.size());
        str += ", overrideShapes=" + std::to_string(_overrideShapes.size());
        str += ", overrideStrides=" + std::to_string(_overrideStrides.size());
    }
    str += "}";
    return str;
}

}
