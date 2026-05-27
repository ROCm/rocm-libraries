// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "VariantDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "FlatbufferUtilities.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>

namespace hipdnn_backend
{

bool VariantDescriptor::hasOverrideAttributes() const
{
    return !_overrideUniqueIds.empty() || !_overrideLengths.empty() || !_overrideShapes.empty()
           || !_overrideStrides.empty();
}

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
    if(hasOverrideAttributes())
    {
        THROW_IF_NE(_overrideUniqueIds.size(),
                    _overrideLengths.size(),
                    HIPDNN_STATUS_BAD_PARAM,
                    "VariantDescriptor::finalize() failed: OVERRIDE_UNIQUE_IDS and "
                    "OVERRIDE_LENGTHS must have the same size");

        const std::unordered_set<int64_t> uniqueIdSet(_uniqueIds.begin(), _uniqueIds.end());
        std::unordered_set<int64_t> overrideIdSet;
        overrideIdSet.reserve(_overrideUniqueIds.size());
        uint64_t rankSum = 0;
        constexpr auto MAX_OVERRIDE_LENGTH = std::numeric_limits<uint32_t>::max();

        for(size_t i = 0; i < _overrideUniqueIds.size(); ++i)
        {
            const auto overrideId = _overrideUniqueIds[i];
            THROW_IF_FALSE(overrideIdSet.insert(overrideId).second,
                           HIPDNN_STATUS_BAD_PARAM,
                           "VariantDescriptor::finalize() failed: duplicate "
                           "OVERRIDE_UNIQUE_IDS entry "
                               + std::to_string(overrideId));

            THROW_IF_TRUE(uniqueIdSet.find(overrideId) == uniqueIdSet.end(),
                          HIPDNN_STATUS_BAD_PARAM,
                          "VariantDescriptor::finalize() failed: OVERRIDE_UNIQUE_IDS entry "
                              + std::to_string(overrideId)
                              + " is not present in VARIANT_PACK_UNIQUE_IDS");

            const auto length = _overrideLengths[i];
            THROW_IF_FALSE(length > 0,
                           HIPDNN_STATUS_BAD_PARAM,
                           "VariantDescriptor::finalize() failed: OVERRIDE_LENGTHS entry "
                               + std::to_string(length) + " must be positive");
            THROW_IF_TRUE(static_cast<uint64_t>(length) > MAX_OVERRIDE_LENGTH,
                          HIPDNN_STATUS_BAD_PARAM,
                          "VariantDescriptor::finalize() failed: OVERRIDE_LENGTHS entry "
                              + std::to_string(length) + " exceeds uint32_t max");
            THROW_IF_TRUE(rankSum > std::numeric_limits<uint64_t>::max()
                                        - static_cast<uint64_t>(length),
                          HIPDNN_STATUS_BAD_PARAM,
                          "VariantDescriptor::finalize() failed: OVERRIDE_LENGTHS sum overflow");
            rankSum += static_cast<uint64_t>(length);
        }

        THROW_IF_NE(static_cast<uint64_t>(_overrideShapes.size()),
                    rankSum,
                    HIPDNN_STATUS_BAD_PARAM,
                    "VariantDescriptor::finalize() failed: OVERRIDE_SHAPES total length does "
                    "not match the sum of OVERRIDE_LENGTHS");
        THROW_IF_NE(static_cast<uint64_t>(_overrideStrides.size()),
                    rankSum,
                    HIPDNN_STATUS_BAD_PARAM,
                    "VariantDescriptor::finalize() failed: OVERRIDE_STRIDES total length does "
                    "not match the sum of OVERRIDE_LENGTHS");
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
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "VariantDescriptor::getAttribute(): arrayOfElements is null");
        THROW_IF_FALSE(requestedElementCount >= 0,
                       HIPDNN_STATUS_BAD_PARAM,
                       "VariantDescriptor::getAttribute(): requestedElementCount "
                       "is negative for DATA_POINTERS");
        *elementCount
            = std::min<int64_t>(requestedElementCount, static_cast<int64_t>(_dataPointers.size()));
        for(size_t i = 0; i < static_cast<size_t>(*elementCount); ++i)
        {
            static_cast<void**>(arrayOfElements)[i] = const_cast<void*>(_dataPointers[i]);
        }
        break;

    case HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS:
        getScalarVector<int64_t>(_uniqueIds,
                                 HIPDNN_TYPE_INT64,
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
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "VariantDescriptor::getAttribute(): arrayOfElements is null");
        if(elementCount != nullptr)
        {
            *elementCount = 1;
        }

        *static_cast<void**>(arrayOfElements) = _workspace;
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS_EXT:
        getScalarVector<int64_t>(_overrideUniqueIds,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "VariantDescriptor::getAttribute(OVERRIDE_UNIQUE_IDS)");
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES_EXT:
        getScalarVector<int64_t>(_overrideShapes,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "VariantDescriptor::getAttribute(OVERRIDE_SHAPES)");
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES_EXT:
        getScalarVector<int64_t>(_overrideStrides,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "VariantDescriptor::getAttribute(OVERRIDE_STRIDES)");
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS_EXT:
        getScalarVector<int64_t>(_overrideLengths,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 "VariantDescriptor::getAttribute(OVERRIDE_LENGTHS)");
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
        setScalarVector<int64_t>(_uniqueIds,
                                 HIPDNN_TYPE_INT64,
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

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS_EXT:
        setScalarVector<int64_t>(_overrideUniqueIds,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "VariantDescriptor::setAttribute(OVERRIDE_UNIQUE_IDS)");
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES_EXT:
        setScalarVector<int64_t>(_overrideShapes,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "VariantDescriptor::setAttribute(OVERRIDE_SHAPES)");
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES_EXT:
        setScalarVector<int64_t>(_overrideStrides,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "VariantDescriptor::setAttribute(OVERRIDE_STRIDES)");
        break;

    case HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS_EXT:
        setScalarVector<int64_t>(_overrideLengths,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 "VariantDescriptor::setAttribute(OVERRIDE_LENGTHS)");
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
    if(hasOverrideAttributes())
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
