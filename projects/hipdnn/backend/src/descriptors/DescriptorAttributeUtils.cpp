// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorAttributeUtils.hpp"

#include <algorithm>
#include <cstring>

namespace hipdnn_backend
{

void checkSetArgs(hipdnnBackendAttributeType_t expectedType,
                  hipdnnBackendAttributeType_t attributeType,
                  const void* arrayOfElements,
                  const char* errorPrefix)
{
    THROW_IF_NULL(errorPrefix, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "errorPrefix is null");
    THROW_IF_FALSE(attributeType == expectedType,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": attributeType mismatch");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  std::string(errorPrefix) + ": arrayOfElements is null");
}

void checkGetArgs(hipdnnBackendAttributeType_t expectedType,
                  hipdnnBackendAttributeType_t attributeType,
                  const void* arrayOfElements,
                  const char* errorPrefix)
{
    THROW_IF_NULL(errorPrefix, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "errorPrefix is null");
    THROW_IF_FALSE(attributeType == expectedType,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": attributeType mismatch");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  std::string(errorPrefix) + ": arrayOfElements is null");
}

void setInt64Vector(std::vector<int64_t>& target,
                    hipdnnBackendAttributeType_t attributeType,
                    int64_t elementCount,
                    const void* arrayOfElements,
                    const char* errorPrefix)
{
    checkSetArgs(HIPDNN_TYPE_INT64, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_LT(elementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                std::string(errorPrefix) + ": elementCount is negative");
    auto ptr = static_cast<const int64_t*>(arrayOfElements);
    target.assign(ptr, ptr + elementCount);
}

void getInt64Vector(const std::vector<int64_t>& source,
                    hipdnnBackendAttributeType_t attributeType,
                    int64_t requestedElementCount,
                    int64_t* elementCount,
                    void* arrayOfElements,
                    const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_INT64, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_LT(requestedElementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                std::string(errorPrefix) + ": requestedElementCount is negative");

    auto copyCount = std::min<size_t>(static_cast<size_t>(requestedElementCount), source.size());
    if(elementCount != nullptr)
    {
        *elementCount = static_cast<int64_t>(copyCount);
    }
    std::memcpy(arrayOfElements, source.data(), copyCount * sizeof(int64_t));
}

} // namespace hipdnn_backend
