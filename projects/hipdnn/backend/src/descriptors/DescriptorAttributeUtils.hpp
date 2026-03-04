// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipdnnBackendAttributeType.h"
#include "HipdnnDataType.h"
#include "HipdnnException.hpp"
#include "HipdnnPointwiseMode.h"
#include "TensorDescriptor.hpp"
#include <cstring>
#include <hipdnn_data_sdk/data_objects/convolution_common_generated.h>
#include <hipdnn_data_sdk/data_objects/data_types_generated.h>
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <memory>
#include <vector>

namespace hipdnn_backend
{

void checkSetArgs(hipdnnBackendAttributeType_t expectedType,
                  hipdnnBackendAttributeType_t attributeType,
                  const void* arrayOfElements,
                  const char* errorPrefix);

void checkGetArgs(hipdnnBackendAttributeType_t expectedType,
                  hipdnnBackendAttributeType_t attributeType,
                  const char* errorPrefix);

void setInt64Vector(std::vector<int64_t>& target,
                    hipdnnBackendAttributeType_t attributeType,
                    int64_t elementCount,
                    const void* arrayOfElements,
                    const char* errorPrefix);

void getInt64Vector(const std::vector<int64_t>& source,
                    hipdnnBackendAttributeType_t attributeType,
                    int64_t requestedElementCount,
                    int64_t* elementCount,
                    void* arrayOfElements,
                    const char* errorPrefix);

template <typename T>
void setScalar(T& target,
               hipdnnBackendAttributeType_t expectedType,
               hipdnnBackendAttributeType_t attributeType,
               int64_t elementCount,
               const void* arrayOfElements,
               const char* errorPrefix)
{
    checkSetArgs(expectedType, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": elementCount is not 1");
    std::memcpy(&target, arrayOfElements, sizeof(T));
}

template <typename T>
void getScalar(const T& source,
               hipdnnBackendAttributeType_t expectedType,
               hipdnnBackendAttributeType_t attributeType,
               int64_t requestedElementCount,
               int64_t* elementCount,
               void* arrayOfElements,
               const char* errorPrefix)
{
    checkGetArgs(expectedType, attributeType, errorPrefix);

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      std::string(errorPrefix) + ": elementCount is null");
        *elementCount = 1;
        return;
    }

    THROW_IF_FALSE(requestedElementCount >= 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": requestedElementCount < 1");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
    std::memcpy(arrayOfElements, &source, sizeof(T));
}

void setDataType(hipdnn_data_sdk::data_objects::DataType& target,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElements,
                 const char* errorPrefix);

void getDataType(hipdnn_data_sdk::data_objects::DataType source,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements,
                 const char* errorPrefix);

void setConvMode(hipdnn_data_sdk::data_objects::ConvMode& target,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElements,
                 const char* errorPrefix);

void getConvMode(hipdnn_data_sdk::data_objects::ConvMode source,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements,
                 const char* errorPrefix);

void setPointwiseMode(hipdnn_data_sdk::data_objects::PointwiseMode& target,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements,
                      const char* errorPrefix);

void getPointwiseMode(hipdnn_data_sdk::data_objects::PointwiseMode source,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements,
                      const char* errorPrefix);

inline void setOptionalFloat(flatbuffers::Optional<float>& target,
                             hipdnnBackendAttributeType_t attributeType,
                             int64_t elementCount,
                             const void* arrayOfElements,
                             const char* context)
{
    checkSetArgs(HIPDNN_TYPE_FLOAT, attributeType, arrayOfElements, context);
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(context) + ": expected elementCount=1, got "
                       + std::to_string(elementCount));
    float value = 0.0F;
    std::memcpy(&value, arrayOfElements, sizeof(float));
    target = value;
}

inline void getOptionalFloat(const flatbuffers::Optional<float>& source,
                             hipdnnBackendAttributeType_t attributeType,
                             int64_t requestedCount,
                             int64_t* elementCount,
                             void* arrayOfElements,
                             const char* context)
{
    checkGetArgs(HIPDNN_TYPE_FLOAT, attributeType, context);

    if(!source.has_value())
    {
        if(elementCount != nullptr)
        {
            *elementCount = 0;
        }
        return;
    }

    if(arrayOfElements == nullptr || requestedCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      std::string(context) + ": elementCount is null");
        *elementCount = 1;
        return;
    }

    THROW_IF_FALSE(requestedCount >= 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(context) + ": requestedElementCount < 1");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
    auto value = source.value();
    std::memcpy(arrayOfElements, &value, sizeof(float));
}

inline void setOptionalInt64(flatbuffers::Optional<int64_t>& target,
                             hipdnnBackendAttributeType_t attributeType,
                             int64_t elementCount,
                             const void* arrayOfElements,
                             const char* context)
{
    checkSetArgs(HIPDNN_TYPE_INT64, attributeType, arrayOfElements, context);
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(context) + ": expected elementCount=1, got "
                       + std::to_string(elementCount));
    int64_t value = 0;
    std::memcpy(&value, arrayOfElements, sizeof(int64_t));
    target = value;
}

inline void getOptionalInt64(const flatbuffers::Optional<int64_t>& source,
                             hipdnnBackendAttributeType_t attributeType,
                             int64_t requestedCount,
                             int64_t* elementCount,
                             void* arrayOfElements,
                             const char* context)
{
    checkGetArgs(HIPDNN_TYPE_INT64, attributeType, context);

    if(!source.has_value())
    {
        if(elementCount != nullptr)
        {
            *elementCount = 0;
        }
        return;
    }

    if(arrayOfElements == nullptr || requestedCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      std::string(context) + ": elementCount is null");
        *elementCount = 1;
        return;
    }

    THROW_IF_FALSE(requestedCount >= 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(context) + ": requestedElementCount < 1");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
    auto value = source.value();
    std::memcpy(arrayOfElements, &value, sizeof(int64_t));
}

void setTensorDescriptor(std::shared_ptr<TensorDescriptor>& descTarget,
                         int64_t& uidTarget,
                         hipdnnBackendAttributeType_t attributeType,
                         int64_t elementCount,
                         const void* arrayOfElements,
                         const char* errorPrefix);

void getTensorDescriptor(const std::shared_ptr<TensorDescriptor>& descSource,
                         hipdnnBackendAttributeType_t attributeType,
                         int64_t requestedElementCount,
                         int64_t* elementCount,
                         void* arrayOfElements,
                         const char* errorPrefix);

} // namespace hipdnn_backend
