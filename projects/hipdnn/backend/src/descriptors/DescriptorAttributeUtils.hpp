// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipdnnAttentionImplementation.h"
#include "HipdnnBackendAttributeType.h"
#include "HipdnnDataType.h"
#include "HipdnnDiagonalAlignment.h"
#include "HipdnnException.hpp"
#include "HipdnnNormFwdPhase.h"
#include "HipdnnPointwiseMode.h"
#include "TensorDescriptor.hpp"
#include <cstring>
#include <hipdnn_data_sdk/data_objects/convolution_common_generated.h>
#include <hipdnn_data_sdk/data_objects/data_types_generated.h>
#include <hipdnn_data_sdk/data_objects/norm_common_generated.h>
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/sdpa_attributes_generated.h>
#include <memory>
#include <type_traits>
#include <unordered_map>
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
    static_assert(std::is_trivially_copyable_v<T>, "setScalar requires a trivially copyable type");
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
    static_assert(std::is_trivially_copyable_v<T>, "getScalar requires a trivially copyable type");
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

template <hipdnnBackendAttributeType_t ExpectedType, typename T>
void setOptionalScalar(flatbuffers::Optional<T>& target,
                       hipdnnBackendAttributeType_t attributeType,
                       int64_t elementCount,
                       const void* arrayOfElements,
                       const char* context)
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "setOptionalScalar requires a trivially copyable type");
    checkSetArgs(ExpectedType, attributeType, arrayOfElements, context);
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(context) + ": expected elementCount=1, got "
                       + std::to_string(elementCount));
    T value{};
    std::memcpy(&value, arrayOfElements, sizeof(T));
    target = value;
}

// NormFwdPhase is passed as HIPDNN_TYPE_NORM_FWD_PHASE (hipdnnNormFwdPhase_t).
void setNormFwdPhase(hipdnn_data_sdk::data_objects::NormFwdPhase& target,
                     hipdnnBackendAttributeType_t attributeType,
                     int64_t elementCount,
                     const void* arrayOfElements,
                     const char* errorPrefix);

void getNormFwdPhase(hipdnn_data_sdk::data_objects::NormFwdPhase source,
                     hipdnnBackendAttributeType_t attributeType,
                     int64_t requestedElementCount,
                     int64_t* elementCount,
                     void* arrayOfElements,
                     const char* errorPrefix);

template <hipdnnBackendAttributeType_t ExpectedType, typename T>
void getOptionalScalar(const flatbuffers::Optional<T>& source,
                       hipdnnBackendAttributeType_t attributeType,
                       int64_t requestedCount,
                       int64_t* elementCount,
                       void* arrayOfElements,
                       const char* context)
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "getOptionalScalar requires a trivially copyable type");
    checkGetArgs(ExpectedType, attributeType, context);

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
    std::memcpy(arrayOfElements, &value, sizeof(T));
}

std::shared_ptr<TensorDescriptor>
    findTensorInMap(const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap,
                    int64_t uid,
                    const char* context);

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

void setOptionalTensorDescriptor(std::shared_ptr<TensorDescriptor>& descTarget,
                                 flatbuffers::Optional<int64_t>& uidTarget,
                                 hipdnnBackendAttributeType_t attributeType,
                                 int64_t elementCount,
                                 const void* arrayOfElements,
                                 const char* errorPrefix);

// Like getTensorDescriptor, but descSource may be null (optional tensor).
// Returns elementCount=0 when the tensor was not set.
void getOptionalTensorDescriptor(const std::shared_ptr<TensorDescriptor>& descSource,
                                 hipdnnBackendAttributeType_t attributeType,
                                 int64_t requestedElementCount,
                                 int64_t* elementCount,
                                 void* arrayOfElements,
                                 const char* errorPrefix);

void setTensorDescriptorArray(std::vector<std::shared_ptr<TensorDescriptor>>& descTarget,
                              std::vector<int64_t>& uidTarget,
                              hipdnnBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              const void* arrayOfElements,
                              const char* errorPrefix);

void getTensorDescriptorArray(const std::vector<std::shared_ptr<TensorDescriptor>>& descSource,
                              hipdnnBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements,
                              const char* errorPrefix);

void setDiagonalAlignment(hipdnn_data_sdk::data_objects::DiagonalAlignment& target,
                          hipdnnBackendAttributeType_t attributeType,
                          int64_t elementCount,
                          const void* arrayOfElements,
                          const char* errorPrefix);

void getDiagonalAlignment(hipdnn_data_sdk::data_objects::DiagonalAlignment source,
                          hipdnnBackendAttributeType_t attributeType,
                          int64_t requestedElementCount,
                          int64_t* elementCount,
                          void* arrayOfElements,
                          const char* errorPrefix);

void setAttentionImplementation(hipdnn_data_sdk::data_objects::AttentionImplementation& target,
                                hipdnnBackendAttributeType_t attributeType,
                                int64_t elementCount,
                                const void* arrayOfElements,
                                const char* errorPrefix);

void getAttentionImplementation(hipdnn_data_sdk::data_objects::AttentionImplementation source,
                                hipdnnBackendAttributeType_t attributeType,
                                int64_t requestedElementCount,
                                int64_t* elementCount,
                                void* arrayOfElements,
                                const char* errorPrefix);

} // namespace hipdnn_backend
