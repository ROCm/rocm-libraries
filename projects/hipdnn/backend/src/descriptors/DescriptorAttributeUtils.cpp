// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorAttributeUtils.hpp"
#include "BackendDescriptor.hpp"
#include "DataTypeConversion.hpp"

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
                  const char* errorPrefix)
{
    THROW_IF_NULL(errorPrefix, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "errorPrefix is null");
    THROW_IF_FALSE(attributeType == expectedType,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": attributeType mismatch");
}

void setInt64Vector(std::vector<int64_t>& target,
                    hipdnnBackendAttributeType_t attributeType,
                    int64_t elementCount,
                    const void* arrayOfElements,
                    const char* errorPrefix)
{
    checkSetArgs(HIPDNN_TYPE_INT64, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_FALSE(elementCount > 0,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": elementCount must be positive");
    target.resize(static_cast<size_t>(elementCount));
    std::memcpy(
        target.data(), arrayOfElements, static_cast<size_t>(elementCount) * sizeof(int64_t));
}

void getInt64Vector(const std::vector<int64_t>& source,
                    hipdnnBackendAttributeType_t attributeType,
                    int64_t requestedElementCount,
                    int64_t* elementCount,
                    void* arrayOfElements,
                    const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_INT64, attributeType, errorPrefix);

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      std::string(errorPrefix) + ": elementCount is null");
        *elementCount = static_cast<int64_t>(source.size());
        return;
    }

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

void setDataType(hipdnn_data_sdk::data_objects::DataType& target,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElements,
                 const char* errorPrefix)
{
    checkSetArgs(HIPDNN_TYPE_DATA_TYPE, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": elementCount is not 1");
    hipdnnDataType_t tmp;
    std::memcpy(&tmp, arrayOfElements, sizeof(tmp));
    target = toSdkDataType(tmp);
}

void getDataType(hipdnn_data_sdk::data_objects::DataType source,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements,
                 const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_DATA_TYPE, attributeType, errorPrefix);

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
    auto tmp = fromSdkDataType(source);
    std::memcpy(arrayOfElements, &tmp, sizeof(tmp));
    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
}

void setConvMode(hipdnn_data_sdk::data_objects::ConvMode& target,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElements,
                 const char* errorPrefix)
{
    checkSetArgs(HIPDNN_TYPE_CONVOLUTION_MODE, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": elementCount is not 1");
    hipdnnConvolutionMode_t tmp;
    std::memcpy(&tmp, arrayOfElements, sizeof(tmp));
    target = toSdkConvMode(tmp);
}

void getConvMode(hipdnn_data_sdk::data_objects::ConvMode source,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements,
                 const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_CONVOLUTION_MODE, attributeType, errorPrefix);

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
    auto tmp = fromSdkConvMode(source);
    std::memcpy(arrayOfElements, &tmp, sizeof(tmp));
    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
}

void setPointwiseMode(hipdnn_data_sdk::data_objects::PointwiseMode& target,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements,
                      const char* errorPrefix)
{
    checkSetArgs(HIPDNN_TYPE_POINTWISE_MODE, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": elementCount is not 1");
    hipdnnPointwiseMode_t tmp;
    std::memcpy(&tmp, arrayOfElements, sizeof(tmp));
    target = toSdkPointwiseMode(tmp);
}

void getPointwiseMode(hipdnn_data_sdk::data_objects::PointwiseMode source,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements,
                      const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_POINTWISE_MODE, attributeType, errorPrefix);

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
    auto tmp = fromSdkPointwiseMode(source);
    std::memcpy(arrayOfElements, &tmp, sizeof(tmp));
    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
}

void setNormFwdPhase(hipdnn_data_sdk::data_objects::NormFwdPhase& target,
                     hipdnnBackendAttributeType_t attributeType,
                     int64_t elementCount,
                     const void* arrayOfElements,
                     const char* errorPrefix)
{
    checkSetArgs(HIPDNN_TYPE_NORM_FWD_PHASE, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": elementCount is not 1");
    hipdnnNormFwdPhase_t tmp;
    std::memcpy(&tmp, arrayOfElements, sizeof(tmp));
    target = toSdkNormFwdPhase(tmp);
}

void getNormFwdPhase(hipdnn_data_sdk::data_objects::NormFwdPhase source,
                     hipdnnBackendAttributeType_t attributeType,
                     int64_t requestedElementCount,
                     int64_t* elementCount,
                     void* arrayOfElements,
                     const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_NORM_FWD_PHASE, attributeType, errorPrefix);

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
    auto tmp = fromSdkNormFwdPhase(source);
    std::memcpy(arrayOfElements, &tmp, sizeof(tmp));
}

void setOptionalFloat(flatbuffers::Optional<float>& target,
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

void getOptionalFloat(const flatbuffers::Optional<float>& source,
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

void setOptionalInt64(flatbuffers::Optional<int64_t>& target,
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

void getOptionalInt64(const flatbuffers::Optional<int64_t>& source,
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

std::shared_ptr<TensorDescriptor>
    findTensorInMap(const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap,
                    int64_t uid,
                    const char* context)
{
    auto it = tensorMap.find(uid);
    THROW_IF_TRUE(it == tensorMap.end(),
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  std::string(context) + ": tensor UID " + std::to_string(uid)
                      + " not found in tensor map");
    return it->second;
}

void setTensorDescriptor(std::shared_ptr<TensorDescriptor>& descTarget,
                         int64_t& uidTarget,
                         hipdnnBackendAttributeType_t attributeType,
                         int64_t elementCount,
                         const void* arrayOfElements,
                         const char* errorPrefix)
{
    checkSetArgs(HIPDNN_TYPE_BACKEND_DESCRIPTOR, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": elementCount is not 1");

    auto tensorDesc = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        arrayOfElements,
        HIPDNN_STATUS_BAD_PARAM,
        std::string(errorPrefix) + ": Failed to unpack tensor descriptor");
    THROW_IF_FALSE(tensorDesc->isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   std::string(errorPrefix) + ": Tensor descriptor not finalized");

    descTarget = tensorDesc;
    uidTarget = tensorDesc->getData().uid;
}

void getTensorDescriptor(const std::shared_ptr<TensorDescriptor>& descSource,
                         hipdnnBackendAttributeType_t attributeType,
                         int64_t requestedElementCount,
                         int64_t* elementCount,
                         void* arrayOfElements,
                         const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_BACKEND_DESCRIPTOR, attributeType, errorPrefix);

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
    HipdnnBackendDescriptor::packDescriptor(descSource, arrayOfElements);
}

void setOptionalTensorDescriptor(std::shared_ptr<TensorDescriptor>& descTarget,
                                 flatbuffers::Optional<int64_t>& uidTarget,
                                 hipdnnBackendAttributeType_t attributeType,
                                 int64_t elementCount,
                                 const void* arrayOfElements,
                                 const char* errorPrefix)
{
    int64_t uid = 0;
    setTensorDescriptor(descTarget, uid, attributeType, elementCount, arrayOfElements, errorPrefix);
    uidTarget = uid;
}

void getOptionalTensorDescriptor(const std::shared_ptr<TensorDescriptor>& descSource,
                                 hipdnnBackendAttributeType_t attributeType,
                                 int64_t requestedElementCount,
                                 int64_t* elementCount,
                                 void* arrayOfElements,
                                 const char* errorPrefix)
{
    if(!descSource)
    {
        checkGetArgs(HIPDNN_TYPE_BACKEND_DESCRIPTOR, attributeType, errorPrefix);
        if(elementCount != nullptr)
        {
            *elementCount = 0;
        }
        return;
    }
    getTensorDescriptor(descSource,
                        attributeType,
                        requestedElementCount,
                        elementCount,
                        arrayOfElements,
                        errorPrefix);
}

void setTensorDescriptorArray(std::vector<std::shared_ptr<TensorDescriptor>>& descTarget,
                              std::vector<int64_t>& uidTarget,
                              hipdnnBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              const void* arrayOfElements,
                              const char* errorPrefix)
{
    checkSetArgs(HIPDNN_TYPE_BACKEND_DESCRIPTOR, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_FALSE(elementCount >= 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": elementCount < 1");

    auto descs = static_cast<HipdnnBackendDescriptor* const*>(arrayOfElements);
    std::vector<std::shared_ptr<TensorDescriptor>> tensorDescs;
    std::vector<int64_t> uids;

    tensorDescs.reserve(static_cast<size_t>(elementCount));
    uids.reserve(static_cast<size_t>(elementCount));

    for(int64_t i = 0; i < elementCount; ++i)
    {
        auto tensorDesc = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
            &descs[i],
            HIPDNN_STATUS_BAD_PARAM,
            std::string(errorPrefix) + ": Failed to unpack tensor descriptor");
        THROW_IF_FALSE(tensorDesc->isFinalized(),
                       HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                       std::string(errorPrefix) + ": Tensor descriptor not finalized");
        tensorDescs.push_back(tensorDesc);
        uids.push_back(tensorDesc->getData().uid);
    }

    descTarget = std::move(tensorDescs);
    uidTarget = std::move(uids);
}

void getTensorDescriptorArray(const std::vector<std::shared_ptr<TensorDescriptor>>& descSource,
                              hipdnnBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements,
                              const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_BACKEND_DESCRIPTOR, attributeType, errorPrefix);

    auto count = static_cast<int64_t>(descSource.size());

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      std::string(errorPrefix) + ": elementCount is null");
        *elementCount = count;
        return;
    }

    if(elementCount != nullptr)
    {
        *elementCount = count;
    }

    auto outDescs = static_cast<HipdnnBackendDescriptor**>(arrayOfElements);
    auto copyCount = std::min(requestedElementCount, count);
    for(int64_t i = 0; i < copyCount; ++i)
    {
        outDescs[i] = HipdnnBackendDescriptor::packDescriptor(descSource[static_cast<size_t>(i)]);
    }
}

} // namespace hipdnn_backend
