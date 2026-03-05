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
    target = toSdkDataType(*static_cast<const hipdnnDataType_t*>(arrayOfElements));
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
    *static_cast<hipdnnDataType_t*>(arrayOfElements) = fromSdkDataType(source);
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
    target = toSdkConvMode(*static_cast<const hipdnnConvolutionMode_t*>(arrayOfElements));
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
    *static_cast<hipdnnConvolutionMode_t*>(arrayOfElements) = fromSdkConvMode(source);
    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
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

} // namespace hipdnn_backend
