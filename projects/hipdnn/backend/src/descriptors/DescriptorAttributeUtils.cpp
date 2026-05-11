// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorAttributeUtils.hpp"
#include "BackendDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "DataTypeConversion.hpp"

#include <algorithm>
#include <cstring>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

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

void setString(std::string& target,
               hipdnnBackendAttributeType_t attributeType,
               int64_t elementCount,
               const void* arrayOfElements,
               const char* errorPrefix)
{
    checkSetArgs(HIPDNN_TYPE_CHAR, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_LT(elementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                std::string(errorPrefix) + ": elementCount is negative");
    target
        = std::string(static_cast<const char*>(arrayOfElements), static_cast<size_t>(elementCount));
}

void setBoundedString(std::string& target,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements,
                      const char* errorPrefix,
                      int64_t maxLength,
                      int64_t minLength)
{
    THROW_IF_TRUE(elementCount < minLength,
                  HIPDNN_STATUS_BAD_PARAM,
                  std::string(errorPrefix)
                      + ": elementCount must be >= " + std::to_string(minLength));
    THROW_IF_TRUE(elementCount > maxLength,
                  HIPDNN_STATUS_BAD_PARAM,
                  std::string(errorPrefix) + ": elementCount exceeds maximum length ("
                      + std::to_string(maxLength) + ")");
    setString(target, attributeType, elementCount, arrayOfElements, errorPrefix);
}

void getString(const std::string& source,
               hipdnnBackendAttributeType_t attributeType,
               int64_t requestedElementCount,
               int64_t* elementCount,
               void* arrayOfElements,
               const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_CHAR, attributeType, errorPrefix);

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      std::string(errorPrefix) + ": elementCount is null");
        *elementCount = static_cast<int64_t>(source.size() + 1);
        return;
    }

    THROW_IF_LT(requestedElementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                std::string(errorPrefix) + ": requestedElementCount is negative");

    auto maxSize = static_cast<size_t>(requestedElementCount);
    hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
        static_cast<char*>(arrayOfElements), source.c_str(), maxSize);

    if(elementCount != nullptr)
    {
        *elementCount = static_cast<int64_t>(std::min(source.size() + 1, maxSize));
    }
}

void setByteArray(std::vector<uint8_t>& target,
                  hipdnnBackendAttributeType_t attributeType,
                  int64_t elementCount,
                  const void* arrayOfElements,
                  const char* errorPrefix)
{
    checkSetArgs(HIPDNN_TYPE_CHAR, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_LT(elementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                std::string(errorPrefix) + ": elementCount is negative");
    auto ptr = static_cast<const uint8_t*>(arrayOfElements);
    target.assign(ptr, ptr + static_cast<size_t>(elementCount));
}

void getByteArray(const std::vector<uint8_t>& source,
                  hipdnnBackendAttributeType_t attributeType,
                  int64_t requestedElementCount,
                  int64_t* elementCount,
                  void* arrayOfElements,
                  const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_CHAR, attributeType, errorPrefix);

    auto count = static_cast<int64_t>(source.size());

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      std::string(errorPrefix) + ": elementCount is null");
        *elementCount = count;
        return;
    }

    THROW_IF_LT(requestedElementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                std::string(errorPrefix) + ": requestedElementCount is negative");

    auto copyCount = std::min(requestedElementCount, count);
    if(elementCount != nullptr)
    {
        *elementCount = copyCount;
    }
    std::memcpy(arrayOfElements, source.data(), static_cast<size_t>(copyCount));
}

void setDataType(hipdnn_flatbuffers_sdk::data_objects::DataType& target,
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

void getDataType(hipdnn_flatbuffers_sdk::data_objects::DataType source,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements,
                 const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_DATA_TYPE, attributeType, errorPrefix);

    // UNSET storage means the field was never assigned a value. Report count=0 so
    // callers can distinguish "absent" from a real value, matching the pattern
    // used by getString and other optional-by-default getters. Validation that a
    // particular field must be set lives in the descriptor's finalize().
    if(source == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      std::string(errorPrefix) + ": elementCount is null");
        *elementCount = 0;
        return;
    }

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

void setConvMode(hipdnn_flatbuffers_sdk::data_objects::ConvMode& target,
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

void getConvMode(hipdnn_flatbuffers_sdk::data_objects::ConvMode source,
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

void setPointwiseMode(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode& target,
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

void getPointwiseMode(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode source,
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

void setNormFwdPhase(hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase& target,
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

void getNormFwdPhase(hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase source,
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

void setReductionMode(hipdnn_flatbuffers_sdk::data_objects::ReductionMode& target,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements,
                      const char* errorPrefix)
{
    checkSetArgs(HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": elementCount is not 1");
    hipdnnReduceTensorOp_t tmp;
    std::memcpy(&tmp, arrayOfElements, sizeof(tmp));
    target = toSdkReductionMode(tmp);
}

void getReductionMode(hipdnn_flatbuffers_sdk::data_objects::ReductionMode source,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements,
                      const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE, attributeType, errorPrefix);

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
    auto tmp = fromSdkReductionMode(source);
    std::memcpy(arrayOfElements, &tmp, sizeof(tmp));
    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
}

void getOperationType(hipdnnOperationType_ext_t source,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements,
                      const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_OPERATION_TYPE_EXT, attributeType, errorPrefix);

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
    std::memcpy(arrayOfElements, &source, sizeof(hipdnnOperationType_ext_t));
    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
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
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      std::string(errorPrefix) + ": elementCount is null");
        *elementCount = 0;
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
            static_cast<const void*>(&descs[i]),
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

    THROW_IF_LT(requestedElementCount,
                static_cast<int64_t>(0),
                HIPDNN_STATUS_BAD_PARAM,
                std::string(errorPrefix) + ": requestedElementCount is negative");

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

void setDiagonalAlignment(hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment& target,
                          hipdnnBackendAttributeType_t attributeType,
                          int64_t elementCount,
                          const void* arrayOfElements,
                          const char* errorPrefix)
{
    checkSetArgs(HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": elementCount is not 1");
    hipdnnDiagonalAlignment_t tmp;
    std::memcpy(&tmp, arrayOfElements, sizeof(tmp));
    target = toSdkDiagonalAlignment(tmp);
}

void getDiagonalAlignment(hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment source,
                          hipdnnBackendAttributeType_t attributeType,
                          int64_t requestedElementCount,
                          int64_t* elementCount,
                          void* arrayOfElements,
                          const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT, attributeType, errorPrefix);

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
    auto tmp = fromSdkDiagonalAlignment(source);
    std::memcpy(arrayOfElements, &tmp, sizeof(tmp));
    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
}

void setAttentionImplementation(
    hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation& target,
    hipdnnBackendAttributeType_t attributeType,
    int64_t elementCount,
    const void* arrayOfElements,
    const char* errorPrefix)
{
    checkSetArgs(
        HIPDNN_TYPE_ATTENTION_IMPLEMENTATION_EXT, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": elementCount is not 1");
    hipdnnAttentionImplementation_t tmp;
    std::memcpy(&tmp, arrayOfElements, sizeof(tmp));
    target = toSdkAttentionImplementation(tmp);
}

void getAttentionImplementation(
    hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation source,
    hipdnnBackendAttributeType_t attributeType,
    int64_t requestedElementCount,
    int64_t* elementCount,
    void* arrayOfElements,
    const char* errorPrefix)
{
    checkGetArgs(HIPDNN_TYPE_ATTENTION_IMPLEMENTATION_EXT, attributeType, errorPrefix);

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
    auto tmp = fromSdkAttentionImplementation(source);
    std::memcpy(arrayOfElements, &tmp, sizeof(tmp));
    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
}

void copyKnobValueUnion(const hipdnn_flatbuffers_sdk::data_objects::KnobValueUnion& src,
                        hipdnn_flatbuffers_sdk::data_objects::KnobValueUnion& dst,
                        const char* errorPrefix)
{
    switch(src.type)
    {
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue:
    {
        hipdnn_flatbuffers_sdk::data_objects::IntValueT intVal;
        intVal.value = src.AsIntValue()->value;
        dst.Set(intVal);
        break;
    }
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue:
    {
        hipdnn_flatbuffers_sdk::data_objects::FloatValueT floatVal;
        floatVal.value = src.AsFloatValue()->value;
        dst.Set(floatVal);
        break;
    }
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue:
    {
        hipdnn_flatbuffers_sdk::data_objects::StringValueT strVal;
        strVal.value = src.AsStringValue()->value;
        dst.Set(std::move(strVal));
        break;
    }
    default:
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              std::string(errorPrefix) + ": unknown value type ("
                                  + std::to_string(static_cast<int>(src.type)) + ")");
    }
}

void setKnobValueUnion(hipdnn_flatbuffers_sdk::data_objects::KnobValueUnion& target,
                       hipdnnBackendAttributeType_t attributeType,
                       int64_t elementCount,
                       const void* arrayOfElements,
                       const char* errorPrefix,
                       int64_t maxStringLength)
{
    switch(attributeType)
    {
    case HIPDNN_TYPE_INT64:
    {
        hipdnn_flatbuffers_sdk::data_objects::IntValueT intVal;
        setScalar(intVal.value,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  errorPrefix);
        target.Set(intVal);
        break;
    }
    case HIPDNN_TYPE_DOUBLE:
    {
        hipdnn_flatbuffers_sdk::data_objects::FloatValueT floatVal;
        setScalar(floatVal.value,
                  HIPDNN_TYPE_DOUBLE,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  errorPrefix);
        target.Set(floatVal);
        break;
    }
    case HIPDNN_TYPE_CHAR:
    {
        hipdnn_flatbuffers_sdk::data_objects::StringValueT strVal;
        setBoundedString(strVal.value,
                         attributeType,
                         elementCount,
                         arrayOfElements,
                         errorPrefix,
                         maxStringLength);
        target.Set(std::move(strVal));
        break;
    }
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              std::string(errorPrefix)
                                  + ": unsupported attribute type for knob value: "
                                  + hipdnn_backend::hipdnnGetAttributeTypeString(attributeType));
    }
}

void getKnobValueUnion(const hipdnn_flatbuffers_sdk::data_objects::KnobValueUnion& source,
                       hipdnnBackendAttributeType_t attributeType,
                       int64_t requestedElementCount,
                       int64_t* elementCount,
                       void* arrayOfElements,
                       const char* errorPrefix)
{
    switch(source.type)
    {
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue:
        getScalar(source.AsIntValue()->value,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  errorPrefix);
        break;
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue:
        getScalar(source.AsFloatValue()->value,
                  HIPDNN_TYPE_DOUBLE,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  errorPrefix);
        break;
    case hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue:
        getString(source.AsStringValue()->value,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  errorPrefix);
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              std::string(errorPrefix) + ": unknown value type ("
                                  + std::to_string(static_cast<int>(source.type)) + ")");
    }
}

} // namespace hipdnn_backend
