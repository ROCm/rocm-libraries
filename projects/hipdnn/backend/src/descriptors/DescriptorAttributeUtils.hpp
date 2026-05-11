// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipdnnAttentionImplementation.h"
#include "HipdnnBackendAttributeName.h"
#include "HipdnnBackendAttributeType.h"
#include "HipdnnDataType.h"
#include "HipdnnDiagonalAlignment.h"
#include "HipdnnException.hpp"
#include "HipdnnNormFwdPhase.h"
#include "HipdnnOperationType.h"
#include "HipdnnPointwiseMode.h"
#include "HipdnnReduceTensorOp.h"
#include "TensorDescriptor.hpp"
#include <cstring>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_common_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/norm_common_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <memory>
#include <string>
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

void setString(std::string& target,
               hipdnnBackendAttributeType_t attributeType,
               int64_t elementCount,
               const void* arrayOfElements,
               const char* errorPrefix);

void setBoundedString(std::string& target,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements,
                      const char* errorPrefix,
                      int64_t maxLength,
                      int64_t minLength = 0);

void getString(const std::string& source,
               hipdnnBackendAttributeType_t attributeType,
               int64_t requestedElementCount,
               int64_t* elementCount,
               void* arrayOfElements,
               const char* errorPrefix);

void setByteArray(std::vector<uint8_t>& target,
                  hipdnnBackendAttributeType_t attributeType,
                  int64_t elementCount,
                  const void* arrayOfElements,
                  const char* errorPrefix);

void getByteArray(const std::vector<uint8_t>& source,
                  hipdnnBackendAttributeType_t attributeType,
                  int64_t requestedElementCount,
                  int64_t* elementCount,
                  void* arrayOfElements,
                  const char* errorPrefix);

template <typename T>
void setScalarVector(std::vector<T>& target,
                     hipdnnBackendAttributeType_t expectedType,
                     hipdnnBackendAttributeType_t attributeType,
                     int64_t elementCount,
                     const void* arrayOfElements,
                     const char* errorPrefix)
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "setScalarVector requires a trivially copyable type");
    checkSetArgs(expectedType, attributeType, arrayOfElements, errorPrefix);
    THROW_IF_FALSE(elementCount > 0,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": elementCount must be positive");
    target.resize(static_cast<size_t>(elementCount));
    std::memcpy(target.data(), arrayOfElements, static_cast<size_t>(elementCount) * sizeof(T));
}

template <typename T>
void getScalarVector(const std::vector<T>& source,
                     hipdnnBackendAttributeType_t expectedType,
                     hipdnnBackendAttributeType_t attributeType,
                     int64_t requestedElementCount,
                     int64_t* elementCount,
                     void* arrayOfElements,
                     const char* errorPrefix)
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "getScalarVector requires a trivially copyable type");
    checkGetArgs(expectedType, attributeType, errorPrefix);

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
    std::memcpy(arrayOfElements, source.data(), copyCount * sizeof(T));
}

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

void setDataType(hipdnn_flatbuffers_sdk::data_objects::DataType& target,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElements,
                 const char* errorPrefix);

void getDataType(hipdnn_flatbuffers_sdk::data_objects::DataType source,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements,
                 const char* errorPrefix);

void setConvMode(hipdnn_flatbuffers_sdk::data_objects::ConvMode& target,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElements,
                 const char* errorPrefix);

void getConvMode(hipdnn_flatbuffers_sdk::data_objects::ConvMode source,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements,
                 const char* errorPrefix);

void getOperationType(hipdnnOperationType_ext_t source,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements,
                      const char* errorPrefix);

void setPointwiseMode(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode& target,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements,
                      const char* errorPrefix);

void getPointwiseMode(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode source,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements,
                      const char* errorPrefix);

// setOptionalScalar/getOptionalScalar are templated on flatbuffers::Optional<T>.
// This works with std::optional<T> members because flatbuffers aliases Optional to
// std::optional when FLATBUFFERS_USE_STD_OPTIONAL is defined. If a FlatBuffers upgrade
// changes this, the static_assert below will fire with a clear message.
static_assert(std::is_same_v<flatbuffers::Optional<int>, std::optional<int>>,
              "flatbuffers::Optional must alias std::optional for these overloads "
              "to work with std::optional members; add explicit std::optional overloads "
              "if this changes");

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
void setNormFwdPhase(hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase& target,
                     hipdnnBackendAttributeType_t attributeType,
                     int64_t elementCount,
                     const void* arrayOfElements,
                     const char* errorPrefix);

void getNormFwdPhase(hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase source,
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

void setDiagonalAlignment(hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment& target,
                          hipdnnBackendAttributeType_t attributeType,
                          int64_t elementCount,
                          const void* arrayOfElements,
                          const char* errorPrefix);

void getDiagonalAlignment(hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment source,
                          hipdnnBackendAttributeType_t attributeType,
                          int64_t requestedElementCount,
                          int64_t* elementCount,
                          void* arrayOfElements,
                          const char* errorPrefix);

void setAttentionImplementation(
    hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation& target,
    hipdnnBackendAttributeType_t attributeType,
    int64_t elementCount,
    const void* arrayOfElements,
    const char* errorPrefix);

void getAttentionImplementation(
    hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation source,
    hipdnnBackendAttributeType_t attributeType,
    int64_t requestedElementCount,
    int64_t* elementCount,
    void* arrayOfElements,
    const char* errorPrefix);

void setReductionMode(hipdnn_flatbuffers_sdk::data_objects::ReductionMode& target,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements,
                      const char* errorPrefix);

void getReductionMode(hipdnn_flatbuffers_sdk::data_objects::ReductionMode source,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements,
                      const char* errorPrefix);

/// Conditionally adds a non-null tensor descriptor to a vector.
/// Used by getTensorDescriptors() to collect optional tensors.
inline void addIfSet(std::vector<std::shared_ptr<TensorDescriptor>>& tensors,
                     const std::shared_ptr<TensorDescriptor>& desc)
{
    if(desc)
    {
        tensors.push_back(desc);
    }
}

/// Looks up an optional tensor UID in a tensor map.
/// Returns nullptr if the UID has no value or is not found in the map.
inline std::shared_ptr<TensorDescriptor> findOptionalTensor(
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap,
    const flatbuffers::Optional<int64_t>& uid)
{
    if(!uid.has_value())
    {
        return nullptr;
    }
    auto it = tensorMap.find(uid.value());
    return (it != tensorMap.end()) ? it->second : nullptr;
}

/// Converts a flatbuffers::Optional<T> to a string for toString() output.
/// Returns "null" if the optional has no value.
template <typename T>
std::string optionalToString(const flatbuffers::Optional<T>& opt)
{
    return opt.has_value() ? std::to_string(*opt) : "null";
}

/// Specialization for Optional<bool> that produces "true"/"false"/"null".
inline std::string optionalBoolToString(const flatbuffers::Optional<bool>& opt)
{
    if(!opt.has_value())
    {
        return "null";
    }
    return *opt ? "true" : "false";
}

/// Deep-copy a KnobValueUnion into another KnobValueUnion.
/// Used by both KnobDescriptor::toKnobT() and KnobSettingDescriptor::toKnobSettingT().
void copyKnobValueUnion(const hipdnn_flatbuffers_sdk::data_objects::KnobValueUnion& src,
                        hipdnn_flatbuffers_sdk::data_objects::KnobValueUnion& dst,
                        const char* errorPrefix);

/// Set a KnobValueUnion from C-API setAttribute parameters.
/// Switches on attributeType to store an int64, double, or bounded string.
void setKnobValueUnion(hipdnn_flatbuffers_sdk::data_objects::KnobValueUnion& target,
                       hipdnnBackendAttributeType_t attributeType,
                       int64_t elementCount,
                       const void* arrayOfElements,
                       const char* errorPrefix,
                       int64_t maxStringLength);

/// Get a KnobValueUnion into C-API getAttribute output parameters.
/// Switches on source.type to retrieve an int64, double, or string.
void getKnobValueUnion(const hipdnn_flatbuffers_sdk::data_objects::KnobValueUnion& source,
                       hipdnnBackendAttributeType_t attributeType,
                       int64_t requestedElementCount,
                       int64_t* elementCount,
                       void* arrayOfElements,
                       const char* errorPrefix);

// ============================================================================
// Convolution descriptor helpers
//
// The three convolution operation descriptors (Fwd, Bwd, Wrw) share identical
// attribute handling for convolution parameters. These templates consolidate
// the shared switch-case logic and finalize validation, leaving each descriptor
// to handle only its tensor-specific cases.
// ============================================================================

/// Validates the shared convolution fields during finalize().
/// Each descriptor calls this after its own tensor null-checks.
template <typename DataT>
void validateConvolutionFinalize(const DataT& data,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType,
                                 const char* className)
{
    THROW_IF_TRUE(data.pre_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  std::string(className) + "::finalize() failed: pre_padding not set");
    THROW_IF_TRUE(data.post_padding.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  std::string(className) + "::finalize() failed: post_padding not set");
    THROW_IF_TRUE(data.stride.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  std::string(className) + "::finalize() failed: stride not set");
    THROW_IF_TRUE(data.dilation.empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  std::string(className) + "::finalize() failed: dilation not set");
    THROW_IF_TRUE(computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  std::string(className) + "::finalize() failed: compute data type not set");
    THROW_IF_TRUE(data.conv_mode == hipdnn_flatbuffers_sdk::data_objects::ConvMode::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  std::string(className) + "::finalize() failed: conv_mode not set");
}

/// Handles the shared convolution setAttribute cases.
/// Called from the default branch of each descriptor's setAttribute switch.
template <typename DataT>
void setConvolutionAttribute(DataT& data,
                             hipdnn_flatbuffers_sdk::data_objects::DataType& computeDataType,
                             std::string& name,
                             hipdnnBackendAttributeName_t attributeName,
                             hipdnnBackendAttributeType_t attributeType,
                             int64_t elementCount,
                             const void* arrayOfElements,
                             const char* errorPrefix)
{
    switch(attributeName)
    {
    case HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS:
        setScalarVector<int64_t>(data.pre_padding,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 errorPrefix);
        break;
    case HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS:
        setScalarVector<int64_t>(data.post_padding,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 errorPrefix);
        break;
    case HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES:
        setScalarVector<int64_t>(data.stride,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 errorPrefix);
        break;
    case HIPDNN_ATTR_CONVOLUTION_DILATIONS:
        setScalarVector<int64_t>(data.dilation,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 elementCount,
                                 arrayOfElements,
                                 errorPrefix);
        break;
    case HIPDNN_ATTR_CONVOLUTION_CONV_MODE:
        setConvMode(data.conv_mode, attributeType, elementCount, arrayOfElements, errorPrefix);
        break;
    case HIPDNN_ATTR_CONVOLUTION_COMP_TYPE:
        setDataType(computeDataType, attributeType, elementCount, arrayOfElements, errorPrefix);
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(name, attributeType, elementCount, arrayOfElements, errorPrefix);
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              std::string(errorPrefix) + ": attributeName not supported");
    }
}

/// Handles the shared convolution getAttribute cases.
/// Called from the default branch of each descriptor's getAttribute switch.
template <typename DataT>
void getConvolutionAttribute(const DataT& data,
                             hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType,
                             const std::string& name,
                             hipdnnOperationType_ext_t operationType,
                             hipdnnBackendAttributeName_t attributeName,
                             hipdnnBackendAttributeType_t attributeType,
                             int64_t requestedElementCount,
                             int64_t* elementCount,
                             void* arrayOfElements,
                             const char* errorPrefix)
{
    switch(attributeName)
    {
    case HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS:
        getScalarVector<int64_t>(data.pre_padding,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 errorPrefix);
        break;
    case HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS:
        getScalarVector<int64_t>(data.post_padding,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 errorPrefix);
        break;
    case HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES:
        getScalarVector<int64_t>(data.stride,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 errorPrefix);
        break;
    case HIPDNN_ATTR_CONVOLUTION_DILATIONS:
        getScalarVector<int64_t>(data.dilation,
                                 HIPDNN_TYPE_INT64,
                                 attributeType,
                                 requestedElementCount,
                                 elementCount,
                                 arrayOfElements,
                                 errorPrefix);
        break;
    case HIPDNN_ATTR_CONVOLUTION_CONV_MODE:
        getConvMode(data.conv_mode,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    errorPrefix);
        break;
    case HIPDNN_ATTR_CONVOLUTION_COMP_TYPE:
        getDataType(computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    errorPrefix);
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(
            name, attributeType, requestedElementCount, elementCount, arrayOfElements, errorPrefix);
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(operationType,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         errorPrefix);
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              std::string(errorPrefix) + ": attributeName not supported");
    }
}

} // namespace hipdnn_backend
