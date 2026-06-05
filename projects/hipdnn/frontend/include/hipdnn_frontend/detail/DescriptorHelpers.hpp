// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace hipdnn_frontend::detail
{

// These helpers wrap the hipDNN backend C API, which communicates errors via
// hipdnnStatus_t return codes. Each helper converts backend failures into
// frontend Error values (error code + message) for the frontend's non-throwing
// error model.

// Sets a string attribute (char array) on a backend descriptor.
inline Error setDescriptorAttrString(hipdnnBackendDescriptor_t desc,
                                     hipdnnBackendAttributeName_t attrName,
                                     const std::string& value,
                                     const std::string& errorContext)
{
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(
            desc, attrName, HIPDNN_TYPE_CHAR, static_cast<int64_t>(value.size()), value.c_str()),
        "Failed to set " + errorContext);
    return {};
}

// Sets a vector-valued attribute on a backend descriptor.
template <typename T>
inline Error setDescriptorAttrVec(hipdnnBackendDescriptor_t desc,
                                  hipdnnBackendAttributeName_t attrName,
                                  hipdnnBackendAttributeType_t attrType,
                                  const std::vector<T>& values,
                                  const std::string& errorContext)
{
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(
            desc, attrName, attrType, static_cast<int64_t>(values.size()), values.data()),
        "Failed to set " + errorContext);
    return {};
}

// Sets a scalar attribute on a backend descriptor.
template <typename T>
inline Error setDescriptorAttrScalar(hipdnnBackendDescriptor_t desc,
                                     hipdnnBackendAttributeName_t attrName,
                                     hipdnnBackendAttributeType_t attrType,
                                     const T& value,
                                     const std::string& errorContext)
{
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(
            desc, attrName, attrType, 1, static_cast<const void*>(&value)),
        "Failed to set " + errorContext);
    return {};
}

// Overload for std::monostate — this is unreachable when guarded by
// get_pass_by_value(), but required for std::visit to compile.
inline Error setDescriptorAttrTensorValue(hipdnnBackendDescriptor_t /*desc*/,
                                          std::monostate /*value*/,
                                          const std::string& errorContext)
{
    return {ErrorCode::HIPDNN_BACKEND_ERROR,
            "Tensor value variant is empty when setting " + errorContext};
}

// Passes raw bytes to the backend's TENSOR_VALUE_EXT attribute.
// The backend dispatches on the tensor's data_type to interpret the bytes.
template <typename T>
inline Error setDescriptorAttrTensorValue(hipdnnBackendDescriptor_t desc,
                                          const T& value,
                                          const std::string& errorContext)
{
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(desc,
                                             HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                             HIPDNN_TYPE_CHAR,
                                             static_cast<int64_t>(sizeof(T)),
                                             &value),
        "Failed to set " + errorContext);
    return {};
}

// Sets a data type attribute on a backend descriptor, converting from the
// frontend DataType enum to hipdnnDataType_t.
inline Error setDescriptorAttrDataType(hipdnnBackendDescriptor_t desc,
                                       hipdnnBackendAttributeName_t attrName,
                                       DataType type,
                                       const std::string& errorContext)
{
    auto hipdnnType = toHipdnnDataType(type);
    if(!hipdnnType.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "Unsupported data type for " + errorContext};
    }
    return setDescriptorAttrScalar(
        desc, attrName, HIPDNN_TYPE_DATA_TYPE, *hipdnnType, errorContext);
}

// Sets an optional scalar attribute on a backend descriptor. No-op if the
// optional has no value.
template <typename T>
inline Error setDescriptorAttrOptionalScalar(hipdnnBackendDescriptor_t desc,
                                             hipdnnBackendAttributeName_t attrName,
                                             hipdnnBackendAttributeType_t attrType,
                                             const std::optional<T>& value,
                                             const std::string& errorContext)
{
    if(!value.has_value())
    {
        return {};
    }
    return setDescriptorAttrScalar(desc, attrName, attrType, value.value(), errorContext);
}

// Sets a tensor reference attribute on an operation descriptor by looking up
// the tensor UID in the tensorDescs map.
inline Error setDescriptorAttrTensorRef(
    hipdnnBackendDescriptor_t desc,
    hipdnnBackendAttributeName_t attrName,
    int64_t tensorUid,
    const std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    const std::string& errorContext)
{
    auto it = tensorDescs.find(tensorUid);
    if(it == tensorDescs.end())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Tensor UID " + std::to_string(tensorUid) + " not found when setting "
                    + errorContext};
    }
    const auto descPtr = it->second.get();
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(
            desc, attrName, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, static_cast<const void*>(&descPtr)),
        "Failed to set " + errorContext);
    return {};
}

// Finalizes a backend descriptor.
inline Error finalizeDescriptor(hipdnnBackendDescriptor_t desc, const std::string& errorContext)
{
    HIPDNN_RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendFinalize(desc),
                                     "Failed to finalize " + errorContext);
    return {};
}

// Creates a backend tensor descriptor for the given TensorAttributes if one
// does not already exist in the map (keyed by UID).
inline Error
    createOrFindTensorDesc(std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
                           const std::shared_ptr<graph::TensorAttributes>& tensor)
{
    auto uid = tensor->get_uid();
    if(tensorDescs.find(uid) != tensorDescs.end())
    {
        return {};
    }

    ScopedHipdnnBackendDescriptor desc(HIPDNN_BACKEND_TENSOR_DESCRIPTOR);
    if(!desc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create tensor descriptor for uid " + std::to_string(uid)};
    }

    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(desc.get(),
                                               HIPDNN_ATTR_TENSOR_UNIQUE_ID,
                                               HIPDNN_TYPE_INT64,
                                               uid,
                                               "tensor UID " + std::to_string(uid)));

    auto& name = tensor->get_name();
    if(!name.empty())
    {
        HIPDNN_CHECK_ERROR(
            setDescriptorAttrString(desc.get(), HIPDNN_ATTR_TENSOR_NAME_EXT, name, "tensor name"));
    }

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(
        desc.get(), HIPDNN_ATTR_TENSOR_DATA_TYPE, tensor->get_data_type(), "tensor data type"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(desc.get(),
                                            HIPDNN_ATTR_TENSOR_DIMENSIONS,
                                            HIPDNN_TYPE_INT64,
                                            tensor->get_dim(),
                                            "tensor dimensions"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(desc.get(),
                                            HIPDNN_ATTR_TENSOR_STRIDES,
                                            HIPDNN_TYPE_INT64,
                                            tensor->get_stride(),
                                            "tensor strides"));

    const bool isVirtual = tensor->get_is_virtual();
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(desc.get(),
                                               HIPDNN_ATTR_TENSOR_IS_VIRTUAL,
                                               HIPDNN_TYPE_BOOLEAN,
                                               isVirtual,
                                               "tensor is_virtual"));

    if(tensor->get_pass_by_value())
    {
        HIPDNN_CHECK_ERROR(std::visit(
            [&](auto&& arg) -> Error {
                return setDescriptorAttrTensorValue(desc.get(), arg, "tensor value");
            },
            tensor->get_value_variant()));
    }

    HIPDNN_CHECK_ERROR(finalizeDescriptor(desc.get(), "tensor descriptor"));

    tensorDescs.emplace(uid, std::move(desc));
    return {};
}

// Creates a tensor descriptor (if needed) and sets it as a tensor reference
// attribute on the given operation descriptor. Combines createOrFindTensorDesc
// + setDescriptorAttrTensorRef in a single call.
inline Error
    ensureAndSetTensorRef(hipdnnBackendDescriptor_t desc,
                          hipdnnBackendAttributeName_t attrName,
                          const std::shared_ptr<graph::TensorAttributes>& tensor,
                          std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
                          const std::string& errorContext)
{
    HIPDNN_CHECK_ERROR(createOrFindTensorDesc(tensorDescs, tensor));
    return setDescriptorAttrTensorRef(desc, attrName, tensor->get_uid(), tensorDescs, errorContext);
}

// Creates a tensor descriptor (if needed) and sets it as a tensor reference
// attribute on the given operation descriptor. No-op if the tensor is null.
inline Error ensureAndSetOptionalTensorRef(
    hipdnnBackendDescriptor_t desc,
    hipdnnBackendAttributeName_t attrName,
    const std::shared_ptr<graph::TensorAttributes>& tensor,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    const std::string& errorContext)
{
    if(!tensor)
    {
        return {};
    }
    return ensureAndSetTensorRef(desc, attrName, tensor, tensorDescs, errorContext);
}

// Creates tensor descriptors (if needed) for each element in the array and
// sets them as a tensor array attribute on the given operation descriptor.
inline Error ensureAndSetTensorArrayRef(
    hipdnnBackendDescriptor_t desc,
    hipdnnBackendAttributeName_t attrName,
    const std::vector<std::shared_ptr<graph::TensorAttributes>>& tensors,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    const std::string& errorContext)
{
    std::vector<hipdnnBackendDescriptor_t> descPtrs;
    descPtrs.reserve(tensors.size());
    for(const auto& tensor : tensors)
    {
        HIPDNN_CHECK_ERROR(createOrFindTensorDesc(tensorDescs, tensor));
        descPtrs.push_back(tensorDescs.at(tensor->get_uid()).get());
    }
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(desc,
                                             attrName,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             static_cast<int64_t>(descPtrs.size()),
                                             static_cast<const void*>(descPtrs.data())),
        "Failed to set " + errorContext);
    return {};
}

} // namespace hipdnn_frontend::detail
