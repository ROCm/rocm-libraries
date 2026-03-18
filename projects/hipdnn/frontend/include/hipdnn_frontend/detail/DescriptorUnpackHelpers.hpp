// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hipdnn_frontend::detail
{

// These helpers wrap the hipDNN backend C API for reading attributes.
// They provide the inverse of DescriptorHelpers.hpp (which sets attributes).
// Each helper converts backend failures into frontend Error values.

/// Gets the element count for an attribute (queries with nullptr arrayOfElements).
[[nodiscard]] inline Error getDescriptorAttrCount(hipdnnBackendDescriptor_t desc,
                                                  hipdnnBackendAttributeName_t attrName,
                                                  hipdnnBackendAttributeType_t attrType,
                                                  int64_t& count,
                                                  const std::string& errorContext)
{
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(desc, attrName, attrType, 0, &count, nullptr),
        "Failed to get count for " + errorContext);
    return {};
}

/// Gets a vector-valued int64 attribute (queries count first, then allocates and queries values).
[[nodiscard]] inline Error getDescriptorAttrVec(hipdnnBackendDescriptor_t desc,
                                                hipdnnBackendAttributeName_t attrName,
                                                std::vector<int64_t>& values,
                                                const std::string& errorContext)
{
    int64_t count = 0;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrCount(desc, attrName, HIPDNN_TYPE_INT64, count, errorContext));

    if(count <= 0)
    {
        values.clear();
        return {};
    }

    values.resize(static_cast<size_t>(count));
    int64_t actualCount = 0;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(
            desc, attrName, HIPDNN_TYPE_INT64, count, &actualCount, values.data()),
        "Failed to get " + errorContext);
    if(actualCount != count)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Element count mismatch for " + errorContext + ": expected " + std::to_string(count)
                    + " but got " + std::to_string(actualCount)};
    }
    return {};
}

/// Gets a scalar attribute of a given type.
template <typename T>
[[nodiscard]] inline Error getDescriptorAttrScalar(hipdnnBackendDescriptor_t desc,
                                                   hipdnnBackendAttributeName_t attrName,
                                                   hipdnnBackendAttributeType_t attrType,
                                                   T& value,
                                                   const std::string& errorContext)
{
    int64_t actualCount = 0;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(desc, attrName, attrType, 1, &actualCount, &value),
        "Failed to get " + errorContext);
    return {};
}

/// Gets a string attribute (char array) from a backend descriptor.
/// Queries the character count first, then retrieves the string value.
/// If the attribute is not supported or the string is empty, sets value to empty and returns
/// success.
[[nodiscard]] inline Error getDescriptorAttrString(hipdnnBackendDescriptor_t desc,
                                                   hipdnnBackendAttributeName_t attrName,
                                                   std::string& value,
                                                   const std::string& errorContext)
{
    int64_t count = 0;
    auto countStatus = hipdnnBackend()->backendGetAttribute(
        desc, attrName, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    if(countStatus == HIPDNN_STATUS_NOT_SUPPORTED)
    {
        value.clear();
        return {};
    }
    if(countStatus != HIPDNN_STATUS_SUCCESS)
    {
        std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> backendErrMsg{};
        hipdnnBackend()->getLastErrorString(backendErrMsg.data(), backendErrMsg.size());
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to get count for " + errorContext
                    + " Backend error: " + backendErrMsg.data()};
    }
    if(count <= 0)
    {
        value.clear();
        return {};
    }

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(
            desc, attrName, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data()),
        "Failed to get " + errorContext);

    // The backend returns a null-terminated string; construct std::string from it
    value = std::string(buffer.data());
    return {};
}

/// Gets a single backend descriptor attribute wrapped in RAII.
/// backendGetAttribute transfers ownership of the wrapper descriptor to the caller —
/// the caller is responsible for destroying it. The returned
/// ScopedHipdnnBackendDescriptor takes that ownership and destroys the descriptor
/// when it goes out of scope.
[[nodiscard]] inline std::pair<ScopedHipdnnBackendDescriptor, Error>
    getDescriptorAttrDesc(hipdnnBackendDescriptor_t sourceDescriptor,
                          hipdnnBackendAttributeName_t attrName,
                          const std::string& errorContext)
{
    hipdnnBackendDescriptor_t rawDesc = nullptr;
    int64_t actualCount = 0;
    auto status = hipdnnBackend()->backendGetAttribute(sourceDescriptor,
                                                       attrName,
                                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       &actualCount,
                                                       static_cast<void*>(&rawDesc));
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> backendErrMsg{};
        hipdnnBackend()->getLastErrorString(backendErrMsg.data(), backendErrMsg.size());
        return std::make_pair(
            ScopedHipdnnBackendDescriptor(),
            Error{ErrorCode::HIPDNN_BACKEND_ERROR,
                  "Failed to get " + errorContext + " Backend error: " + backendErrMsg.data()});
    }
    return std::make_pair(ScopedHipdnnBackendDescriptor(rawDesc), Error{});
}

/// Gets an array of backend descriptor attributes, each wrapped in RAII.
/// Queries the element count first, then retrieves the descriptors.
/// Each returned ScopedHipdnnBackendDescriptor owns its wrapper descriptor.
[[nodiscard]] inline std::pair<std::vector<ScopedHipdnnBackendDescriptor>, Error>
    getDescriptorAttrDescArray(hipdnnBackendDescriptor_t sourceDescriptor,
                               hipdnnBackendAttributeName_t attrName,
                               const std::string& errorContext)
{
    // Query count
    int64_t count = 0;
    auto countStatus = hipdnnBackend()->backendGetAttribute(
        sourceDescriptor, attrName, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr);
    if(countStatus != HIPDNN_STATUS_SUCCESS)
    {
        std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> backendErrMsg{};
        hipdnnBackend()->getLastErrorString(backendErrMsg.data(), backendErrMsg.size());
        return std::make_pair(std::vector<ScopedHipdnnBackendDescriptor>{},
                              Error{ErrorCode::HIPDNN_BACKEND_ERROR,
                                    "Failed to get count for " + errorContext
                                        + " Backend error: " + backendErrMsg.data()});
    }

    if(count <= 0)
    {
        return std::make_pair(std::vector<ScopedHipdnnBackendDescriptor>{}, Error{});
    }

    // Retrieve raw descriptors
    std::vector<hipdnnBackendDescriptor_t> rawDescs(static_cast<size_t>(count));
    int64_t actualCount = 0;
    auto getStatus = hipdnnBackend()->backendGetAttribute(sourceDescriptor,
                                                          attrName,
                                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                          count,
                                                          &actualCount,
                                                          static_cast<void*>(rawDescs.data()));
    if(getStatus != HIPDNN_STATUS_SUCCESS)
    {
        std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> backendErrMsg{};
        hipdnnBackend()->getLastErrorString(backendErrMsg.data(), backendErrMsg.size());
        for(auto& desc : rawDescs)
        {
            if(desc != nullptr)
            {
                hipdnnBackend()->backendDestroyDescriptor(desc);
            }
        }
        return std::make_pair(
            std::vector<ScopedHipdnnBackendDescriptor>{},
            Error{ErrorCode::HIPDNN_BACKEND_ERROR,
                  "Failed to get " + errorContext + " Backend error: " + backendErrMsg.data()});
    }

    if(actualCount < 0 || actualCount > count)
    {
        for(int64_t i = 0; i < count; ++i)
        {
            if(rawDescs[static_cast<size_t>(i)] != nullptr)
            {
                hipdnnBackend()->backendDestroyDescriptor(rawDescs[static_cast<size_t>(i)]);
            }
        }
        return std::make_pair(
            std::vector<ScopedHipdnnBackendDescriptor>{},
            Error{ErrorCode::HIPDNN_BACKEND_ERROR,
                  "Unexpected element count from backendGetAttribute for " + errorContext});
    }

    // Wrap each in RAII
    std::vector<ScopedHipdnnBackendDescriptor> result;
    result.reserve(static_cast<size_t>(actualCount));
    for(int64_t i = 0; i < actualCount; ++i)
    {
        result.emplace_back(rawDescs[static_cast<size_t>(i)]);
    }

    return std::make_pair(std::move(result), Error{});
}

/// Unpacks a graph-level data type attribute from a backend descriptor.
/// Queries the attribute, validates the count, and converts the hipdnnDataType_t
/// to a frontend DataType.
[[nodiscard]] inline std::pair<DataType, Error>
    unpackGraphDataType(hipdnnBackendDescriptor_t desc,
                        hipdnnBackendAttributeName_t attrName,
                        const std::string& errorContext)
{
    hipdnnDataType_t dt{};
    auto err = getDescriptorAttrScalar(desc, attrName, HIPDNN_TYPE_DATA_TYPE, dt, errorContext);
    if(err.is_bad())
    {
        return {DataType::NOT_SET, err};
    }
    return fromHipdnnDataType(dt);
}

/// Extracts TensorAttributes from a backend TensorDescriptor.
/// The tensor descriptor must already be finalized.
[[nodiscard]] inline Error unpackTensorAttributes(hipdnnBackendDescriptor_t tensorDesc,
                                                  std::shared_ptr<graph::TensorAttributes>& tensor)
{
    // Read UID
    int64_t uid = 0;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(
        tensorDesc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, uid, "tensor UID"));

    // Read data type
    hipdnnDataType_t dataType{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(tensorDesc,
                                               HIPDNN_ATTR_TENSOR_DATA_TYPE,
                                               HIPDNN_TYPE_DATA_TYPE,
                                               dataType,
                                               "tensor data type"));

    // Read dimensions
    std::vector<int64_t> dims;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrVec(tensorDesc, HIPDNN_ATTR_TENSOR_DIMENSIONS, dims, "tensor dimensions"));

    // Read strides
    std::vector<int64_t> strides;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrVec(tensorDesc, HIPDNN_ATTR_TENSOR_STRIDES, strides, "tensor strides"));

    // Read is_virtual
    bool isVirtual = false;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(tensorDesc,
                                               HIPDNN_ATTR_TENSOR_IS_VIRTUAL,
                                               HIPDNN_TYPE_BOOLEAN,
                                               isVirtual,
                                               "tensor is_virtual"));

    // Read tensor name (may be empty if not set)
    std::string name;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(tensorDesc, HIPDNN_ATTR_TENSOR_NAME_EXT, name, "tensor name"));

    // Convert data type
    auto [dt, dtErr] = fromHipdnnDataType(dataType);
    if(dtErr.is_bad())
    {
        return dtErr;
    }

    // Construct the TensorAttributes object
    tensor = std::make_shared<graph::TensorAttributes>();
    tensor->set_uid(uid).set_data_type(dt).set_dim(dims).set_stride(strides).set_is_virtual(
        isVirtual);
    tensor->set_name(name);

    return {};
}

/// Unpacks a tensor from an operation descriptor and either finds it in the
/// tensor map (by UID) or creates a new one. This ensures tensor sharing
/// is preserved during reconstruction.
///
/// The tensor descriptor obtained from backendGetAttribute is a wrapper created by
/// packDescriptor(). getDescriptorAttrDesc() returns it already wrapped in
/// ScopedHipdnnBackendDescriptor RAII, ensuring cleanup on all paths.
[[nodiscard]] inline Error unpackAndRegisterTensor(
    hipdnnBackendDescriptor_t opDesc,
    hipdnnBackendAttributeName_t tensorAttrName,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    std::shared_ptr<graph::TensorAttributes>& outTensor,
    const std::string& errorContext)
{
    // Get the tensor descriptor from the operation, already wrapped in RAII.
    auto [tensorDesc, descErr] = getDescriptorAttrDesc(opDesc, tensorAttrName, errorContext);
    if(descErr.is_bad())
    {
        return descErr;
    }

    if(tensorDesc.get() == nullptr)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Null tensor descriptor for " + errorContext};
    }

    // First read the UID to check if we already have this tensor
    int64_t uid = 0;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(tensorDesc.get(),
                                               HIPDNN_ATTR_TENSOR_UNIQUE_ID,
                                               HIPDNN_TYPE_INT64,
                                               uid,
                                               "tensor UID for " + errorContext));

    // Check if this tensor already exists in the map
    auto it = tensorMap.find(uid);
    if(it != tensorMap.end())
    {
        outTensor = it->second;
        return {};
    }

    // Tensor not in map, unpack it fully
    HIPDNN_CHECK_ERROR(unpackTensorAttributes(tensorDesc.get(), outTensor));

    // Register in map for future sharing
    tensorMap[uid] = outTensor;

    return {};
}

} // namespace hipdnn_frontend::detail
