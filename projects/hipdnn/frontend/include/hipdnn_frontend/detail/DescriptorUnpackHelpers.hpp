// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstring>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <memory>
#include <optional>
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

/// Gets a vector-valued attribute (queries count first, then allocates and queries values).
template <typename T>
[[nodiscard]] inline Error getDescriptorAttrVec(hipdnnBackendDescriptor_t desc,
                                                hipdnnBackendAttributeName_t attrName,
                                                hipdnnBackendAttributeType_t attrType,
                                                std::vector<T>& values,
                                                const std::string& errorContext)
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "getDescriptorAttrVec requires a trivially copyable type");

    int64_t count = 0;
    HIPDNN_CHECK_ERROR(getDescriptorAttrCount(desc, attrName, attrType, count, errorContext));

    if(count <= 0)
    {
        values.clear();
        return {};
    }

    values.resize(static_cast<size_t>(count));
    int64_t actualCount = 0;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(
            desc, attrName, attrType, count, &actualCount, values.data()),
        "Failed to get " + errorContext);
    if(actualCount != count)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Element count mismatch for " + errorContext + ": expected " + std::to_string(count)
                    + " but got " + std::to_string(actualCount)};
    }
    return {};
}

/// Gets a vector-valued int64 attribute (queries count first, then allocates and queries values).
[[nodiscard]] inline Error getDescriptorAttrVec(hipdnnBackendDescriptor_t desc,
                                                hipdnnBackendAttributeName_t attrName,
                                                std::vector<int64_t>& values,
                                                const std::string& errorContext)
{
    return getDescriptorAttrVec<int64_t>(desc, attrName, HIPDNN_TYPE_INT64, values, errorContext);
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

    // Clean up any remaining descriptors beyond actualCount that the backend may have populated
    for(int64_t i = actualCount; i < count; ++i)
    {
        if(rawDescs[static_cast<size_t>(i)] != nullptr)
        {
            hipdnnBackend()->backendDestroyDescriptor(rawDescs[static_cast<size_t>(i)]);
        }
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

    // Restore pass-by-value scalar if present.
    bool isByValue = false;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(tensorDesc,
                                               HIPDNN_ATTR_TENSOR_IS_BY_VALUE,
                                               HIPDNN_TYPE_BOOLEAN,
                                               isByValue,
                                               "tensor is_by_value"));
    if(isByValue)
    {
        // Read the raw bytes and dispatch on the already-known data type.
        // Pass the buffer size (8) as requestedElementCount — the backend validates
        // it is >= the data type's byte size and returns the actual count.
        std::array<uint8_t, 8> valueBytes = {};
        int64_t actualByteCount = 0;
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendGetAttribute(tensorDesc,
                                                 HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                                 HIPDNN_TYPE_CHAR,
                                                 static_cast<int64_t>(valueBytes.size()),
                                                 &actualByteCount,
                                                 valueBytes.data()),
            "Failed to get tensor pass-by-value");

        switch(dt)
        {
        case DataType::FLOAT:
        {
            float val = 0;
            std::memcpy(&val, valueBytes.data(), sizeof(float));
            tensor->set_value(val);
            break;
        }
        case DataType::DOUBLE:
        {
            double val = 0;
            std::memcpy(&val, valueBytes.data(), sizeof(double));
            tensor->set_value(val);
            break;
        }
        case DataType::HALF:
        {
            half val{};
            std::memcpy(&val, valueBytes.data(), sizeof(half));
            tensor->set_value(val);
            break;
        }
        case DataType::BFLOAT16:
        {
            bfloat16 val{};
            std::memcpy(&val, valueBytes.data(), sizeof(bfloat16));
            tensor->set_value(val);
            break;
        }
        case DataType::INT32:
        {
            int32_t val = 0;
            std::memcpy(&val, valueBytes.data(), sizeof(int32_t));
            tensor->set_value(val);
            break;
        }
        case DataType::INT64:
        {
            int64_t val = 0;
            std::memcpy(&val, valueBytes.data(), sizeof(int64_t));
            tensor->set_value(val);
            break;
        }
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        {
            const uint8_t val = valueBytes[0];
            tensor->set_value(val);
            break;
        }
        default:
            break;
        }

        // set_value() overwrites _dataType via getDataTypeEnumFromType<T>(),
        // which is wrong for types that share a C++ type (e.g. INT8, FP8_E4M3,
        // FP8_E5M2 all use uint8_t → UINT8). Restore the original data type.
        // set_value() also resets _dim and _stride to {1}. Restore the
        // dimensions and strides that were read from the backend descriptor
        // so the round-trip is symmetric with lowering.
        tensor->set_data_type(dt);
        tensor->set_dim(dims);
        tensor->set_stride(strides);
    }

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

/// Unpacks an optional tensor attribute. Returns nullptr if the attribute has no
/// elements set or is not supported by the backend.
[[nodiscard]] inline Error unpackOptionalTensor(
    hipdnnBackendDescriptor_t opDesc,
    hipdnnBackendAttributeName_t tensorAttrName,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    std::shared_ptr<graph::TensorAttributes>& outTensor,
    const std::string& errorContext)
{
    int64_t count = 0;
    auto status = hipdnnBackend()->backendGetAttribute(
        opDesc, tensorAttrName, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr);

    if(status == HIPDNN_STATUS_NOT_SUPPORTED || count <= 0)
    {
        outTensor = nullptr;
        return {};
    }
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> backendErrMsg{};
        hipdnnBackend()->getLastErrorString(backendErrMsg.data(), backendErrMsg.size());
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to query count for optional " + errorContext
                    + " Backend error: " + backendErrMsg.data()};
    }

    return unpackAndRegisterTensor(opDesc, tensorAttrName, tensorMap, outTensor, errorContext);
}

/// Gets an optional scalar attribute. Returns std::nullopt if the attribute has no
/// elements set or is not supported by the backend.
template <typename T>
[[nodiscard]] inline Error getDescriptorAttrOptionalScalar(hipdnnBackendDescriptor_t desc,
                                                           hipdnnBackendAttributeName_t attrName,
                                                           hipdnnBackendAttributeType_t attrType,
                                                           std::optional<T>& value,
                                                           const std::string& errorContext)
{
    int64_t count = 0;
    auto status
        = hipdnnBackend()->backendGetAttribute(desc, attrName, attrType, 0, &count, nullptr);
    if(status == HIPDNN_STATUS_NOT_SUPPORTED || count <= 0)
    {
        value = std::nullopt;
        return {};
    }
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> backendErrMsg{};
        hipdnnBackend()->getLastErrorString(backendErrMsg.data(), backendErrMsg.size());
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to get count for " + errorContext
                    + " Backend error: " + backendErrMsg.data()};
    }
    T raw{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(desc, attrName, attrType, raw, errorContext));
    value = raw;
    return {};
}

/// Unpacks a tensor array attribute from an operation descriptor, deduplicating
/// against the tensor map. Each tensor is either found in the map (shared) or
/// unpacked fresh and registered.
[[nodiscard]] inline Error unpackAndRegisterTensorArray(
    hipdnnBackendDescriptor_t opDesc,
    hipdnnBackendAttributeName_t tensorAttrName,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    std::vector<std::shared_ptr<graph::TensorAttributes>>& outTensors,
    const std::string& errorContext)
{
    auto [descs, descErr] = getDescriptorAttrDescArray(opDesc, tensorAttrName, errorContext);
    if(descErr.is_bad())
    {
        return descErr;
    }

    outTensors.clear();
    outTensors.reserve(descs.size());
    for(auto& scopedDesc : descs)
    {
        if(scopedDesc.get() == nullptr)
        {
            continue;
        }

        int64_t uid = 0;
        HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(scopedDesc.get(),
                                                   HIPDNN_ATTR_TENSOR_UNIQUE_ID,
                                                   HIPDNN_TYPE_INT64,
                                                   uid,
                                                   errorContext + " tensor UID"));

        auto it = tensorMap.find(uid);
        if(it != tensorMap.end())
        {
            outTensors.push_back(it->second);
        }
        else
        {
            std::shared_ptr<graph::TensorAttributes> tensor;
            HIPDNN_CHECK_ERROR(unpackTensorAttributes(scopedDesc.get(), tensor));
            tensorMap[uid] = tensor;
            outTensors.push_back(std::move(tensor));
        }
    }

    return {};
}

/// Unpacks a byte array attribute (HIPDNN_TYPE_CHAR) from a backend descriptor.
/// Returns an empty vector if the attribute is not supported or has no elements.
[[nodiscard]] inline Error getDescriptorAttrByteArray(hipdnnBackendDescriptor_t desc,
                                                      hipdnnBackendAttributeName_t attrName,
                                                      std::vector<uint8_t>& value,
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

    value.resize(static_cast<size_t>(count));
    int64_t actualCount = 0;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(
            desc, attrName, HIPDNN_TYPE_CHAR, count, &actualCount, value.data()),
        "Failed to get " + errorContext);
    if(actualCount != count)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Element count mismatch for " + errorContext + ": expected " + std::to_string(count)
                    + " but got " + std::to_string(actualCount)};
    }

    return {};
}

} // namespace hipdnn_frontend::detail
