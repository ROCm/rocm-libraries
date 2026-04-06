// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file Utilities.hpp
 * @brief Helpers for creating tensor descriptors and handling backend errors
 *
 * In hipDNN, tensors passed to graph operations are described by
 * TensorAttributes — lightweight metadata objects that hold shape (dims),
 * memory layout (strides), and data type, but **not** the actual data.
 * Think of them as tensor metadata (dtype, shape, stride) without the
 * underlying storage — a descriptor, not the data itself.
 *
 * The `makeTensorAttributes()` helpers create these descriptors from
 * shapes you provide or from existing Data SDK Tensor objects.
 */

#pragma once

#include "attributes/TensorAttributes.hpp"
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <numeric>
#include <vector>

#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>

namespace hipdnn_frontend
{

/** @def HIPDNN_RETURN_ON_BACKEND_FAILURE
 *  @brief Return an Error if a backend call fails, including the backend error string
 *  @param backend_status The hipdnnStatus_t returned by the backend call
 *  @param error_message A human-readable description of the failed operation
 */
#define HIPDNN_RETURN_ON_BACKEND_FAILURE(backend_status, error_message)                           \
    do                                                                                            \
    {                                                                                             \
        if((backend_status) != HIPDNN_STATUS_SUCCESS)                                             \
        {                                                                                         \
            std::array<char, 1024> backend_err_msg{};                                             \
            hipdnn_frontend::detail::hipdnnBackend()->getLastErrorString(backend_err_msg.data(),  \
                                                                         backend_err_msg.size()); \
            const std::string full_error_msg                                                      \
                = std::string(error_message) + " Backend error: " + backend_err_msg.data();       \
            return Error(ErrorCode::HIPDNN_BACKEND_ERROR, full_error_msg);                        \
        }                                                                                         \
    } while(0)

namespace graph
{

/**
 * @brief Create TensorAttributes by copying shape and layout from an existing Tensor
 *
 * Extracts dims and strides from a Data SDK Tensor object. Useful when
 * you already have allocated test tensors and want matching descriptors.
 *
 * @tparam T Element type of the source tensor (e.g. float, half)
 * @param name Human-readable name for debugging and serialization
 * @param dataType The numeric precision (e.g. DataType::HALF)
 * @param tensor Source tensor whose dims and strides are copied
 * @return Configured TensorAttributes ready to pass to Graph operations
 */
template <class T,
          class HostAlloc = hipdnn_data_sdk::utilities::HostAllocator<T>,
          class DeviceAlloc = hipdnn_data_sdk::utilities::DeviceAllocator<T>>
inline TensorAttributes makeTensorAttributes(
    const std::string& name,
    DataType dataType,
    const hipdnn_data_sdk::utilities::Tensor<T, HostAlloc, DeviceAlloc>& tensor)
{
    return TensorAttributes()
        .set_name(name)
        .set_data_type(dataType)
        .set_dim(tensor.dims())
        .set_stride(tensor.strides());
}

/**
 * @brief Create TensorAttributes from explicit dimensions, strides, and data type
 *
 * This is the most common way to describe a tensor when you know the
 * shape and precision up front.
 *
 * @param name Human-readable name for debugging and serialization
 * @param dataType The numeric precision (e.g. DataType::FLOAT)
 * @param dims Tensor dimensions, e.g. {N, C, H, W}
 * @param strides Memory strides for each dimension
 * @return Configured TensorAttributes ready to pass to Graph operations
 */
inline TensorAttributes makeTensorAttributes(const std::string& name,
                                             DataType dataType,
                                             const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides)
{
    return TensorAttributes().set_name(name).set_data_type(dataType).set_dim(dims).set_stride(
        strides);
}

/**
 * @brief Create TensorAttributes without specifying a data type
 *
 * The data type is left unset and will be inferred from the Graph's
 * `io_data_type` at build time. Handy when all tensors in your graph
 * share the same precision.
 *
 * @param name Human-readable name for debugging and serialization
 * @param dims Tensor dimensions, e.g. {N, C, H, W}
 * @param strides Memory strides for each dimension
 * @return TensorAttributes whose data type will be filled at build time
 */
inline TensorAttributes makeTensorAttributes(const std::string& name,
                                             const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides)
{
    return TensorAttributes().set_name(name).set_dim(dims).set_stride(strides);
}

/**
 * @brief Create TensorAttributes from a single constant value
 *
 * The data type will be set from the type of the value. Useful for tensors that contain single constants, for example an epsilon.
 *
 * @param name Human-readable name for debugging and serialization
 * @param value Constant value to be inserted into the tensor
 * @return Configured TensorAttributes ready to pass to Graph operations
 */
template <typename T>
inline TensorAttributes makeTensorAttributes(const std::string& name, const T value)
{
    return TensorAttributes().set_name(name).set_value(value);
}

/**
 * @brief Convert frontend DataType to Data SDK DataType
 *
 * Maps the frontend DataType enum to the data_sdk DataType enum for tensor
 * allocation. The two enums have different numeric values, so an explicit
 * mapping is required.
 *
 * @param dt The frontend DataType value
 * @return The corresponding data_sdk DataType value, or UNSET if not mapped
 */
// Note: hipdnn_test_sdk::utilities::SdkFrontendTypeConversions.hpp has a
// parallel implementation. If new data types are added, update both.
inline hipdnn_data_sdk::data_objects::DataType frontendToSdkDataType(DataType dt)
{
    namespace data_objects = hipdnn_data_sdk::data_objects;
    switch(dt)
    {
    case DataType::FLOAT:
        return data_objects::DataType::FLOAT;
    case DataType::HALF:
        return data_objects::DataType::HALF;
    case DataType::BFLOAT16:
        return data_objects::DataType::BFLOAT16;
    case DataType::DOUBLE:
        return data_objects::DataType::DOUBLE;
    case DataType::UINT8:
        return data_objects::DataType::UINT8;
    case DataType::INT32:
        return data_objects::DataType::INT32;
    case DataType::INT8:
        return data_objects::DataType::INT8;
    case DataType::FP8_E4M3:
        return data_objects::DataType::FP8_E4M3;
    case DataType::FP8_E5M2:
        return data_objects::DataType::FP8_E5M2;
    case DataType::INT64:
        return data_objects::DataType::INT64;
    case DataType::FP8_E8M0:
        return data_objects::DataType::FP8_E8M0;
    case DataType::FP4_E2M1:
        return data_objects::DataType::FP4_E2M1;
    case DataType::INT4:
        return data_objects::DataType::INT4;
    case DataType::FP6_E2M3:
        return data_objects::DataType::FP6_E2M3;
    case DataType::FP6_E3M2:
        return data_objects::DataType::FP6_E3M2;
    default:
        return data_objects::DataType::UNSET;
    }
}

/**
 * @brief Allocate a Data SDK ITensor that matches the given attributes
 *
 * Creates an actual tensor object from a descriptor. Host memory is
 * allocated immediately; device memory is allocated lazily on first
 * access. Primarily used in tests and utilities — in production code
 * you typically manage your own device memory and just pass pointers
 * via the variant pack.
 *
 * @param attribute The tensor descriptor (type, dims, strides)
 * @return Owning pointer to the created ITensor
 */
inline std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>
    createTensorFromAttribute(const TensorAttributes& attribute)
{
    auto sdkType = frontendToSdkDataType(attribute.get_data_type());
    return hipdnn_data_sdk::utilities::createTensor(
        sdkType, attribute.get_dim(), attribute.get_stride());
}

} // namespace graph

} // namespace hipdnn_frontend
