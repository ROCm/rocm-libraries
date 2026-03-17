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
            std::string full_error_msg                                                            \
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
 * @brief Allocate a Data SDK ITensor that matches the given attributes
 *
 * Creates an actual tensor object (with host/device storage) from a
 * descriptor. Primarily used in tests and utilities — in production
 * code you typically manage your own device memory and just pass
 * pointers via the variant pack.
 *
 * @param attribute The tensor descriptor (type, dims, strides)
 * @return Owning pointer to the created ITensor
 */
inline std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>
    createTensorFromAttribute(const TensorAttributes& attribute)
{
    return hipdnn_data_sdk::utilities::createTensor(
        toSdkType(attribute.get_data_type()), attribute.get_dim(), attribute.get_stride());
}

} // namespace graph

} // namespace hipdnn_frontend
