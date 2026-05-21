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
 * shapes you provide.
 */

#pragma once

#include "attributes/TensorAttributes.hpp"
#include <hipdnn_backend.h>
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
 * @brief Create TensorAttributes by copying shape and layout from an existing tensor-like object
 *
 * Extracts dims and strides from any object that provides `.dims()` and
 * `.strides()` methods returning containers of int64_t. Works with
 * Data SDK Tensor objects and any compatible type without pulling in
 * their headers.
 *
 * @tparam TensorLike Any type with `.dims()` and `.strides()` methods
 * @param name Human-readable name for debugging and serialization
 * @param dataType The numeric precision (e.g. DataType::FLOAT)
 * @param tensor Source object whose dims and strides are copied
 * @return Configured TensorAttributes ready to pass to Graph operations
 */
template <typename TensorLike>
inline TensorAttributes
    makeTensorAttributes(const std::string& name, DataType dataType, const TensorLike& tensor)
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

} // namespace graph

} // namespace hipdnn_frontend
