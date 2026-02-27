// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file Utilities.hpp
 * @brief Frontend utility functions and macros
 *
 * Provides helper functions for creating TensorAttributes from Data SDK
 * tensor objects, and the HIPDNN_RETURN_ON_BACKEND_FAILURE error-handling
 * macro used throughout the frontend.
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
 * @brief Create TensorAttributes from a Data SDK Tensor object
 * @tparam T Element type of the source tensor
 * @param name Name to assign to the tensor
 * @param dataType Frontend data type for the tensor
 * @param tensor Source tensor whose dims and strides are copied
 * @return Configured TensorAttributes
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
 * @param name Name to assign to the tensor
 * @param dataType Frontend data type for the tensor
 * @param dims Tensor dimensions
 * @param strides Tensor strides
 * @return Configured TensorAttributes
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
 * @brief Create TensorAttributes from explicit dimensions and strides (data type from graph context)
 * @param name Name to assign to the tensor
 * @param dims Tensor dimensions
 * @param strides Tensor strides
 * @return Configured TensorAttributes (data type will be filled from graph context)
 */
inline TensorAttributes makeTensorAttributes(const std::string& name,
                                             const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides)
{
    return TensorAttributes().set_name(name).set_dim(dims).set_stride(strides);
}

/**
 * @brief Create a Data SDK ITensor from TensorAttributes
 * @param attribute The tensor attributes describing type, dims, and strides
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
