// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
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

// When an error occurs, get the backend error string and append it to the error_message.
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

// Utility function to create Tensor_attributes from a Tensor
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

inline TensorAttributes makeTensorAttributes(const std::string& name,
                                             DataType dataType,
                                             const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides)
{
    return TensorAttributes().set_name(name).set_data_type(dataType).set_dim(dims).set_stride(
        strides);
}

inline TensorAttributes makeTensorAttributes(const std::string& name,
                                             const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides)
{
    return TensorAttributes().set_name(name).set_dim(dims).set_stride(strides);
}

inline std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>
    createTensorFromAttribute(const TensorAttributes& attribute)
{
    return hipdnn_data_sdk::utilities::createTensor(
        toSdkType(attribute.get_data_type()), attribute.get_dim(), attribute.get_stride());
}

} // namespace graph
} // namespace hipdnn_frontend
