// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "attributes/TensorAttributes.hpp"
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <numeric>
#include <vector>

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

inline constexpr const char* K_COMPONENT_NAME = "hipdnn_frontend";

// HIPDNN_HIDDEN ensures each shared object has its own copy of the static variable
HIPDNN_HIDDEN inline int32_t initializeFrontendLogging(hipdnnCallback_t fn
                                                       = hipdnnLoggingCallback_ext)
{
    if(fn == nullptr)
    {
        return -1;
    }

    static bool s_loggingInitialized = false;

    if(s_loggingInitialized)
    {
        return 0;
    }

    // Initialize log level from environment variable
    hipdnn_data_sdk::logging::initializeLogLevel();

    // Register the callback so log messages get routed to the backend
    hipdnn_data_sdk::logging::registerLoggingCallback(fn);

    s_loggingInitialized = true;

    // Use this logging macro directly to avoid re-entrant logging call.
    HIPDNN_SDK_LOG_INFO_WITH_COMPONENT(K_COMPONENT_NAME, "Frontend logging initialized");

    return 0;
}

// ============================================================================
// Frontend Logging Macros (HIPDNN_FE_LOG_*)
// ============================================================================
// These macros auto-initialize logging on first use, then log with "hipdnn_frontend"
// as the component name.
// Usage: HIPDNN_FE_LOG_INFO("Message " << value);

#define HIPDNN_FE_LOG_INFO(msg)                                                     \
    do                                                                              \
    {                                                                               \
        hipdnn_frontend::initializeFrontendLogging();                               \
        HIPDNN_SDK_LOG_INFO_WITH_COMPONENT(hipdnn_frontend::K_COMPONENT_NAME, msg); \
    } while(0)

#define HIPDNN_FE_LOG_WARN(msg)                                                     \
    do                                                                              \
    {                                                                               \
        hipdnn_frontend::initializeFrontendLogging();                               \
        HIPDNN_SDK_LOG_WARN_WITH_COMPONENT(hipdnn_frontend::K_COMPONENT_NAME, msg); \
    } while(0)

#define HIPDNN_FE_LOG_ERROR(msg)                                                     \
    do                                                                               \
    {                                                                                \
        hipdnn_frontend::initializeFrontendLogging();                                \
        HIPDNN_SDK_LOG_ERROR_WITH_COMPONENT(hipdnn_frontend::K_COMPONENT_NAME, msg); \
    } while(0)

#define HIPDNN_FE_LOG_FATAL(msg)                                                     \
    do                                                                               \
    {                                                                                \
        hipdnn_frontend::initializeFrontendLogging();                                \
        HIPDNN_SDK_LOG_FATAL_WITH_COMPONENT(hipdnn_frontend::K_COMPONENT_NAME, msg); \
    } while(0)

}
