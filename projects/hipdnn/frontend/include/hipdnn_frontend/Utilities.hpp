// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "attributes/TensorAttributes.hpp"
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/logging/LoggingUtils.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <vector>

namespace hipdnn_frontend
{

// Utility function to create Tensor_attributes from a Tensor
template <class T,
          class HostAlloc = hipdnn_data_sdk::utilities::HostAllocator<T>,
          class DeviceAlloc = hipdnn_data_sdk::utilities::DeviceAllocator<T>>
inline graph::TensorAttributes makeTensorAttributes(
    const std::string& name,
    DataType dataType,
    const hipdnn_data_sdk::utilities::Tensor<T, HostAlloc, DeviceAlloc>& tensor)
{
    return graph::TensorAttributes()
        .set_name(name)
        .set_data_type(dataType)
        .set_dim(tensor.dims())
        .set_stride(tensor.strides());
}

inline graph::TensorAttributes makeTensorAttributes(const std::string& name,
                                                    DataType dataType,
                                                    const std::vector<int64_t>& dims,
                                                    const std::vector<int64_t>& strides)
{
    return graph::TensorAttributes()
        .set_name(name)
        .set_data_type(dataType)
        .set_dim(dims)
        .set_stride(strides);
}

inline graph::TensorAttributes makeTensorAttributes(const std::string& name,
                                                    const std::vector<int64_t>& dims,
                                                    const std::vector<int64_t>& strides)
{
    return graph::TensorAttributes().set_name(name).set_dim(dims).set_stride(strides);
}

inline std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>
    createTensorFromAttribute(const graph::TensorAttributes& attribute)
{
    return hipdnn_data_sdk::utilities::createTensor(
        toSdkType(attribute.get_data_type()), attribute.get_dim(), attribute.get_stride());
}

inline int32_t initializeFrontendLogging(hipdnnCallback_t fn = hipdnnLoggingCallback_ext)
{
    if(fn == nullptr)
    {
        return -1;
    }

    static bool s_loggingInitialized = false;
    static bool s_loggingEnabled = hipdnn_data_sdk::logging::isLoggingEnabled();

    if(s_loggingInitialized || !s_loggingEnabled)
    {
        return 0;
    }

#ifdef COMPONENT_NAME
    hipdnn::logging::initializeCallbackLogging(COMPONENT_NAME, fn);
#else
    return -1;
#endif

    s_loggingInitialized = true;
    HIPDNN_LOG_INFO("Frontend logging initialized via callback.");

    return 0;
}

#define HIPDNN_FE_LOG_INFO(...)                       \
    do                                                \
    {                                                 \
        hipdnn_frontend::initializeFrontendLogging(); \
        HIPDNN_LOG_INFO(__VA_ARGS__);                 \
    } while(0)

#define HIPDNN_FE_LOG_WARN(...)                       \
    do                                                \
    {                                                 \
        hipdnn_frontend::initializeFrontendLogging(); \
        HIPDNN_LOG_WARN(__VA_ARGS__);                 \
    } while(0)

#define HIPDNN_FE_LOG_ERROR(...)                      \
    do                                                \
    {                                                 \
        hipdnn_frontend::initializeFrontendLogging(); \
        HIPDNN_LOG_ERROR(__VA_ARGS__);                \
    } while(0)

#define HIPDNN_FE_LOG_FATAL(...)                      \
    do                                                \
    {                                                 \
        hipdnn_frontend::initializeFrontendLogging(); \
        HIPDNN_LOG_FATAL(__VA_ARGS__);                \
    } while(0)

}
