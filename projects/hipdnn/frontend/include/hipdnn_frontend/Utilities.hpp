// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Error.hpp"
#include "attributes/TensorAttributes.hpp"
#include "node/Node.hpp"
#include <algorithm>
#include <hipdnn_backend.h>
#include <hipdnn_sdk/logging/CallbackTypes.h>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/logging/LoggingUtils.hpp>
#include <hipdnn_sdk/test_utilities/LoggingUtils.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <numeric>
#include <ranges>
#include <vector>

namespace hipdnn_frontend
{
namespace graph
{
// Find common shape from inputs.
// Takes the max in each dim, and if any dim is not 1, or equal, then it's incompatible.
// For example:
// input_shapes = {{1, 2}, {1, 2}, {1, 2, 5}} -> common_shape = {1, 2, 5}
// input_shapes = {{1, 2, 3}, {1, 2, 4}, {1, 2}} -> error
inline Error findCommonShape(const std::vector<std::vector<int64_t>>& inputShapes,
                             std::vector<int64_t>& commonShape)
{
    if(inputShapes.empty())
    {
        return {ErrorCode::INVALID_VALUE, "Input shapes cannot be empty"};
    }

    size_t dims
        = std::max_element(inputShapes.begin(),
                           inputShapes.end(),
                           [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
                               return a.size() < b.size();
                           })
              ->size();

    commonShape.resize(dims, 1);

    for(auto& current : inputShapes)
    {
        for(size_t j = current.size(); j-- > 0;)
        {
            if(commonShape[j] != current[j] && commonShape[j] != 1 && current[j] != 1)
            {
                return {ErrorCode::INVALID_VALUE, "Incompatible shapes"};
            }

            commonShape[j] = std::max(commonShape[j], current[j]);
        }
    }

    return {};
}

// Utility function to create Tensor_attributes from a Tensor
template <class T,
          class HostAlloc = hipdnn_sdk::utilities::HostAllocator<T>,
          class DeviceAlloc = hipdnn_sdk::utilities::DeviceAllocator<T>>
inline TensorAttributes
    makeTensorAttributes(const std::string& name,
                         DataType dataType,
                         const hipdnn_sdk::utilities::Tensor<T, HostAlloc, DeviceAlloc>& tensor)
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

inline std::unique_ptr<hipdnn_sdk::utilities::ITensor>
    createTensorFromAttribute(const TensorAttributes& attribute)
{
    switch(attribute.get_data_type())
    {
    case DataType::FLOAT:
        return std::make_unique<hipdnn_sdk::utilities::Tensor<float>>(attribute.get_dim(),
                                                                      attribute.get_stride());
    case DataType::HALF:
        return std::make_unique<hipdnn_sdk::utilities::Tensor<half>>(attribute.get_dim(),
                                                                     attribute.get_stride());
    case DataType::BFLOAT16:
        return std::make_unique<hipdnn_sdk::utilities::Tensor<hip_bfloat16>>(
            attribute.get_dim(), attribute.get_stride());
    case DataType::DOUBLE:
        return std::make_unique<hipdnn_sdk::utilities::Tensor<double>>(attribute.get_dim(),
                                                                       attribute.get_stride());
    case DataType::UINT8:
        return std::make_unique<hipdnn_sdk::utilities::Tensor<uint8_t>>(attribute.get_dim(),
                                                                        attribute.get_stride());
    case DataType::INT32:
        return std::make_unique<hipdnn_sdk::utilities::Tensor<int32_t>>(attribute.get_dim(),
                                                                        attribute.get_stride());
    default:
        throw std::runtime_error("Unsupported data type for tensor");
    }
}

// Visit a tree of INodes with a lambda function
// Performs pre-order traversal (visits parent before children)
// Example usage:
//   visitTree(rootNode, [](INode& node) {
//       // Process node
//   });
template <typename Func>
inline void visitGraph(INode& root, Func&& visitor)
{
    // Visit current node first (pre-order traversal)
    visitor(root);

    // Then visit all children
    for(const auto& child : root.getSubNodes())
    {
        if(child)
        {
            visitGraph(*child, std::forward<Func>(visitor));
        }
    }
}

// Overload for shared_ptr
template <typename Func>
inline void visitGraph(const std::shared_ptr<INode>& root, Func&& visitor)
{
    if(root)
    {
        visitGraph(*root, std::forward<Func>(visitor));
    }
}

}

inline int32_t initializeFrontendLogging(hipdnnCallback_t fn = hipdnnLoggingCallback_ext)
{
    if(fn == nullptr)
    {
        return -1;
    }

    static bool s_loggingInitialized = false;
    static bool s_loggingEnabled = hipdnn_sdk::logging::isLoggingEnabled();

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
