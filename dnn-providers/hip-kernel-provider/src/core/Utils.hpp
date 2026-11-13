// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>

/**
 * @brief Macro that returns and prints info log on passing provided condition.
 * Arguments after first are passed into std::format().
 * Requires a HIP_KERNEL_LOG_PREFIX in scope which will prefix the log.
 *
 * `message` is intentionally substituted without enclosing parentheses so call
 * sites can pass `+`-chained expressions like `"foo " + EnumName(...)` that
 * rely on left-to-right associativity to fold into a std::string.
 */
#define HIP_KERNEL_RETURN_FALSE_IF(condition, message)                            \
    do                                                                            \
    {                                                                             \
        if(condition)                                                             \
        {                                                                         \
            /* NOLINTNEXTLINE(bugprone-macro-parentheses) */                      \
            HIPDNN_PLUGIN_LOG_INFO(std::string{HIP_KERNEL_LOG_PREFIX} + message); \
            return false;                                                         \
        }                                                                         \
    } while(0)

/**
 * @brief Applicability-check convention.
 *
 * A check returns hip_kernel_provider::core::utils::Rejection: std::nullopt when the
 * graph passes, otherwise the reason the engine declines it. Checks compose with
 * HIP_KERNEL_RETURN_IF_REJECTED and the top-level isApplicable() logs the reason once,
 * so no check expands a logging macro of its own.
 *
 * HIP_KERNEL_REJECT_IF returns `message` from a Rejection-returning function when
 * `condition` holds. `message` follows HIP_KERNEL_RETURN_FALSE_IF's rule: it is
 * substituted unparenthesised after `std::string{} +` so `+`-chained pieces fold into
 * a std::string left to right.
 */
#define HIP_KERNEL_REJECT_IF(condition, message)                                           \
    do                                                                                     \
    {                                                                                      \
        if(condition)                                                                      \
        {                                                                                  \
            /* NOLINTNEXTLINE(bugprone-macro-parentheses) */                               \
            return ::hip_kernel_provider::core::utils::Rejection{std::string{} + message}; \
        }                                                                                  \
    } while(0)

/**
 * @brief Forwards a sub-check's Rejection to the caller (which must return Rejection).
 */
#define HIP_KERNEL_RETURN_IF_REJECTED(check)  \
    do                                        \
    {                                         \
        if(auto hipKernelRejection = (check)) \
        {                                     \
            return hipKernelRejection;        \
        }                                     \
    } while(0)

namespace hip_kernel_provider::core::utils
{

// Why an engine declines a graph: std::nullopt when applicable, otherwise the reason.
using Rejection = std::optional<std::string>;

enum class ActivationMode : int
{
    PASTHRU = 0,
    LOGISTIC = 1, // sigmoid
    TANH = 2,
    RELU = 3,
    SOFTRELU = 4, // softplus
    ABS = 5,
    POWER = 6,
    CLIPPED_RELU = 7,
    LEAKY_RELU = 8,
    ELU = 9,
    CLAMP = 10
};

struct ActivationParams
{
    ActivationMode mode;
    double alpha;
    double beta;
    double gamma;
};

ActivationParams
    parseActivation(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attrs);

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& findTensorAttributes(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid);

bool isChannelLastLayout(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tensor);

// Provider kernels assume buffers aligned to a multiple of 16 bytes.
bool hasOnly16ByteAlignedTensors(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

} // namespace hip_kernel_provider::core::utils
