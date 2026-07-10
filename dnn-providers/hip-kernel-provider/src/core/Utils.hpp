// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

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

namespace hip_kernel_provider::core::utils
{

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

hipdnnPluginDeviceBuffer_t findDeviceBuffer(int64_t uid,
                                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                            uint32_t numDeviceBuffers);

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& findTensorAttributes(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid);

bool isChannelLastLayout(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tensor);

/// @brief A scalar tensor operand (epsilon/momentum) resolved either at plan-build
/// (compile-time constant or runtime-with-default) or at execute (pure runtime
/// user-supplied, i.e. is_runtime_pass_by_value() && value_type() == NONE).
struct ScalarOperand
{
    int64_t uid = 0;
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    bool isRuntimeUserSupplied = false;
    double bakedDefault = 0.0;
};

/// @brief Builds a ScalarOperand from the op-graph tensor at plan-build time.
/// Pure user-supplied tensors (is_runtime_pass_by_value() && value_type()==NONE)
/// record uid+dtype only, deferring the read to execute. Every other state
/// (compile-time constant, or runtime-with-default) extracts the baked value now.
ScalarOperand makeScalarOperand(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid,
    const char* paramName);

/// @brief Resolves a ScalarOperand at execute time. Pure user-supplied operands
/// read the host scalar from the matching device_buffers slot (throws
/// HIPDNN_PLUGIN_STATUS_INVALID_VALUE if absent); all other operands return the
/// baked default and ignore any device_buffers slot for that uid.
double resolveScalarOperand(const ScalarOperand& op,
                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                            uint32_t numDeviceBuffers);

} // namespace hip_kernel_provider::core::utils
