// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipblaslt/hipblaslt.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <string>

#define LOG_ON_HIPBLASLT_FAILURE(status)                                                 \
    do                                                                                   \
    {                                                                                    \
        if(status != HIPBLAS_STATUS_SUCCESS)                                             \
        {                                                                                \
            HIPDNN_PLUGIN_LOG_ERROR(                                                     \
                "hipBLASLt error occurred: "                                             \
                << hipblaslt_plugin::hipblaslt_utils::hipblas_status_to_string(status)); \
        }                                                                                \
    } while(0)

#define THROW_ON_HIPBLASLT_FAILURE(status)                                                     \
    do                                                                                         \
    {                                                                                          \
        if(status != HIPBLAS_STATUS_SUCCESS)                                                   \
        {                                                                                      \
            throw hipdnn_plugin_sdk::HipdnnPluginException(                                    \
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,                                           \
                "hipBLASLt error occurred: "                                                   \
                    + std::string(                                                             \
                        hipblaslt_plugin::hipblaslt_utils::hipblas_status_to_string(status))); \
        }                                                                                      \
    } while(0)

#define HIPDNN_PREPEND_MESSAGE_ON_THROW(statement, message)                               \
    do                                                                                    \
    {                                                                                     \
        try                                                                               \
        {                                                                                 \
            statement;                                                                    \
        }                                                                                 \
        catch(hipdnn_plugin_sdk::HipdnnPluginException error)                             \
        {                                                                                 \
            throw hipdnn_plugin_sdk::HipdnnPluginException(error.getStatus(),             \
                                                           message + error.getMessage()); \
        }                                                                                 \
    } while(0)

namespace hipblaslt_plugin::hipblaslt_utils
{

inline const char* hipblas_status_to_string(hipblasStatus_t status)
{
#define CASE(x) \
    case x:     \
        return #x
    switch(status)
    {
        CASE(HIPBLAS_STATUS_SUCCESS);
        CASE(HIPBLAS_STATUS_NOT_INITIALIZED);
        CASE(HIPBLAS_STATUS_ALLOC_FAILED);
        CASE(HIPBLAS_STATUS_INVALID_VALUE);
        CASE(HIPBLAS_STATUS_MAPPING_ERROR);
        CASE(HIPBLAS_STATUS_EXECUTION_FAILED);
        CASE(HIPBLAS_STATUS_INTERNAL_ERROR);
        CASE(HIPBLAS_STATUS_NOT_SUPPORTED);
        CASE(HIPBLAS_STATUS_ARCH_MISMATCH);
        CASE(HIPBLAS_STATUS_INVALID_ENUM);
        CASE(HIPBLAS_STATUS_UNKNOWN);
        CASE(HIPBLAS_STATUS_HANDLE_IS_NULLPTR);
    default:
        return "<undefined hipblasStatus_t value>";
    }
#undef CASE
}

struct EpilogueParams
{
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    float act0 = 0;
    float act1 = 0;
};

EpilogueParams mapPointwiseModeToHipblasLtEpilogue(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* attrs, bool withBias);

hipDataType
    tensorDataTypeToHipDataType(const hipdnn_flatbuffers_sdk::data_objects::DataType& dataType);

hipdnnPluginDeviceBuffer_t findDeviceBuffer(int64_t uid,
                                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                            uint32_t numDeviceBuffers);

hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper findTensorAttributes(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid);

} // namespace hipblaslt_plugin::hipblaslt_utils
