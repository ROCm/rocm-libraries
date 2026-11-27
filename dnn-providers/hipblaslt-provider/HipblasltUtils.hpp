// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <hipblaslt/hipblaslt.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <string>
#include <vector>

// Each macro evaluates its argument exactly once, so a wrapped call is never re-issued and the
// message names the status of the first call.
#define LOG_ON_HIPBLASLT_FAILURE(status)                                                         \
    do                                                                                           \
    {                                                                                            \
        const hipblasStatus_t hipblasltMacroStatus = (status);                                   \
        if(hipblasltMacroStatus != HIPBLAS_STATUS_SUCCESS)                                       \
        {                                                                                        \
            HIPDNN_PLUGIN_LOG_ERROR("hipBLASLt error occurred: "                                 \
                                    << hipblaslt_plugin::hipblaslt_utils::hipblasStatusToString( \
                                           hipblasltMacroStatus));                               \
        }                                                                                        \
    } while(0)

#define THROW_ON_HIPBLASLT_FAILURE(status)                                                  \
    do                                                                                      \
    {                                                                                       \
        const hipblasStatus_t hipblasltMacroStatus = (status);                              \
        if(hipblasltMacroStatus != HIPBLAS_STATUS_SUCCESS)                                  \
        {                                                                                   \
            throw hipdnn_plugin_sdk::HipdnnPluginException(                                 \
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,                                        \
                "hipBLASLt error occurred: "                                                \
                    + std::string(hipblaslt_plugin::hipblaslt_utils::hipblasStatusToString( \
                        hipblasltMacroStatus)));                                            \
        }                                                                                   \
    } while(0)

#define THROW_ON_HIP_FAILURE(status)                                                    \
    do                                                                                  \
    {                                                                                   \
        const hipError_t hipStatusValue = (status);                                     \
        if(hipStatusValue != hipSuccess)                                                \
        {                                                                               \
            throw hipdnn_plugin_sdk::HipdnnPluginException(                             \
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,                                    \
                "HIP error occurred: " + std::string(hipGetErrorName(hipStatusValue))); \
        }                                                                               \
    } while(0)

#define HIPDNN_PREPEND_MESSAGE_ON_THROW(statement, message)                                 \
    do                                                                                      \
    {                                                                                       \
        try                                                                                 \
        {                                                                                   \
            statement;                                                                      \
        }                                                                                   \
        catch(hipdnn_plugin_sdk::HipdnnPluginException error)                               \
        {                                                                                   \
            throw hipdnn_plugin_sdk::HipdnnPluginException(error.getStatus(),               \
                                                           (message) + error.getMessage()); \
        }                                                                                   \
    } while(0)

namespace hipblaslt_plugin::hipblaslt_utils
{

// Matches hipMalloc's 256-byte alignment guarantee; regions carved out of a
// caller's workspace are aligned to it so each is as aligned as an allocation.
constexpr size_t WORKSPACE_ALIGNMENT = 256;

constexpr size_t alignUp(size_t value, size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

inline const char* hipblasStatusToString(hipblasStatus_t status)
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

using hipdnn_plugin_sdk::findDeviceBuffer;

hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper findTensorAttributes(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid);

// Storage order of the matrix formed by the last two dims. A size-1 dim has no
// meaningful stride, so a matrix is row-major when dims[rank-1] == 1 or
// strides[rank-1] == 1; otherwise column-major when dims[rank-2] == 1 or
// strides[rank-2] == 1; otherwise unsupported.
enum class MatrixOrder
{
    ROW_MAJOR,
    COLUMN_MAJOR,
    UNSUPPORTED
};

MatrixOrder matrixOrder(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides);

// Leading dimension hipBLASLt uses for the matrix formed by the last two dims:
// the row stride of a row-major matrix (the column count when the matrix has a
// single row) or the column stride of a column-major matrix. The stride of a
// size-1 dim is ignored. Throws BAD_PARAM for an unsupported layout.
int64_t leadingDimension(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tensor);

// hipBLASLt transpose op for an input matrix: row-major -> HIPBLAS_OP_N,
// column-major -> HIPBLAS_OP_T. Throws BAD_PARAM for any other layout.
hipblasOperation_t operandTranspose(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tensor);

// Throws BAD_PARAM unless the output matrix is row-major and its rows do not overlap:
// with more than one row, the leading dimension must be at least the column count.
// A single-row output keeps any stride.
void requireRowMajorOutput(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tensor);

/// True when `strides` describe dense row-major data for `dims` (innermost dimension
/// contiguous, each outer stride the product of the inner extents). The stride of a
/// size-1 dimension does not affect the layout and is ignored.
bool isDenseRowMajor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides);

bool isTypeFp8Ocp(const hipdnn_flatbuffers_sdk::data_objects::DataType& dataType);

bool isTypeFp6Ocp(const hipdnn_flatbuffers_sdk::data_objects::DataType& dataType);

bool isTypeMxOcp(const hipdnn_flatbuffers_sdk::data_objects::DataType& dataType);

} // namespace hipblaslt_plugin::hipblaslt_utils
