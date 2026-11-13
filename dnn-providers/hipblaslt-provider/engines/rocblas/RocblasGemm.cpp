// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/rocblas/RocblasGemm.hpp"

#include <map>
#include <mutex>
#include <string>

#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hipblaslt_plugin::rocblas_backend
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

std::string currentGcnArchName()
{
    static std::mutex s_mutex;
    static std::map<int, std::string> s_names;
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "hipGetDevice failed");
    }
    const std::lock_guard<std::mutex> lock(s_mutex);
    auto found = s_names.find(device);
    if(found == s_names.end())
    {
        hipDeviceProp_t properties{};
        if(hipGetDeviceProperties(&properties, device) != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           "hipGetDeviceProperties failed");
        }
        found = s_names.emplace(device, properties.gcnArchName).first;
    }
    return found->second;
}

rocblas_datatype rocblasDataType(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::FLOAT:
        return rocblas_datatype_f32_r;
    case data_objects::DataType::HALF:
        return rocblas_datatype_f16_r;
    case data_objects::DataType::BFLOAT16:
        return rocblas_datatype_bf16_r;
    default:
        break;
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "rocBLAS GEMM does not support element type "
            + std::string(data_objects::EnumNameDataType(dataType)));
}

ScopedXf32::ScopedXf32(rocblas_handle handle)
    : _handle(handle)
{
    if(rocblas_get_math_mode(_handle, &_previous) == rocblas_status_success)
    {
        _set = rocblas_set_math_mode(_handle, rocblas_xf32_xdl_math_op) == rocblas_status_success;
    }
}

ScopedXf32::~ScopedXf32()
{
    if(_set)
    {
        static_cast<void>(rocblas_set_math_mode(_handle, _previous));
    }
}

bool permitsXf32(const GemmCall& call)
{
    return call.tf32 && call.a.dataType == data_objects::DataType::FLOAT
           && call.b.dataType == data_objects::DataType::FLOAT;
}

void runGemm(rocblas_handle handle, const GemmCall& call, const void* a, const void* b, void* d)
{

    // Host-side scalars, f32 to match the compute type.
    const float alpha = 1.0f;
    const float beta = call.accumulate ? 1.0f : 0.0f;
    const auto dType = rocblasDataType(call.d.dataType);

    // Parenthesized: rocblas.h wraps the function in a variadic compatibility macro for its
    // obsolete workspace arguments, which -Wvariadic-macro-arguments-omitted rejects when they
    // are left out.
    const rocblas_status status
        = (rocblas_gemm_strided_batched_ex)(handle,
                                            call.transposeA ? rocblas_operation_transpose
                                                            : rocblas_operation_none,
                                            call.transposeB ? rocblas_operation_transpose
                                                            : rocblas_operation_none,
                                            static_cast<rocblas_int>(call.m),
                                            static_cast<rocblas_int>(call.n),
                                            static_cast<rocblas_int>(call.k),
                                            &alpha,
                                            a,
                                            rocblasDataType(call.a.dataType),
                                            static_cast<rocblas_int>(call.a.leadingDimension),
                                            call.a.batchStride,
                                            b,
                                            rocblasDataType(call.b.dataType),
                                            static_cast<rocblas_int>(call.b.leadingDimension),
                                            call.b.batchStride,
                                            &beta,
                                            d, // c is d: read only when accumulating
                                            dType,
                                            static_cast<rocblas_int>(call.d.leadingDimension),
                                            call.d.batchStride,
                                            d,
                                            dType,
                                            static_cast<rocblas_int>(call.d.leadingDimension),
                                            call.d.batchStride,
                                            static_cast<rocblas_int>(call.batchCount),
                                            rocblas_datatype_f32_r,
                                            rocblas_gemm_algo_standard,
                                            0,
                                            rocblas_gemm_flags_none);

    if(status != rocblas_status_success)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string("rocblas_gemm_strided_batched_ex failed: ")
                + rocblas_status_to_string(status));
    }
}

} // namespace hipblaslt_plugin::rocblas_backend
