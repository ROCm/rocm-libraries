/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <hipsparselt/hipsparselt.h>
#include <regex>
#include <string_view>

HIPSPARSELT_EXPORT
constexpr const char* hipsparse_status_to_string(hipsparseStatus_t status)
{
#define CASE(x) \
    case x:     \
        return #x
    switch(status)
    {
        CASE(HIPSPARSE_STATUS_SUCCESS);
        CASE(HIPSPARSE_STATUS_NOT_INITIALIZED);
        CASE(HIPSPARSE_STATUS_ALLOC_FAILED);
        CASE(HIPSPARSE_STATUS_INVALID_VALUE);
        CASE(HIPSPARSE_STATUS_ARCH_MISMATCH);
        CASE(HIPSPARSE_STATUS_MAPPING_ERROR);
        CASE(HIPSPARSE_STATUS_EXECUTION_FAILED);
        CASE(HIPSPARSE_STATUS_INTERNAL_ERROR);
        CASE(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
        CASE(HIPSPARSE_STATUS_ZERO_PIVOT);
        CASE(HIPSPARSE_STATUS_NOT_SUPPORTED);
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11003)
        CASE(HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES);
#endif
    }
#undef CASE
    // We don't use default: so that the compiler warns us if any valid enums are missing
    // from our switch. If the value is not a valid hipsparseStatus_t, we return this string.
    return "<undefined hipsparseStatus_t value>";
}

HIPSPARSELT_EXPORT
constexpr const char* hipsparselt_operation_to_string(hipsparseOperation_t value)
{
    switch(value)
    {
    case HIPSPARSE_OPERATION_NON_TRANSPOSE:
        return "N";
    case HIPSPARSE_OPERATION_TRANSPOSE:
        return "T";
    case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
        return "C";
    }
    return "invalid";
}

HIPSPARSELT_EXPORT
constexpr hipsparseOperation_t char_to_hipsparselt_operation(char value)
{
    switch(value)
    {
    case 'N':
    case 'n':
        return HIPSPARSE_OPERATION_NON_TRANSPOSE;
    case 'T':
    case 't':
        return HIPSPARSE_OPERATION_TRANSPOSE;
    case 'C':
    case 'c':
        return HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    default:
        return static_cast<hipsparseOperation_t>(-1);
    }
}

HIPSPARSELT_EXPORT
constexpr hipsparseOrder_t char_to_hipsparselt_order(char value)
{
    switch(value)
    {
    case 'C':
    case 'c':
        return HIPSPARSE_ORDER_COL;
    case 'R':
    case 'r':
        return HIPSPARSE_ORDER_ROW;
    default:
        return static_cast<hipsparseOrder_t>(-1);
    }
}

// return precision string for hipDataType
HIPSPARSELT_EXPORT
constexpr const char* hip_datatype_to_string(hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return "f32_r";
    case HIP_R_16F:
        return "f16_r";
    case HIP_R_16BF:
        return "bf16_r";
    case HIP_R_8I:
        return "i8_r";
#if defined(HIP_FP8_TYPE_OCP) || defined(__HIP_PLATFORM_NVIDIA__)
    case HIP_R_8F_E4M3:
        return "f8_r";
    case HIP_R_8F_E5M2:
        return "bf8_r";
#endif
    default:
        return "invalid";
    }
}

// return precision string for hipDataType
HIPSPARSELT_EXPORT
constexpr const char* hipsparselt_computetype_to_string(hipsparseLtComputetype_t type)
{
    switch(type)
    {
    case HIPSPARSELT_COMPUTE_16F:
        return "f16_r";
    case HIPSPARSELT_COMPUTE_32I:
        return "i32_r";
    case HIPSPARSELT_COMPUTE_32F:
        return "f32_r";
    default:
        return "invalid";
    }
}

// clang-format off
HIPSPARSELT_EXPORT
const hipDataType string_to_hip_datatype(const std::string& value);

HIPSPARSELT_EXPORT
const hipsparseLtComputetype_t string_to_hipsparselt_computetype(const std::string& value);

// clang-format on

/*********************************************************************************************************
 * \brief The main structure for Numerical checking to detect numerical abnormalities such as NaN/zero/Inf
 *********************************************************************************************************/
typedef struct hipsparselt_check_numerics_s
{
    // Set to true if there is a NaN in the vector/matrix
    bool has_NaN = false;

    // Set to true if there is a zero in the vector/matrix
    bool has_zero = false;

    // Set to true if there is an Infinity in the vector/matrix
    bool has_Inf = false;
} hipsparselt_check_numerics_t;

/*******************************************************************************
* \brief  returns true if arg is NaN
********************************************************************************/
template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool hipsparselt_isnan(T)
{
    return false;
}

template <typename T, std::enable_if_t<!std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool hipsparselt_isnan(T arg)
{
    return std::isnan(arg);
}

__host__ __device__ inline bool hipsparselt_isnan(__half arg)
{
    __half_raw x = __half_raw(arg);
    return (~x.x & 0x7c00) == 0 && (x.x & 0x3ff) != 0;
}

#ifdef __HIP_PLATFORM_NVIDIA__
__host__ __device__ inline bool hipsparselt_isnan(__nv_bfloat16 arg)
{
    return __hisnan(arg);
}
__host__ __device__ inline bool hipsparselt_isnan(__nv_fp8_e4m3 arg)
{
    return (static_cast<unsigned char>(arg) & 0x7f) == 0x7f;
}

__host__ __device__ inline bool hipsparselt_isnan(__nv_fp8_e5m2 arg)
{
    return (static_cast<unsigned char>(arg) & 0x7f) > 0x7c;
}
#endif

#ifdef HIP_FP8_TYPE_OCP
__host__ __device__ inline bool hipsparselt_isnan(__hip_fp8_e4m3 arg)
{
    return internal::hip_fp8_ocp_is_nan(arg, __HIP_E4M3);
}

__host__ __device__ inline bool hipsparselt_isnan(__hip_fp8_e5m2 arg)
{
    return internal::hip_fp8_ocp_is_nan(arg, __HIP_E5M2);
}
#endif
/*******************************************************************************
* \brief  returns true if arg is Infinity
********************************************************************************/

template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool hipsparselt_isinf(T)
{
    return false;
}

template <typename T, std::enable_if_t<!std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool hipsparselt_isinf(T arg)
{
    return std::isinf(arg);
}

__host__ __device__ inline bool hipsparselt_isinf(__half arg)
{
    __half_raw x = __half_raw(arg);
    return (~x.x & 0x7c00) == 0 && (x.x & 0x3ff) == 0;
}

/*******************************************************************************
* \brief  returns true if arg is zero
********************************************************************************/

template <typename T>
__host__ __device__ inline bool hipsparselt_iszero(T arg)
{
    return arg == 0;
}

/*! \brief device matches pattern */
inline
bool gpu_arch_match(std::string_view gpu_arch, std::string_view pattern)
{
    if(!pattern.length())
    {
        return true;
    }

    constexpr char    prefix[]   = "gfx";
    const std::size_t prefix_len = std::string_view(prefix).length();
    gpu_arch.remove_prefix(prefix_len);
    std::regex arch_regex(pattern.data());
    return std::regex_search(gpu_arch.data(), arch_regex);
}
