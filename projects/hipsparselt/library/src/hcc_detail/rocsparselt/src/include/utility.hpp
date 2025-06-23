/*! \file */
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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
#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "auxiliary.hpp"
#include "handle.h"
#include "logging.h"
#include <algorithm>
#include <exception>

#pragma STDC CX_LIMITED_RANGE ON

#ifdef __cplusplus
extern "C" {
#endif
hipsparseOperation_t     HCCOperationToHIPOperation(rocsparselt_operation_ op);
#ifdef __cplusplus
}
#endif

inline bool isAligned(const void* pointer, size_t byte_count)
{
    return reinterpret_cast<uintptr_t>(pointer) % byte_count == 0;
}

constexpr const char* rocsparselt_transpose_letter(rocsparselt_operation op)
{
    return hipsparselt_operation_to_string(HCCOperationToHIPOperation(op));
}

template <typename>
static constexpr char rocsparselt_precision_string[] = "invalid";
template <>
static constexpr char rocsparselt_precision_string<hip_bfloat16>[] = "bf16_r";
template <>
static constexpr char rocsparselt_precision_string<__half>[] = "f16_r";
template <>
static constexpr char rocsparselt_precision_string<float>[] = "f32_r";
template <>
static constexpr char rocsparselt_precision_string<double>[] = "f64_r";
template <>
static constexpr char rocsparselt_precision_string<int8_t>[] = "i8_r";
template <>
static constexpr char rocsparselt_precision_string<uint8_t>[] = "u8_r";
template <>
static constexpr char rocsparselt_precision_string<int32_t>[] = "i32_r";
template <>
static constexpr char rocsparselt_precision_string<uint32_t>[] = "u32_r";
template <>
static constexpr char rocsparselt_precision_string<__hip_fp8_e4m3>[] = "f8_r";
template <>
static constexpr char rocsparselt_precision_string<__hip_fp8_e5m2>[] = "bf8_r";

std::string prefix(const char* layer, const char* caller);

const char* rocsparselt_compute_type_to_string(rocsparselt_compute_type type);

const char* rocsparselt_order_to_string(rocsparselt_order order);

const char* rocsparselt_operation_to_string(rocsparselt_operation op);

const char* rocsparselt_sparsity_to_string(rocsparselt_sparsity sparsity);

const char* rocsparselt_matrix_type_to_string(rocsparselt_matrix_type type);

const char* rocsparselt_layer_mode2string(rocsparselt_layer_mode layer_mode);

const char* rocsparselt_activation_type_to_string(rocsparselt_matmul_descr_attribute type);

// if trace logging is turned on with
// (handle->layer_mode & rocsparselt_layer_mode_log_trace) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_base(const _rocsparselt_handle* handle,
              rocsparselt_layer_mode     layer_mode,
              const char*                func,
              H                          head,
              Ts&&... xs)
{
    if(nullptr != handle && nullptr != handle->log_trace_os)
    {
        if(handle->layer_mode & layer_mode)
        {
            std::string comma_separator = ",";

            std::ostream* os = handle->log_trace_os;

            std::string prefix_str = prefix(rocsparselt_layer_mode2string(layer_mode), func);

            log_arguments(*os, comma_separator, prefix_str, head, std::forward<Ts>(xs)...);
        }
    }
}

// if trace logging is turned on with
// (handle->layer_mode & rocsparselt_layer_mode_log_error) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_error(const _rocsparselt_handle* handle, const char* func, H head, Ts&&... xs)
{
    log_base(handle, rocsparselt_layer_mode_log_error, func, head, std::forward<Ts>(xs)...);
}

// if trace logging is turned on with
// (handle->layer_mode & rocsparselt_layer_mode_log_trace) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_trace(const _rocsparselt_handle* handle, const char* func, H head, Ts&&... xs)
{
    log_base(handle, rocsparselt_layer_mode_log_trace, func, head, std::forward<Ts>(xs)...);
}

// if trace logging is turned on with
// (handle->layer_mode & rocsparselt_layer_mode_log_hints) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_hints(const _rocsparselt_handle* handle, const char* func, H head, Ts&&... xs)
{
    log_base(handle, rocsparselt_layer_mode_log_hints, func, head, std::forward<Ts>(xs)...);
}

// if trace logging is turned on with
// (handle->layer_mode & rocsparselt_layer_mode_log_info) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_info(const _rocsparselt_handle* handle, const char* func, H head, Ts&&... xs)
{
    log_base(handle, rocsparselt_layer_mode_log_info, func, head, std::forward<Ts>(xs)...);
}

// if trace logging is turned on with
// (handle->layer_mode & rocsparselt_layer_mode_log_api) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_api(const _rocsparselt_handle* handle, const char* func, H head, Ts&&... xs)
{
    log_base(handle, rocsparselt_layer_mode_log_api, func, head, std::forward<Ts>(xs)...);
}

// if bench logging is turned on with
// (handle->layer_mode & rocsparselt_layer_mode_log_bench) == true
// then
// log_bench will call log_arguments to log a string that
// can be input to the executable rocsparselt-bench.
template <typename H, typename... Ts>
void log_bench(
    const _rocsparselt_handle* handle, const char* func, H head, std::string precision, Ts&&... xs)
{
    if(nullptr != handle && nullptr != handle->log_bench_os)
    {
        if(handle->log_bench)
        {
            std::string comma_separator = " ";

            std::ostream* os = handle->log_bench_os;

            std::string prefix_str = prefix("Bench", func);

            log_arguments(*os, comma_separator, prefix_str, head, std::forward<Ts>(xs)...);
        }
    }
}

template <typename... Ts>
inline std::string concatenate(Ts&&... vals)
{
    std::ostringstream msg;
    std::string        none_separator = "";
    each_args(log_arg{msg, none_separator}, std::forward<Ts>(vals)...);
    return msg.str();
}

template <bool T_Enable, typename... Ts>
inline std::string concatenate_if(Ts&&... vals)
{
    if(!T_Enable)
        return "";
    return concatenate(std::forward<Ts>(vals)...);
}

// Return the value category for a value, as a double precision value, such
// such as whether it's 0, 1, -1 or some other value. Tensile uses a double
// precision value to express the category of beta. This function is to
// convert complex or other types to a double representing the category.
template <typename T>
constexpr double value_category(const T& beta)
{
    return beta == T(0) ? 0.0 : beta == T(1) ? 1.0 : beta == T(-1) ? -1.0 : 2.0;
}
#endif // UTILITY_H
