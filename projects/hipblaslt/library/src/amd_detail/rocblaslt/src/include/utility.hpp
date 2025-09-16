/*! \file */
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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

// Why doesn't this header use a namespace?

#pragma once
#ifndef ROCBLASLT_UTILITY_HPP
#define ROCBLASLT_UTILITY_HPP

#include "handle.h"
#include "logging.h"
#include <algorithm>
#include <exception>
#include <mutex>

#pragma STDC CX_LIMITED_RANGE ON

static std::mutex log_mutex;

inline bool isAligned(const void* pointer, size_t byte_count)
{
    return reinterpret_cast<uintptr_t>(pointer) % byte_count == 0;
}

bool rocblaslt_is_complex_datatype(hipDataType type);

constexpr const char* rocblaslt_compute_type_string(rocblaslt_compute_type type)
{
    switch(type)
    {
    case rocblaslt_compute_f16:
        return "f16_r";
    case rocblaslt_compute_f32:
        return "f32_r";
    case rocblaslt_compute_f32_fast_xf32:
        return "xf32_r";
    case rocblaslt_compute_i32:
        return "i32_r";
    case rocblaslt_compute_f64:
        return "f64_r";
    case rocblaslt_compute_f32_fast_f16:
        return "f32_f16_r";
    case rocblaslt_compute_f32_fast_bf16:
        return "f32_bf16_r";
    case rocblaslt_compute_f32_fast_f8:
        return "f32_f8_r";
    case rocblaslt_compute_f32_fast_f8_fnuz:
        return "f32_f8_fnuz_r";
    case rocblaslt_compute_f32_fast_bf8:
        return "f32_bf8_fnuz_r";
    case rocblaslt_compute_f32_fast_bf8_fnuz:
        return "f32_bf8_r";
    case rocblaslt_compute_f32_fast_f8bf8:
        return "f32_f8bf8_r";
    case rocblaslt_compute_f32_fast_f8bf8_fnuz:
        return "f32_f8bf8_fnuz_r";
    case rocblaslt_compute_f32_fast_bf8f8:
        return "f32_bf8f8_r";
    case rocblaslt_compute_f32_fast_bf8f8_fnuz:
        return "f32_bf8f8_fnuz_r";
    default:
        return "invalidType";
    }
}

template <typename>
constexpr inline char rocblaslt_precision_string[] = "invalid";
template <>
constexpr inline char rocblaslt_precision_string<rocblaslt_bfloat16>[] = "bf16_r";
template <>
constexpr inline char rocblaslt_precision_string<rocblaslt_half>[] = "f16_r";
template <>
constexpr inline char rocblaslt_precision_string<float>[] = "f32_r";
template <>
constexpr inline char rocblaslt_precision_string<double>[] = "f64_r";
template <>
constexpr inline char rocblaslt_precision_string<int8_t>[] = "i8_r";
template <>
constexpr inline char rocblaslt_precision_string<uint8_t>[] = "u8_r";
template <>
constexpr inline char rocblaslt_precision_string<int32_t>[] = "i32_r";
template <>
constexpr inline char rocblaslt_precision_string<uint32_t>[] = "u32_r";

std::string prefix(const char* layer, const char* caller);

const char* hipDataType_to_string(hipDataType type);

const char* hipDataType_to_bench_string(hipDataType type);

const char* rocblaslt_compute_type_to_string(rocblaslt_compute_type type);

const char* rocblaslt_matrix_layout_attributes_to_string(rocblaslt_matrix_layout_attribute_ type);

const char* rocblaslt_matmul_desc_attributes_to_string(rocblaslt_matmul_desc_attributes type);

const char* hipblasOperation_to_string(hipblasOperation_t op);

const char* rocblaslt_layer_mode2string(rocblaslt_layer_mode layer_mode);

const char* rocblaslt_epilogue_to_string(rocblaslt_epilogue epilogue);

std::string rocblaslt_matrix_layout_to_string(rocblaslt_matrix_layout mat);

std::string rocblaslt_matmul_desc_to_string(rocblaslt_matmul_desc matmul_desc);

// Return the leftmost significant bit position
#if defined(rocblaslt_ILP64)
static inline rocblaslt_int rocblaslt_clz(rocblaslt_int n)
{
    return 64 - __builtin_clzll(n);
}
#else
static inline rocblaslt_int rocblaslt_clz(rocblaslt_int n)
{
    return 32 - __builtin_clz(n);
}
#endif
std::ostream* get_logger_os();
uint32_t      get_logger_layer_mode();

template <typename H, typename... Ts>
void log_base(rocblaslt_layer_mode layer_mode, const char* func, H head, Ts&&... xs)
{
    if(get_logger_layer_mode() & layer_mode)
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::string                 comma_separator = " ";

        std::ostream* os = get_logger_os();

        std::string prefix_str = prefix(rocblaslt_layer_mode2string(layer_mode), func);

        log_arguments(*os, comma_separator, prefix_str, head, std::forward<Ts>(xs)...);
    }
}

template <typename H, typename... Ts>
void log_error(const char* func, H head, Ts&&... xs)
{
    log_base(rocblaslt_layer_mode_log_error, func, head, std::forward<Ts>(xs)...);
}

// if trace logging is turned on with
// (handle->layer_mode & rocblaslt_layer_mode_log_trace) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_trace(const char* func, H head, Ts&&... xs)
{
    log_base(rocblaslt_layer_mode_log_trace, func, head, std::forward<Ts>(xs)...);
}

// if trace logging is turned on with
// (handle->layer_mode & rocblaslt_layer_mode_log_hints) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_hints(const char* func, H head, Ts&&... xs)
{
    log_base(rocblaslt_layer_mode_log_hints, func, head, std::forward<Ts>(xs)...);
}

// if trace logging is turned on with
// (handle->layer_mode & rocblaslt_layer_mode_log_info) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_info(const char* func, H head, Ts&&... xs)
{
    log_base(rocblaslt_layer_mode_log_info, func, head, std::forward<Ts>(xs)...);
}

// if trace logging is turned on with
// (handle->layer_mode & rocblaslt_layer_mode_log_api) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_api(const char* func, H head, Ts&&... xs)
{
    log_base(rocblaslt_layer_mode_log_api, func, head, std::forward<Ts>(xs)...);
}

// if bench logging is turned on with
// (handle->layer_mode & rocblaslt_layer_mode_log_bench) == true
// then
// log_bench will call log_arguments to log a string that
// can be input to the executable rocblaslt-bench.
template <typename... Ts>
void log_bench(const char* func, Ts&&... xs)
{
    std::lock_guard<std::mutex> lock(log_mutex);
    std::ostream*               os = get_logger_os();
    *os << "hipblaslt-bench ";
    log_arguments_bench(*os, std::forward<Ts>(xs)...);
    *os << std::endl;
}

inline void log_bench_from_str(std::string s)
{
    std::lock_guard<std::mutex> lock(log_mutex);
    std::ostream*               os = get_logger_os();
    *os << s.c_str();
    *os << std::endl;
}

template <typename... Ts>
inline std::string log_str(const char* func, Ts&&... xs)
{
    std::stringstream ss;
    ss << "hipblaslt-bench ";
    log_arguments_bench(ss, std::forward<Ts>(xs)...);
    return ss.str();
}

// if profile logging is turned on with
// (handle->layer_mode & rocblaslt_layer_mode_log_profile) == true
// log_profile will call argument_profile to profile actual arguments,
// keeping count of the number of times each set of arguments is used
template <typename... Ts>
void log_profile(const char* func, Ts&&... xs)
{
    // Make a tuple with the arguments
    auto tup = std::make_tuple("function", func, std::forward<Ts>(xs)...);

    // Set up profile
    static argument_profile<decltype(tup)> profile(get_logger_os());

    // Add at_quick_exit handler in case the program exits early
    static int aqe = at_quick_exit([] { profile.~argument_profile(); });

    // Profile the tuple
    profile(std::move(tup));
}

// Convert the current C++ exception to rocblaslt_status
// This allows extern "C" functions to return this function in a catch(...)
// block while converting all C++ exceptions to an equivalent rocblaslt_status
// here
inline rocblaslt_status exception_to_rocblaslt_status(std::exception_ptr e
                                                      = std::current_exception())
try
{
    if(e)
        std::rethrow_exception(e);
    return rocblaslt_status_success;
}
catch(const rocblaslt_status& status)
{
    return status;
}
catch(const std::bad_alloc&)
{
    return rocblaslt_status_memory_error;
}
catch(...)
{
    return rocblaslt_status_internal_error;
}

// For host scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar_device_host(T x)
{
    return x;
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar_device_host(const T* xp)
{
    return *xp;
}

// For host scalars
template <typename T>
__forceinline__ __device__ __host__ T zero_scalar_device_host(T x)
{
    return static_cast<T>(0);
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T zero_scalar_device_host(const T* xp)
{
    return static_cast<T>(0);
}

inline bool is_grad_enabled(rocblaslt_epilogue value_)
{
    switch(value_)
    {
    case ROCBLASLT_EPILOGUE_DGELU:
    case ROCBLASLT_EPILOGUE_DGELU_BGRAD:
    case ROCBLASLT_EPILOGUE_BGRADA:
    case ROCBLASLT_EPILOGUE_BGRADB:
        return true;
    default:
        return false;
    }
};

inline bool is_e_enabled(rocblaslt_epilogue value_)
{
    switch(value_)
    {
    // forward pass:
    case ROCBLASLT_EPILOGUE_RELU_AUX:
    case ROCBLASLT_EPILOGUE_RELU_AUX_BIAS:
    case ROCBLASLT_EPILOGUE_GELU_AUX:
    case ROCBLASLT_EPILOGUE_GELU_AUX_BIAS:
    case ROCBLASLT_EPILOGUE_CLAMP_AUX_EXT:
    case ROCBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT:
    // backward pass:
    case ROCBLASLT_EPILOGUE_DGELU:
    case ROCBLASLT_EPILOGUE_DGELU_BGRAD:
        return true;
    default:
        return false;
    }
};

inline bool is_bias_enabled(rocblaslt_epilogue value_)
{
    switch(value_)
    {
    case ROCBLASLT_EPILOGUE_BIAS:
    case ROCBLASLT_EPILOGUE_GELU_BIAS:
    case ROCBLASLT_EPILOGUE_RELU_BIAS:
    case ROCBLASLT_EPILOGUE_RELU_AUX_BIAS:
    case ROCBLASLT_EPILOGUE_GELU_AUX_BIAS:
    case ROCBLASLT_EPILOGUE_DGELU_BGRAD:
    case ROCBLASLT_EPILOGUE_BGRADA:
    case ROCBLASLT_EPILOGUE_BGRADB:
    case ROCBLASLT_EPILOGUE_SWISH_BIAS_EXT:
    case ROCBLASLT_EPILOGUE_CLAMP_BIAS_EXT:
    case ROCBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT:
        return true;
    default:
        return false;
    }
};

inline bool is_act_enabled(rocblaslt_epilogue value_)
{
    switch(value_)
    {
    case ROCBLASLT_EPILOGUE_RELU:
    case ROCBLASLT_EPILOGUE_RELU_BIAS:
    case ROCBLASLT_EPILOGUE_RELU_AUX:
    case ROCBLASLT_EPILOGUE_RELU_AUX_BIAS:
    case ROCBLASLT_EPILOGUE_GELU:
    case ROCBLASLT_EPILOGUE_GELU_BIAS:
    case ROCBLASLT_EPILOGUE_GELU_AUX:
    case ROCBLASLT_EPILOGUE_GELU_AUX_BIAS:
    case ROCBLASLT_EPILOGUE_DGELU:
    case ROCBLASLT_EPILOGUE_DGELU_BGRAD:
    case ROCBLASLT_EPILOGUE_SWISH_EXT:
    case ROCBLASLT_EPILOGUE_SWISH_BIAS_EXT:
    case ROCBLASLT_EPILOGUE_CLAMP_EXT:
    case ROCBLASLT_EPILOGUE_CLAMP_BIAS_EXT:
    case ROCBLASLT_EPILOGUE_CLAMP_AUX_EXT:
    case ROCBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT:
        return true;
    case ROCBLASLT_EPILOGUE_DEFAULT:
    case ROCBLASLT_EPILOGUE_BIAS:
    default:
        return false;
    }
};

template <typename T>
struct floating_traits
{
    using data_t = T;
};

template <typename T>
using floating_data_t = typename floating_traits<T>::data_t;

// Internal use, whether Tensile supports ldc != ldd
// We assume true if the value is greater than or equal to 906
bool rocblaslt_internal_tensile_supports_ldc_ne_ldd(rocblaslt_handle handle);

// for internal use during testing, fetch arch name
//std::string rocblaslt_internal_get_arch_name();

/*! \brief User defined client arguments.
 *
 * \details This class sets the value of flush and rotating size used in the client which could be further used in the logging, only for internal use.
 */

class UserClientArguments
{
private:
    static bool    m_flush;
    static int32_t m_rotatingBufferSize;
    static int32_t m_coldIterations;
    static int32_t m_hotIterations;

public:
    // Getter and setter for the flush member variable.
    bool GetFlushValue() const
    {
        return m_flush;
    }
    void SetFlushValue(bool newFlush)
    {
        m_flush = newFlush;
    }

    // Getter and setter for the rotatingBufferSize member variable.
    int32_t GetRotatingBufferSizeValue() const
    {
        return m_rotatingBufferSize;
    }
    void SetRotatingBufferSizeValue(int32_t newrotatingBufferSize)
    {
        m_rotatingBufferSize = newrotatingBufferSize;
    }

    // Getter and setter for the coldIterations member variable.
    int32_t GetColdIterationsValue() const
    {
        return m_coldIterations;
    }
    void SetColdIterationsValue(int32_t newColdIterations)
    {
        m_coldIterations = newColdIterations;
    }

    // Getter and setter for the hotIterations member variable.
    int32_t GetHotIterationsValue() const
    {
        return m_hotIterations;
    }
    void SetHotIterationsValue(int32_t newHotIterations)
    {
        m_hotIterations = newHotIterations;
    }
};

//! Estimates based on problem size, solution tile, and  machine hardware
struct hipblasltClientPerformanceArgs
{
    //! Granularity is measured 0..1 with 1.0 meaning no granularity loss
    static double totalGranularity;
    static double tilesPerCu;
    static double tile0Granularity; // loss due to tile0
    static double tile1Granularity;
    static double cuGranularity;
    static double waveGranularity;
    static int    CUs;
    static size_t memWriteBytesD; //! Estimated memory writes D
    static size_t memReadBytes;
};

#endif // ROCBLASLT_UTILITY_HPP
