/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include <limits>

#ifndef DISABLE_ROCTX
#if !defined(ROCBLAS_STATIC_LIB) && !defined(WIN32)
#include <roctracer/roctx.h>
#endif
#endif

#include "logging.hpp"

/*************************************************
 * Bench log scalar values pointed to by pointer *
 *************************************************/

template <typename T>
std::string
    rocblas_internal_log_bench_scalar_value(rocblas_handle handle, const char* name, const T* value)
{
    T host;
    if(value && handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipMemcpyAsync(&host, value, sizeof(host), hipMemcpyDeviceToHost, handle->get_stream());
        hipStreamSynchronize(handle->get_stream());
        value = &host;
    }
    return rocblas_internal_log_bench_scalar_value(name, value);
}

// instantiate helpers
template std::string rocblas_internal_log_bench_scalar_value(rocblas_handle      handle,
                                                             const char*         name,
                                                             const rocblas_half* value);
template std::string rocblas_internal_log_bench_scalar_value(rocblas_handle handle,
                                                             const char*    name,
                                                             const int32_t* value);
template std::string rocblas_internal_log_bench_scalar_value(rocblas_handle handle,
                                                             const char*    name,
                                                             const float*   value);
template std::string rocblas_internal_log_bench_scalar_value(rocblas_handle handle,
                                                             const char*    name,
                                                             const double*  value);
template std::string rocblas_internal_log_bench_scalar_value(rocblas_handle               handle,
                                                             const char*                  name,
                                                             const rocblas_float_complex* value);
template std::string rocblas_internal_log_bench_scalar_value(rocblas_handle                handle,
                                                             const char*                   name,
                                                             const rocblas_double_complex* value);

/*************************************************
 * Trace log scalar values pointed to by pointer *
 *************************************************/

inline float rocblas_internal_log_trace_scalar_value(const rocblas_half* value)
{
    return value ? float(*value) : std::numeric_limits<float>::quiet_NaN();
}

template <typename T>
T rocblas_internal_log_trace_scalar_value(const T* value)
{
    if constexpr(!rocblas_is_complex<T>)
    {
        return value ? *value : std::numeric_limits<T>::quiet_NaN();
    }
    else
    {
        return value ? *value
                     : T{std::numeric_limits<typename T::value_type>::quiet_NaN(),
                         std::numeric_limits<typename T::value_type>::quiet_NaN()};
    }
}

template <typename T>
std::string rocblas_internal_log_trace_scalar_value(rocblas_handle handle, const T* value)
{
    rocblas_internal_ostream os;

    T host;
    if(value && handle->pointer_mode == rocblas_pointer_mode_device)
    {
        hipMemcpyAsync(&host, value, sizeof(host), hipMemcpyDeviceToHost, handle->get_stream());
        hipStreamSynchronize(handle->get_stream());
        value = &host;
    }
    os << rocblas_internal_log_trace_scalar_value(value);
    return os.str();
}

// instantiate helpers
template std::string rocblas_internal_log_trace_scalar_value(rocblas_handle      handle,
                                                             const rocblas_half* value);
template std::string rocblas_internal_log_trace_scalar_value(rocblas_handle handle,
                                                             const int32_t* value);
template std::string rocblas_internal_log_trace_scalar_value(rocblas_handle handle,
                                                             const float*   value);
template std::string rocblas_internal_log_trace_scalar_value(rocblas_handle handle,
                                                             const double*  value);
template std::string rocblas_internal_log_trace_scalar_value(rocblas_handle               handle,
                                                             const rocblas_float_complex* value);
template std::string rocblas_internal_log_trace_scalar_value(rocblas_handle                handle,
                                                             const rocblas_double_complex* value);

const char* c_rocblas_internal = "rocblas_internal";

void rocblas_internal_logger::log_endline(rocblas_internal_ostream& os)
{
#ifndef DISABLE_ROCTX
#if !defined(ROCBLAS_STATIC_LIB) && !defined(WIN32)
    if(!m_active)
    {
        const std::string& name = os.str();
        roctxRangePush(name.c_str());
        m_active = true;
    }
#endif
#endif
    os << std::endl;
}

void rocblas_internal_logger::log_cleanup()
{
#ifndef DISABLE_ROCTX
#if !defined(ROCBLAS_STATIC_LIB) && !defined(WIN32)
    roctxRangePop();
#endif
#endif
}

/******************************************************************
 * Log alpha and beta with dynamic compute_type in *_ex functions *
 ******************************************************************/
rocblas_status rocblas_internal_log_trace_alpha_beta_ex(rocblas_datatype          compute_type,
                                                        const void*               alpha,
                                                        const void*               beta,
                                                        rocblas_internal_ostream& alphass,
                                                        rocblas_internal_ostream& betass)
{
    switch(compute_type)
    {
    case rocblas_datatype_f16_r:
        alphass << rocblas_internal_log_trace_scalar_value(
            reinterpret_cast<const rocblas_half*>(alpha));
        betass << rocblas_internal_log_trace_scalar_value(
            reinterpret_cast<const rocblas_half*>(beta));
        break;
    case rocblas_datatype_f32_r:
        alphass << rocblas_internal_log_trace_scalar_value(reinterpret_cast<const float*>(alpha));
        betass << rocblas_internal_log_trace_scalar_value(reinterpret_cast<const float*>(beta));
        break;
    case rocblas_datatype_f64_r:
        alphass << rocblas_internal_log_trace_scalar_value(reinterpret_cast<const double*>(alpha));
        betass << rocblas_internal_log_trace_scalar_value(reinterpret_cast<const double*>(beta));
        break;
    case rocblas_datatype_i32_r:
        alphass << rocblas_internal_log_trace_scalar_value(reinterpret_cast<const int32_t*>(alpha));
        betass << rocblas_internal_log_trace_scalar_value(reinterpret_cast<const int32_t*>(beta));
        break;
    case rocblas_datatype_f32_c:
        alphass << rocblas_internal_log_trace_scalar_value(
            reinterpret_cast<const rocblas_float_complex*>(alpha));
        betass << rocblas_internal_log_trace_scalar_value(
            reinterpret_cast<const rocblas_float_complex*>(beta));
        break;
    case rocblas_datatype_f64_c:
        alphass << rocblas_internal_log_trace_scalar_value(
            reinterpret_cast<const rocblas_double_complex*>(alpha));
        betass << rocblas_internal_log_trace_scalar_value(
            reinterpret_cast<const rocblas_double_complex*>(beta));
        break;
    default:
        return rocblas_status_not_implemented;
    }
    return rocblas_status_success;
}

rocblas_status rocblas_internal_log_bench_alpha_beta_ex(rocblas_datatype compute_type,
                                                        const void*      alpha,
                                                        const void*      beta,
                                                        std::string&     alphas,
                                                        std::string&     betas)
{
    switch(compute_type)
    {
    case rocblas_datatype_f16_r:
        alphas = rocblas_internal_log_bench_scalar_value(
            "alpha", reinterpret_cast<const rocblas_half*>(alpha));
        betas = rocblas_internal_log_bench_scalar_value(
            "beta", reinterpret_cast<const rocblas_half*>(beta));
        break;
    case rocblas_datatype_f32_r:
        alphas = rocblas_internal_log_bench_scalar_value("alpha",
                                                         reinterpret_cast<const float*>(alpha));
        betas
            = rocblas_internal_log_bench_scalar_value("beta", reinterpret_cast<const float*>(beta));
        break;
    case rocblas_datatype_f64_r:
        alphas = rocblas_internal_log_bench_scalar_value("alpha",
                                                         reinterpret_cast<const double*>(alpha));
        betas  = rocblas_internal_log_bench_scalar_value("beta",
                                                        reinterpret_cast<const double*>(beta));
        break;
    case rocblas_datatype_i32_r:
        alphas = rocblas_internal_log_bench_scalar_value("alpha",
                                                         reinterpret_cast<const int32_t*>(alpha));
        betas  = rocblas_internal_log_bench_scalar_value("beta",
                                                        reinterpret_cast<const int32_t*>(beta));
        break;
    case rocblas_datatype_f32_c:
        alphas = rocblas_internal_log_bench_scalar_value(
            "alpha", reinterpret_cast<const rocblas_float_complex*>(alpha));
        betas = rocblas_internal_log_bench_scalar_value(
            "beta", reinterpret_cast<const rocblas_float_complex*>(beta));
        break;
    case rocblas_datatype_f64_c:
        alphas = rocblas_internal_log_bench_scalar_value(
            "alpha", reinterpret_cast<const rocblas_double_complex*>(alpha));
        betas = rocblas_internal_log_bench_scalar_value(
            "beta", reinterpret_cast<const rocblas_double_complex*>(beta));
        break;
    default:
        return rocblas_status_not_implemented;
    }
    return rocblas_status_success;
}

/*********************************************************************
 * Bench log precision for mixed precision scal_ex and nrm2_ex calls *
 *********************************************************************/
std::string rocblas_internal_log_bench_ex_precisions(rocblas_datatype a_type,
                                                     rocblas_datatype x_type,
                                                     rocblas_datatype ex_type)
{
    rocblas_internal_ostream ss;
    if(a_type == x_type && x_type == ex_type)
        ss << "-r " << a_type;
    else
        ss << "--a_type " << a_type << " --b_type " << x_type << " --compute_type " << ex_type;
    return ss.str();
}

template <typename T>
double rocblas_internal_value_category(const T* beta, rocblas_datatype compute_type)
{
    if(beta == nullptr)
        return 0.0;

    switch(compute_type)
    {
    case rocblas_datatype_f16_r:
        return rocblas_internal_value_category(*reinterpret_cast<const rocblas_half*>(beta));
    case rocblas_datatype_f32_r:
        return rocblas_internal_value_category(*reinterpret_cast<const float*>(beta));
    case rocblas_datatype_f64_r:
        return rocblas_internal_value_category(*reinterpret_cast<const double*>(beta));
    case rocblas_datatype_i32_r:
        return rocblas_internal_value_category(*reinterpret_cast<const int32_t*>(beta));
    case rocblas_datatype_f32_c:
        return rocblas_internal_value_category(
            *reinterpret_cast<const rocblas_float_complex*>(beta));
    case rocblas_datatype_f64_c:
        return rocblas_internal_value_category(
            *reinterpret_cast<const rocblas_double_complex*>(beta));
    default:
        throw rocblas_status_internal_error;
    }
}

// instantiate support
template double rocblas_internal_value_category(const void* beta, rocblas_datatype compute_type);
template double rocblas_internal_value_category(const rocblas_half* beta,
                                                rocblas_datatype    compute_type);
template double rocblas_internal_value_category(const int32_t* beta, rocblas_datatype compute_type);
template double rocblas_internal_value_category(const float* beta, rocblas_datatype compute_type);
template double rocblas_internal_value_category(const double* beta, rocblas_datatype compute_type);
template double rocblas_internal_value_category(const rocblas_float_complex* beta,
                                                rocblas_datatype             compute_type);
template double rocblas_internal_value_category(const rocblas_double_complex* beta,
                                                rocblas_datatype              compute_type);
