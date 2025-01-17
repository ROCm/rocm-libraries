/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "control.h"
#include "handle.h"
#include "logging.h"
#include "scalar.h"
namespace rocsparse
{
// Return the leftmost significant bit position
#if defined(rocsparse_ILP64)
    static inline rocsparse_int clz(rocsparse_int n)
    {
        // __builtin_clzll is undefined for n == 0
        if(n == 0)
        {
            return 0;
        }
        return 64 - __builtin_clzll(n);
    }
#else
    static inline rocsparse_int clz(rocsparse_int n)
    {
        // __builtin_clz is undefined for n == 0
        if(n == 0)
        {
            return 0;
        }
        return 32 - __builtin_clz(n);
    }
#endif

    // Return one on the device
    static inline void one(const rocsparse_handle handle, float** one)
    {
        *one = handle->sone;
    }

    static inline void one(const rocsparse_handle handle, double** one)
    {
        *one = handle->done;
    }

    static inline void one(const rocsparse_handle handle, rocsparse_float_complex** one)
    {
        *one = handle->cone;
    }

    static inline void one(const rocsparse_handle handle, rocsparse_double_complex** one)
    {
        *one = handle->zone;
    }

    template <typename T>
    ROCSPARSE_KERNEL(1)
    void assign_kernel(T* dest, T value)
    {
        *dest = value;
    }

    // Set a single value on the device from the host asynchronously.
    template <typename T>
    static inline hipError_t assign_async(T* dest, T value, hipStream_t stream)
    {
        // Use a kernel instead of memcpy, because memcpy is synchronous if the source is not in
        // pinned memory.
        // Memset lacks a 64bit option, but would involve a similar implicit kernel anyways.

        if(false == rocsparse_debug_variables.get_debug_kernel_launch())
        {
            hipLaunchKernelGGL(rocsparse::assign_kernel, dim3(1), dim3(1), 0, stream, dest, value);
            return hipSuccess;
        }
        else
        {
            {
                const hipError_t err = hipGetLastError();
                if(err != hipSuccess)
                {
                    std::stringstream s;
                    s << "prior to hipLaunchKernelGGL"
                      << ", hip error detected: code '" << err << "', name '"
                      << hipGetErrorName(err) << "', description '" << hipGetErrorString(err)
                      << "'";
                    ROCSPARSE_ERROR_MESSAGE(rocsparse::get_rocsparse_status_for_hip_status(err),
                                            s.str().c_str());
                    return err;
                }
            }
            hipLaunchKernelGGL(rocsparse::assign_kernel, dim3(1), dim3(1), 0, stream, dest, value);
            {
                const hipError_t err = hipGetLastError();
                if(err != hipSuccess)
                {
                    std::stringstream s;
                    s << "hip error detected: code '" << err << "', name '" << hipGetErrorName(err)
                      << "', description '" << hipGetErrorString(err) << "'";
                    ROCSPARSE_ERROR_MESSAGE(rocsparse::get_rocsparse_status_for_hip_status(err),
                                            s.str().c_str());
                    return err;
                }
            }
            return hipSuccess;
        }
    }

    // if trace logging is turned on with
    // (handle->layer_mode & rocsparse_layer_mode_log_trace) == true
    // then
    // log_function will call log_arguments to log function
    // arguments with a comma separator
    template <typename H, typename... Ts>
    void log_trace(rocsparse_handle handle, H head, Ts&&... xs)
    {
        if(nullptr != handle)
        {
            if(handle->layer_mode & rocsparse_layer_mode_log_trace)
            {
                std::string comma_separator = ",";

                std::ostream* os = handle->log_trace_os;
                rocsparse::log_arguments(*os, comma_separator, head, std::forward<Ts>(xs)...);
            }
        }
    }

    template <typename H, typename... Ts>
    void log_trace(H head, rocsparse_handle handle, Ts&&... xs)
    {
        rocsparse::log_trace(handle, head, xs...);
    }

    // if bench logging is turned on with
    // (handle->layer_mode & rocsparse_layer_mode_log_bench) == true
    // then
    // log_bench will call log_arguments to log a string that
    // can be input to the executable rocsparse-bench.
    template <typename H, typename... Ts>
    void log_bench(rocsparse_handle handle, H head, std::string precision, Ts&&... xs)
    {
        if(nullptr != handle)
        {
            if(handle->layer_mode & rocsparse_layer_mode_log_bench)
            {
                std::string space_separator = " ";

                std::ostream* os = handle->log_bench_os;
                rocsparse::log_arguments(
                    *os, space_separator, head, precision, std::forward<Ts>(xs)...);
            }
        }
    }

    // if debug logging is turned on with
    // (handle->layer_mode & rocsparse_layer_mode_log_debug) == true
    // then
    // log_debug will call log_arguments to log a error message
    // when a routine returns a status which is not rocsparse_status_success.
    static inline void log_debug(rocsparse_handle handle, std::string message)
    {
        if(nullptr != handle)
        {
            if(handle->layer_mode & rocsparse_layer_mode_log_debug)
            {
                std::string space_separator = " ";

                std::ostream* os = handle->log_debug_os;
                rocsparse::log_arguments(*os, space_separator, message);
            }
        }
    }

    // Trace log scalar values pointed to by pointer
    template <typename T>
    T log_trace_scalar_value(const T* value)
    {
        return value ? *value : std::numeric_limits<T>::quiet_NaN();
    }

    template <typename T>
    T log_trace_scalar_value(rocsparse_handle handle, const T* value)
    {
        if(nullptr != handle)
        {
            if(handle->layer_mode & rocsparse_layer_mode_log_trace)
            {
                T host;
                if(value && handle->pointer_mode == rocsparse_pointer_mode_device)
                {
                    hipStreamCaptureStatus capture_status;
                    RETURN_IF_HIP_ERROR(hipStreamIsCapturing(handle->stream, &capture_status));

                    if(capture_status == hipStreamCaptureStatusNone)
                    {
                        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                            &host, value, sizeof(host), hipMemcpyDeviceToHost, handle->stream));
                        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
                        value = &host;
                    }
                    else
                    {
                        value = nullptr;
                    }
                }
                return rocsparse::log_trace_scalar_value(value);
            }
        }
        return T{};
    }

#define LOG_TRACE_SCALAR_VALUE(handle, value) rocsparse::log_trace_scalar_value(handle, value)

    // Bench log scalar values pointed to by pointer
    template <typename T>
    T log_bench_scalar_value(const T* value)
    {
        return (value ? *value : std::numeric_limits<T>::quiet_NaN());
    }

    template <typename T>
    T log_bench_scalar_value(rocsparse_handle handle, const T* value)
    {
        if(handle->layer_mode & rocsparse_layer_mode_log_bench)
        {
            T host;
            if(value && handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                hipStreamCaptureStatus capture_status;
                RETURN_IF_HIP_ERROR(hipStreamIsCapturing(handle->stream, &capture_status));

                if(capture_status == hipStreamCaptureStatusNone)
                {
                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                        &host, value, sizeof(host), hipMemcpyDeviceToHost, handle->stream));
                    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
                    value = &host;
                }
                else
                {
                    value = nullptr;
                }
            }
            return rocsparse::log_bench_scalar_value(value);
        }
        return T{};
    }

#define LOG_BENCH_SCALAR_VALUE(handle, name) log_bench_scalar_value(handle, name)

    // replaces X in string with s, d, c, z or h depending on typename T
    template <typename T>
    std::string replaceX(std::string input_string)
    {
        if(std::is_same<T, float>::value)
        {
            std::replace(input_string.begin(), input_string.end(), 'X', 's');
        }
        else if(std::is_same<T, double>::value)
        {
            std::replace(input_string.begin(), input_string.end(), 'X', 'd');
        }
        /*
    else if(std::is_same<T, rocsparse_float_complex>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'c');
    }
    else if(std::is_same<T, rocsparse_double_complex>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'z');
    }
    else if(std::is_same<T, rocsparse_half>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'h');
    }
    */
        return input_string;
    }

//
// These macros can be redefined if the developer includes src/include/debug.h
//
#define ROCSPARSE_DEBUG_VERBOSE(msg__) (void)0
#define ROCSPARSE_RETURN_STATUS(token__) return rocsparse_status_##token__

    // Convert the current C++ exception to rocsparse_status
    // This allows extern "C" functions to return this function in a catch(...) block
    // while converting all C++ exceptions to an equivalent rocsparse_status here
    inline rocsparse_status exception_to_rocsparse_status(std::exception_ptr e
                                                          = std::current_exception())
    try
    {
        if(e)
            std::rethrow_exception(e);
        return rocsparse_status_success;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }
    catch(const std::bad_alloc&)
    {
        return rocsparse_status_memory_error;
    }
    catch(...)
    {
        return rocsparse_status_thrown_exception;
    }

    //
    // Provide some utility methods for enums.
    //
    struct enum_utils
    {
        template <typename U>
        static inline bool is_invalid(U value_);
        template <typename U>
        static inline const char* to_string(U value_);
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_sparse_to_sparse_stage value)
    {
        switch(value)
        {
        case rocsparse_sparse_to_sparse_stage_analysis:
        case rocsparse_sparse_to_sparse_stage_compute:
        {
            return false;
        }
        }
        return true;
    };
    template <>
    inline bool enum_utils::is_invalid(rocsparse_extract_stage value)
    {
        switch(value)
        {
        case rocsparse_extract_stage_analysis:
        case rocsparse_extract_stage_compute:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse::blas_impl value)
    {
        switch(value)
        {
        case rocsparse::blas_impl_none:
        case rocsparse::blas_impl_default:
        case rocsparse::blas_impl_rocblas:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_sparse_to_sparse_alg value)
    {
        switch(value)
        {
        case rocsparse_sparse_to_sparse_alg_default:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_extract_alg value)
    {
        switch(value)
        {
        case rocsparse_extract_alg_default:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_pointer_mode value)
    {
        switch(value)
        {
        case rocsparse_pointer_mode_device:
        case rocsparse_pointer_mode_host:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_spmat_attribute value)
    {
        switch(value)
        {
        case rocsparse_spmat_fill_mode:
        case rocsparse_spmat_diag_type:
        case rocsparse_spmat_matrix_type:
        case rocsparse_spmat_storage_mode:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_itilu0_alg value)
    {
        switch(value)
        {
        case rocsparse_itilu0_alg_default:
        case rocsparse_itilu0_alg_async_inplace:
        case rocsparse_itilu0_alg_async_split:
        case rocsparse_itilu0_alg_sync_split:
        case rocsparse_itilu0_alg_sync_split_fusion:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_diag_type value)
    {
        switch(value)
        {
        case rocsparse_diag_type_unit:
        case rocsparse_diag_type_non_unit:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_fill_mode value_)
    {
        switch(value_)
        {
        case rocsparse_fill_mode_lower:
        case rocsparse_fill_mode_upper:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_storage_mode value_)
    {
        switch(value_)
        {
        case rocsparse_storage_mode_sorted:
        case rocsparse_storage_mode_unsorted:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_index_base value_)
    {
        switch(value_)
        {
        case rocsparse_index_base_zero:
        case rocsparse_index_base_one:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_matrix_type value_)
    {
        switch(value_)
        {
        case rocsparse_matrix_type_general:
        case rocsparse_matrix_type_symmetric:
        case rocsparse_matrix_type_hermitian:
        case rocsparse_matrix_type_triangular:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_direction value_)
    {
        switch(value_)
        {
        case rocsparse_direction_row:
        case rocsparse_direction_column:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_operation value_)
    {
        switch(value_)
        {
        case rocsparse_operation_none:
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_indextype value_)
    {
        switch(value_)
        {
        case rocsparse_indextype_u16:
        case rocsparse_indextype_i32:
        case rocsparse_indextype_i64:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_datatype value_)
    {
        switch(value_)
        {
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_f32_r:
        case rocsparse_datatype_f64_r:
        case rocsparse_datatype_f32_c:
        case rocsparse_datatype_f64_c:
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_order value_)
    {
        switch(value_)
        {
        case rocsparse_order_row:
        case rocsparse_order_column:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_action value)
    {
        switch(value)
        {
        case rocsparse_action_numeric:
        case rocsparse_action_symbolic:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_hyb_partition value)
    {
        switch(value)
        {
        case rocsparse_hyb_partition_auto:
        case rocsparse_hyb_partition_user:
        case rocsparse_hyb_partition_max:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_gtsv_interleaved_alg value_)
    {
        switch(value_)
        {
        case rocsparse_gtsv_interleaved_alg_default:
        case rocsparse_gtsv_interleaved_alg_thomas:
        case rocsparse_gtsv_interleaved_alg_lu:
        case rocsparse_gtsv_interleaved_alg_qr:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_sparse_to_dense_alg value_)
    {
        switch(value_)
        {
        case rocsparse_sparse_to_dense_alg_default:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_dense_to_sparse_alg value_)
    {
        switch(value_)
        {
        case rocsparse_dense_to_sparse_alg_default:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_spmv_alg value_)
    {
        switch(value_)
        {
        case rocsparse_spmv_alg_default:
        case rocsparse_spmv_alg_coo:
        case rocsparse_spmv_alg_csr_adaptive:
        case rocsparse_spmv_alg_csr_rowsplit:
        case rocsparse_spmv_alg_ell:
        case rocsparse_spmv_alg_coo_atomic:
        case rocsparse_spmv_alg_bsr:
        case rocsparse_spmv_alg_csr_lrb:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_spsv_alg value_)
    {
        switch(value_)
        {
        case rocsparse_spsv_alg_default:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_spitsv_alg value_)
    {
        switch(value_)
        {
        case rocsparse_spitsv_alg_default:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_check_spmat_stage value_)
    {
        switch(value_)
        {
        case rocsparse_check_spmat_stage_buffer_size:
        case rocsparse_check_spmat_stage_compute:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_spmv_stage value_)
    {
        switch(value_)
        {
        case rocsparse_spmv_stage_buffer_size:
        case rocsparse_spmv_stage_preprocess:
        case rocsparse_spmv_stage_compute:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_spsv_stage value_)
    {
        switch(value_)
        {
        case rocsparse_spsv_stage_buffer_size:
        case rocsparse_spsv_stage_preprocess:
        case rocsparse_spsv_stage_compute:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_spitsv_stage value_)
    {
        switch(value_)
        {
        case rocsparse_spitsv_stage_buffer_size:
        case rocsparse_spitsv_stage_preprocess:
        case rocsparse_spitsv_stage_compute:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_spsm_alg value_)
    {
        switch(value_)
        {
        case rocsparse_spsm_alg_default:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_spsm_stage value_)
    {
        switch(value_)
        {
        case rocsparse_spsm_stage_buffer_size:
        case rocsparse_spsm_stage_preprocess:
        case rocsparse_spsm_stage_compute:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_spmm_alg value_)
    {
        switch(value_)
        {
        case rocsparse_spmm_alg_default:
        case rocsparse_spmm_alg_csr:
        case rocsparse_spmm_alg_coo_segmented:
        case rocsparse_spmm_alg_coo_atomic:
        case rocsparse_spmm_alg_csr_row_split:
        case rocsparse_spmm_alg_csr_nnz_split:
        case rocsparse_spmm_alg_csr_merge_path:
        case rocsparse_spmm_alg_coo_segmented_atomic:
        case rocsparse_spmm_alg_bell:
        case rocsparse_spmm_alg_bsr:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_spmm_stage value_)
    {
        switch(value_)
        {
        case rocsparse_spmm_stage_buffer_size:
        case rocsparse_spmm_stage_preprocess:
        case rocsparse_spmm_stage_compute:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_sddmm_alg value_)
    {
        switch(value_)
        {
        case rocsparse_sddmm_alg_default:
        {
            return false;
        }
        case rocsparse_sddmm_alg_dense:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_spgemm_alg value_)
    {
        switch(value_)
        {
        case rocsparse_spgemm_alg_default:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_spgemm_stage value_)
    {
        switch(value_)
        {
        case rocsparse_spgemm_stage_buffer_size:
        case rocsparse_spgemm_stage_nnz:
        case rocsparse_spgemm_stage_compute:
        case rocsparse_spgemm_stage_symbolic:
        case rocsparse_spgemm_stage_numeric:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_solve_policy value_)
    {
        switch(value_)
        {
        case rocsparse_solve_policy_auto:
        {
            return false;
        }
        }
        return true;
    };

    template <>
    inline bool enum_utils::is_invalid(rocsparse_analysis_policy value_)
    {
        switch(value_)
        {
        case rocsparse_analysis_policy_reuse:
        case rocsparse_analysis_policy_force:
        {
            return false;
        }
        }
        return true;
    };

    template <typename T>
    struct floating_traits
    {
        using data_t = T;
    };

    template <>
    struct floating_traits<rocsparse_float_complex>
    {
        using data_t = float;
    };

    template <>
    struct floating_traits<rocsparse_double_complex>
    {
        using data_t = double;
    };

    template <typename T>
    using floating_data_t = typename floating_traits<T>::data_t;

    template <typename T>
    rocsparse_indextype get_indextype();

    template <>
    inline rocsparse_indextype get_indextype<int32_t>()
    {
        return rocsparse_indextype_i32;
    }

    template <>
    inline rocsparse_indextype get_indextype<uint16_t>()
    {
        return rocsparse_indextype_u16;
    }

    template <>
    inline rocsparse_indextype get_indextype<int64_t>()
    {
        return rocsparse_indextype_i64;
    }

    template <typename T>
    rocsparse_datatype get_datatype();

    template <>
    inline rocsparse_datatype get_datatype<float>()
    {
        return rocsparse_datatype_f32_r;
    }

    template <>
    inline rocsparse_datatype get_datatype<double>()
    {
        return rocsparse_datatype_f64_r;
    }

    template <>
    inline rocsparse_datatype get_datatype<rocsparse_float_complex>()
    {
        return rocsparse_datatype_f32_c;
    }

    template <>
    inline rocsparse_datatype get_datatype<rocsparse_double_complex>()
    {
        return rocsparse_datatype_f64_c;
    }

    inline size_t indextype_sizeof(rocsparse_indextype that)
    {
        switch(that)
        {

        case rocsparse_indextype_i32:
        {
            return sizeof(int32_t);
        }
        case rocsparse_indextype_i64:
        {
            return sizeof(int64_t);
        }
        case rocsparse_indextype_u16:
        {
            return sizeof(uint16_t);
        }
        }
    }

    inline size_t datatype_sizeof(rocsparse_datatype that)
    {
        switch(that)
        {
        case rocsparse_datatype_i32_r:
            return sizeof(int32_t);

        case rocsparse_datatype_u32_r:
        {
            return sizeof(uint32_t);
        }

        case rocsparse_datatype_i8_r:
        {
            return sizeof(int8_t);
        }

        case rocsparse_datatype_u8_r:
        {
            return sizeof(uint8_t);
        }
        case rocsparse_datatype_f16_r:
        {
            return sizeof(_Float16);
        }
        case rocsparse_datatype_f32_r:
        {
            return sizeof(float);
        }

        case rocsparse_datatype_f64_r:
        {
            return sizeof(double);
        }

        case rocsparse_datatype_f32_c:
        {
            return sizeof(rocsparse_float_complex);
        }
        case rocsparse_datatype_f64_c:
        {
            return sizeof(rocsparse_double_complex);
        }
        }
    }

#include "memstat.h"

    inline rocsparse_status calculate_nnz(
        int64_t m, rocsparse_indextype indextype, const void* ptr, int64_t* nnz, hipStream_t stream)
    {
        if(m == 0)
        {
            nnz[0] = 0;
            return rocsparse_status_success;
        }
        const char* p
            = reinterpret_cast<const char*>(ptr) + rocsparse::indextype_sizeof(indextype) * m;
        int64_t end, start;
        switch(indextype)
        {
        case rocsparse_indextype_i32:
        {
            int32_t u, v;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &u, ptr, rocsparse::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &v, p, rocsparse::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
            start = u;
            end   = v;
            break;
        }
        case rocsparse_indextype_i64:
        {
            int64_t u, v;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &u, ptr, rocsparse::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &v, p, rocsparse::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
            start = u;
            end   = v;
            break;
        }
        case rocsparse_indextype_u16:
        {
            uint16_t u, v;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &u, ptr, rocsparse::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &v, p, rocsparse::indextype_sizeof(indextype), hipMemcpyDeviceToHost, stream));
            start = u;
            end   = v;
            break;
        }
        }
        nnz[0] = end - start;
        return rocsparse_status_success;
    }

    template <typename S, typename T>
    inline rocsparse_status internal_convert_scalar(const S s, T& t)
    {
        if(s <= std::numeric_limits<T>::max() && s >= std::numeric_limits<T>::min())
        {
            t = static_cast<T>(s);
            return rocsparse_status_success;
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_type_mismatch);
        }
    }
}
