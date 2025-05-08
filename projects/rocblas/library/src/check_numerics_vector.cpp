/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "check_numerics_vector.hpp"
#include "int64_helpers.hpp"
#include "utility.hpp"

/**
  *
  * rocblas_check_numerics_vector_kernel(n, xa, offset_x, inc_x, stride_x, abnormal)
  *
  * Info about rocblas_check_numerics_vector_kernel function:
  *
  *    It is the kernel function which checks a vector for numerical abnormalities such as NaN/zero/Inf/denormal values and updates the rocblas_check_numerics_t structure.
  *
  * Parameters   : n            : Total number of elements in the vector
  *                xa           : Pointer to the vector which is under consideration for numerical abnormalities
  *                offset_x     : Offset of vector 'xa'
  *                inc_x        : Stride between consecutive values of vector 'xa'
  *                stride_x     : Specifies the pointer increment between one vector 'x_i' and the next one (xa_i+1) (where (xa_i) is the i-th instance of the batch)
  *                abnormal     : Device pointer to the rocblas_check_numerics_t structure
  *
  * Return Value : Nothing --
  *
**/

template <int DIM_X, typename T>
ROCBLAS_KERNEL(DIM_X)
rocblas_check_numerics_vector_kernel(rocblas_int               n,
                                     T                         xa,
                                     rocblas_stride            offset_x,
                                     int64_t                   inc_x,
                                     rocblas_stride            stride_x,
                                     rocblas_check_numerics_t* abnormal)
{
    auto*   x   = load_ptr_batch(xa, blockIdx.y, offset_x, stride_x);
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    //Check every element of the x vector for a NaN/zero/Inf/denormal value
    if(tid < n)
    {
        auto value = x[tid * inc_x];
        if(!abnormal->has_zero && rocblas_iszero(value))
            abnormal->has_zero = true;
        if(!abnormal->has_NaN && rocblas_isnan(value))
            abnormal->has_NaN = true;
        if(!abnormal->has_Inf && rocblas_isinf(value))
            abnormal->has_Inf = true;
        if(!abnormal->has_denorm && rocblas_isdenorm(value))
            abnormal->has_denorm = true;
    }
}

/**
  *
  * rocblas_check_numerics_abnormal_struct(function_name, check_numerics, is_input, h_abnormal)
  *
  * Info about rocblas_check_numerics_abnormal_struct function:
  *
  *    It is the host function which accepts the 'h_abnormal' structure and
  *    also helps in debugging based on the different types of flags in rocblas_check_numerics_mode that users set to debug potential NaN/zero/Inf/denormal value.
  *
  * Parameters   : function_name         : Name of the rocBLAS math function
  *                check_numerics        : User defined flag for debugging
  *                is_input              : To check if the vector under consideration is an Input or an Output vector
  *                h_abnormal            : Structure holding the boolean NaN/zero/Inf/denormal
  *
  * If check_numerics contains flag rocblas_check_numerics_mode_only_nan_inf then denorms are ignored
  *
  * Return Value : rocblas_status
  *        rocblas_status_success        : Return status if the vector does not have a NaN/Inf/denormal value
  *   rocblas_status_check_numerics_fail : Return status if the vector contains a NaN/Inf/denormal value and 'check_numerics' enum is set to 'rocblas_check_numerics_mode_fail'
  *
**/

rocblas_status rocblas_check_numerics_abnormal_struct(const char*               function_name,
                                                      const int                 check_numerics,
                                                      bool                      is_input,
                                                      rocblas_check_numerics_t* h_abnormal)
{
    //is_abnormal will be set if the vector has a NaN/Inf/denormal value
    bool is_abnormal = (h_abnormal->has_NaN != 0) || (h_abnormal->has_Inf != 0)
                       || (h_abnormal->has_denorm != 0
                           && !(check_numerics & rocblas_check_numerics_mode_only_nan_inf));

    //Fully informative message will be printed if 'check_numerics == ROCBLAS_CHECK_NUMERICS_INFO' or 'check_numerics == ROCBLAS_CHECK_NUMERICS_WARN' and 'is_abnormal'
    if(((check_numerics & rocblas_check_numerics_mode_info) != 0)
       || (((check_numerics & rocblas_check_numerics_mode_warn) != 0) && is_abnormal))
    {
        if(is_input)
        {
            rocblas_cerr << "Function name:\t" << function_name << " :- Input :\t"
                         << " has_NaN " << h_abnormal->has_NaN << " has_zero "
                         << h_abnormal->has_zero << " has_Inf " << h_abnormal->has_Inf
                         << " has_denorm " << h_abnormal->has_denorm << std::endl;
        }
        else
        {
            rocblas_cerr << "Function name:\t" << function_name << " :- Output :\t"
                         << " has_NaN " << h_abnormal->has_NaN << " has_zero "
                         << h_abnormal->has_zero << " has_Inf " << h_abnormal->has_Inf
                         << " has_denorm " << h_abnormal->has_denorm << std::endl;
        }
    }

    if(is_abnormal)
    { //If 'check_numerics ==rocblas_check_numerics_mode_fail' then the 'rocblas_status_check_numerics_fail' status
        //is returned which signifies that the vector has a NaN/Inf/denormal value
        if((check_numerics & rocblas_check_numerics_mode_fail) != 0)
            return rocblas_status_check_numerics_fail;
    }

    return rocblas_status_success;
}
/**
  *
  * rocblas_internal_check_numerics_vector_template(function_name, handle, n, x, offset_x, inc_x, stride_x, batch_count, check_numerics, is_input)
  *
  * Info about rocblas_internal_check_numerics_vector_template function:
  *
  *    It is the host function which accepts a vector and calls the 'rocblas_check_numerics_vector_kernel' kernel function
  *    to check for numerical abnormalities such as NaN/zero/Inf/denormal value in that vector.
  *    It also helps in debugging based on the different types of flags in rocblas_check_numerics_mode that users set to debug potential NaN/zero/Inf/denormal value.
  *
  * Parameters   : function_name         : Name of the rocBLAS math function
  *                handle                : Handle to the rocblas library context queue
  *                n                     : Total number of elements in the vector 'x'
  *                x                     : Pointer to the vector which is under check for numerical abnormalities
  *                offset_x              : Offset of vector 'x'
  *                inc_x                 : Stride between consecutive values of vector 'x'
  *                stride_x              : Specifies the pointer increment between one vector 'x_i' and the next one (x_i+1) (where (x_i) is the i-th instance of the batch)
  *                check_numerics        : User defined flag for debugging
  *                is_input              : To check if the vector under consideration is an Input or an Output vector
  *
  * Return Value : rocblas_status
  *        rocblas_status_success        : Return status if the vector does not have a NaN/Inf/denormal value
  *   rocblas_status_check_numerics_fail : Return status if the vector contains a NaN/Inf/denormal value and 'check_numerics' enum is set to 'rocblas_check_numerics_mode_fail'
  *
**/

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_check_numerics_vector_template(const char*    function_name,
                                                    rocblas_handle handle,
                                                    int64_t        n_64,
                                                    T              x,
                                                    rocblas_stride offset_x,
                                                    int64_t        inc_x,
                                                    rocblas_stride stride_x,
                                                    int64_t        batch_count_64,
                                                    const int      check_numerics,
                                                    bool           is_input)
{
    //Graph capture do not support any use of sync APIs.
    //Quick return: check numerics not supported
    if(handle->is_stream_in_capture_mode())
    {
        return rocblas_status_success;
    }

    //Quick return if possible. Not Argument error
    if(n_64 <= 0 || batch_count_64 <= 0 || !x)
    {
        return rocblas_status_success;
    }

    //Creating structure host object
    rocblas_check_numerics_t h_abnormal;

    //Allocating memory for device structure
    auto        d_abnormal     = handle->device_malloc(sizeof(rocblas_check_numerics_t));
    hipStream_t rocblas_stream = handle->get_stream();
    if(!d_abnormal)
    {
        rocblas_cerr << "rocBLAS internal error: No workspace memory available to allocate the "
                        "struct d_abnormal in "
                        "rocblas_check_numerics"
                     << std::endl;
        return rocblas_status_memory_error;
    }

    //Transferring the rocblas_check_numerics_t structure from host to the device
    RETURN_IF_HIP_ERROR(hipMemcpyAsync((rocblas_check_numerics_t*)d_abnormal,
                                       &h_abnormal,
                                       sizeof(rocblas_check_numerics_t),
                                       hipMemcpyHostToDevice,
                                       rocblas_stream));
    constexpr rocblas_int NB = 256;

    size_t abs_inc = inc_x < 0 ? -inc_x : inc_x;

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto    x_ptr       = adjust_ptr_batch(x, b_base, stride_x);
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));
        for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_X_chunk)
        {
            int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_X_chunk));

            dim3 blocks((n - 1) / NB + 1, batch_count);
            dim3 threads(NB);

            ROCBLAS_LAUNCH_KERNEL((rocblas_check_numerics_vector_kernel<NB>),
                                  blocks,
                                  threads,
                                  0,
                                  rocblas_stream,
                                  n,
                                  x_ptr,
                                  offset_x + abs_inc * n_base,
                                  abs_inc,
                                  stride_x,
                                  (rocblas_check_numerics_t*)d_abnormal);
        }
    }

    //Transferring the rocblas_check_numerics_t structure from device to the host
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&h_abnormal,
                                       (rocblas_check_numerics_t*)d_abnormal,
                                       sizeof(rocblas_check_numerics_t),
                                       hipMemcpyDeviceToHost,
                                       rocblas_stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(rocblas_stream));

    return rocblas_check_numerics_abnormal_struct(
        function_name, check_numerics, is_input, &h_abnormal);
}

// INSTANTIATIONS TO SUPPORT output: T*, T* const*, and input: const T*, const T* const*

#ifdef INST
#error INST IS ALREADY DEFINED
#endif
#define INST(typet_)                                                                   \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                           \
        rocblas_internal_check_numerics_vector_template(const char*    function_name,  \
                                                        rocblas_handle handle,         \
                                                        int64_t        n,              \
                                                        typet_         x,              \
                                                        rocblas_stride offset_x,       \
                                                        int64_t        incx,           \
                                                        rocblas_stride stride_x,       \
                                                        int64_t        batch_count,    \
                                                        const int      check_numerics, \
                                                        bool           is_input)
INST(float*);
INST(float* const*);
INST(float const*);
INST(float const* const*);

INST(double*);
INST(double* const*);
INST(double const*);
INST(double const* const*);

INST(rocblas_float_complex*);
INST(rocblas_float_complex* const*);
INST(rocblas_float_complex const*);
INST(rocblas_float_complex const* const*);

INST(rocblas_double_complex*);
INST(rocblas_double_complex* const*);
INST(rocblas_double_complex const*);
INST(rocblas_double_complex const* const*);

INST(rocblas_half*);
INST(rocblas_half* const*);
INST(rocblas_half const*);
INST(rocblas_half const* const*);

INST(rocblas_bfloat16*);
INST(rocblas_bfloat16* const*);
INST(rocblas_bfloat16 const*);
INST(rocblas_bfloat16 const* const*);

#undef INST
