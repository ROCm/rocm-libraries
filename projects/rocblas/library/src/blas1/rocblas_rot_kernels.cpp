/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas_rot_kernels.hpp"
#include "rocblas_rot.hpp"

template <typename T>
rocblas_status rocblas_rot_check_numerics(const char*    function_name,
                                          rocblas_handle handle,
                                          int64_t        n,
                                          T              x,
                                          rocblas_stride offset_x,
                                          int64_t        inc_x,
                                          rocblas_stride stride_x,
                                          T              y,
                                          rocblas_stride offset_y,
                                          int64_t        inc_y,
                                          rocblas_stride stride_y,
                                          int64_t        batch_count,
                                          const int      check_numerics,
                                          bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          n,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n,
                                                                            y,
                                                                            offset_y,
                                                                            inc_y,
                                                                            stride_y,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);

    return check_numerics_status;
}

// If there are any changes in template parameters in the files *rot*.cpp
// instantiations below will need to be manually updated to match the changes.

// clang-format off
#ifdef INSTANTIATE_ROT_CHECK_NUMERICS
#error INSTANTIATE_ROT_CHECK_NUMERICS already defined
#endif

#define INSTANTIATE_ROT_CHECK_NUMERICS(T_)                               \
template rocblas_status rocblas_rot_check_numerics<T_>                   \
                                         (const char*    function_name,  \
                                          rocblas_handle handle,         \
                                          int64_t    n,              \
                                          T_             x,              \
                                          rocblas_stride offset_x,       \
                                          int64_t    inc_x,          \
                                          rocblas_stride stride_x,       \
                                          T_             y,              \
                                          rocblas_stride offset_y,       \
                                          int64_t    inc_y,          \
                                          rocblas_stride stride_y,       \
                                          int64_t    batch_count,    \
                                          const int      check_numerics, \
                                          bool           is_input);

//  instantiate for rocblas_Xrot and rocblas_Xrot_strided_batched
INSTANTIATE_ROT_CHECK_NUMERICS(float* )
INSTANTIATE_ROT_CHECK_NUMERICS(double* )
INSTANTIATE_ROT_CHECK_NUMERICS(rocblas_float_complex* )
INSTANTIATE_ROT_CHECK_NUMERICS(rocblas_double_complex* )

//  instantiate for rocblas_Xrot__batched
INSTANTIATE_ROT_CHECK_NUMERICS(float* const*)
INSTANTIATE_ROT_CHECK_NUMERICS(double* const*)
INSTANTIATE_ROT_CHECK_NUMERICS(rocblas_float_complex* const*)
INSTANTIATE_ROT_CHECK_NUMERICS(rocblas_double_complex* const*)

//  instantiate for rocblas_Xrot_ex and rocblas_Xrot_strided_batched_ex
INSTANTIATE_ROT_CHECK_NUMERICS(rocblas_bfloat16*)
INSTANTIATE_ROT_CHECK_NUMERICS(rocblas_half*)

//  instantiate for rocblas_Xrot__batched_ex
INSTANTIATE_ROT_CHECK_NUMERICS(rocblas_bfloat16* const*)
INSTANTIATE_ROT_CHECK_NUMERICS(rocblas_half* const*)

#undef INSTANTIATE_ROT_CHECK_NUMERICS

template <typename API_INT,
          rocblas_int NB,
          typename Tex,
          typename Tx,
          typename Ty,
          typename Tc,
          typename Ts>
rocblas_status rocblas_internal_rot_launcher(rocblas_handle handle,
                                             API_INT        n,
                                             Tx             x,
                                             rocblas_stride offset_x,
                                             int64_t        incx,
                                             rocblas_stride stride_x,
                                             Ty             y,
                                             rocblas_stride offset_y,
                                             int64_t        incy,
                                             rocblas_stride stride_y,
                                             Tc*            c,
                                             rocblas_stride c_stride,
                                             Ts*            s,
                                             rocblas_stride s_stride,
                                             API_INT        batch_count)
{
    // Quick return if possible
    if(n <= 0 || batch_count <= 0)
        return rocblas_status_success;

    int64_t shiftx = incx < 0 ? offset_x - int64_t(incx) * (n - 1) : offset_x;
    int64_t shifty = incy < 0 ? offset_y - int64_t(incy) * (n - 1) : offset_y;

    int batches = handle->getBatchGridDim((int)batch_count);

    dim3        blocks((n - 1) / NB + 1, 1, batches);
    dim3        threads(NB);
    hipStream_t rocblas_stream = handle->get_stream();

    if(rocblas_pointer_mode_device == handle->pointer_mode)
        ROCBLAS_LAUNCH_KERNEL((rocblas_rot_kernel<API_INT, NB, Tex>),
                              blocks,
                              threads,
                              0,
                              rocblas_stream,
                              n,
                              x,
                              shiftx,
                              incx,
                              stride_x,
                              y,
                              shifty,
                              incy,
                              stride_y,
                              c,
                              c_stride,
                              s,
                              s_stride, batch_count);
    else // c and s are on host
        ROCBLAS_LAUNCH_KERNEL((rocblas_rot_kernel<API_INT, NB, Tex>),
                              blocks,
                              threads,
                              0,
                              rocblas_stream,
                              n,
                              x,
                              shiftx,
                              incx,
                              stride_x,
                              y,
                              shifty,
                              incy,
                              stride_y,
                              *c,
                              c_stride,
                              *s,
                              s_stride, batch_count);

    return rocblas_status_success;
}

#ifdef INSTANTIATE_ROT_LAUNCHER
#error INSTANTIATE_ROT_LAUNCHER already defined
#endif

#define INSTANTIATE_ROT_LAUNCHER(NB_, Tex_, Tx_, Ty_, Tc_, Ts_)                    \
    template rocblas_status                                                        \
        rocblas_internal_rot_launcher<rocblas_int, NB_, Tex_, Tx_, Ty_, Tc_, Ts_>( \
            rocblas_handle handle,                                                 \
            rocblas_int    n,                                                      \
            Tx_            x,                                                      \
            rocblas_stride offset_x,                                               \
            int64_t    incx,                                                   \
            rocblas_stride stride_x,                                               \
            Ty_            y,                                                      \
            rocblas_stride offset_y,                                               \
            int64_t    incy,                                                   \
            rocblas_stride stride_y,                                               \
            Tc_ * c,                                                               \
            rocblas_stride c_stride,                                               \
            Ts_ * s,                                                               \
            rocblas_stride s_stride,                                               \
            rocblas_int    batch_count);


//  instantiate for rocblas_Xrot and rocblas_Xrot_strided_batched
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB,  float,  float*,         float*,         float const,  float const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, double, double*,        double*,        double const, double const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, float, rocblas_bfloat16*,        rocblas_bfloat16*,        rocblas_bfloat16 const, rocblas_bfloat16 const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, float,     rocblas_half*,            rocblas_half*,            rocblas_half const,     rocblas_half const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, rocblas_float_complex, rocblas_float_complex*,        rocblas_float_complex*,                        float const,                 float const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, rocblas_float_complex, rocblas_float_complex*,        rocblas_float_complex*,                        float const, rocblas_float_complex const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, rocblas_float_complex, rocblas_float_complex*,        rocblas_float_complex*,        rocblas_float_complex const, rocblas_float_complex const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, rocblas_double_complex, rocblas_double_complex*,               rocblas_double_complex*, double const, double const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, rocblas_double_complex, rocblas_double_complex*,               rocblas_double_complex*, double const, rocblas_double_complex const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, rocblas_double_complex, rocblas_double_complex*,               rocblas_double_complex*, rocblas_double_complex const, rocblas_double_complex const)

//  instantiate for rocblas_Xrot__batched
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB,  float,  float* const*,  float* const*,  float const,  float const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, double, double* const*, double* const*, double const, double const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, float, rocblas_bfloat16* const*, rocblas_bfloat16* const*, rocblas_bfloat16 const, rocblas_bfloat16 const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, float,     rocblas_half* const*,     rocblas_half* const*,     rocblas_half const,     rocblas_half const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, rocblas_float_complex, rocblas_float_complex* const*, rocblas_float_complex* const*,                 float const,                 float const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, rocblas_float_complex, rocblas_float_complex* const*, rocblas_float_complex* const*,                 float const, rocblas_float_complex const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, rocblas_float_complex, rocblas_float_complex* const*, rocblas_float_complex* const*, rocblas_float_complex const, rocblas_float_complex const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, rocblas_double_complex, rocblas_double_complex* const*, rocblas_double_complex* const*, rocblas_double_complex const, rocblas_double_complex const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, rocblas_double_complex, rocblas_double_complex* const*, rocblas_double_complex* const*, double const, double const)
INSTANTIATE_ROT_LAUNCHER(ROCBLAS_ROT_NB, rocblas_double_complex, rocblas_double_complex* const*, rocblas_double_complex* const*, double const, rocblas_double_complex const)

#undef INSTANTIATE_ROT_LAUNCHER
// clang-format on
