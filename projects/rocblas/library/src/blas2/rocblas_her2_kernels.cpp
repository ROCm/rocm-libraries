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

#include "rocblas_her2_kernels.hpp"
#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas_her2.hpp"

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_her2_template(rocblas_handle handle,
                                   rocblas_fill   uplo,
                                   rocblas_int    n,
                                   const T*       alpha,
                                   const T*       x,
                                   rocblas_stride offset_x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   const T*       y,
                                   rocblas_stride offset_y,
                                   rocblas_int    incy,
                                   rocblas_stride stride_y,
                                   T*             A,
                                   rocblas_int    lda,
                                   rocblas_stride offset_A,
                                   rocblas_stride stride_A,
                                   rocblas_int    batch_count)
{
    return rocblas_her2_launcher(handle,
                                 uplo,
                                 n,
                                 alpha,
                                 x,
                                 offset_x,
                                 incx,
                                 stride_x,
                                 y,
                                 offset_y,
                                 incy,
                                 stride_y,
                                 A,
                                 offset_A,
                                 lda,
                                 stride_A,
                                 batch_count);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_her2_batched_template(rocblas_handle  handle,
                                           rocblas_fill    uplo,
                                           rocblas_int     n,
                                           const T*        alpha,
                                           const T* const* x,
                                           rocblas_stride  offset_x,
                                           rocblas_int     incx,
                                           rocblas_stride  stride_x,
                                           const T* const* y,
                                           rocblas_stride  offset_y,
                                           rocblas_int     incy,
                                           rocblas_stride  stride_y,
                                           T* const*       A,
                                           rocblas_int     lda,
                                           rocblas_stride  offset_A,
                                           rocblas_stride  stride_A,
                                           rocblas_int     batch_count)
{
    return rocblas_her2_launcher(handle,
                                 uplo,
                                 n,
                                 alpha,
                                 x,
                                 offset_x,
                                 incx,
                                 stride_x,
                                 y,
                                 offset_y,
                                 incy,
                                 stride_y,
                                 A,
                                 offset_A,
                                 lda,
                                 stride_A,
                                 batch_count);
}

template <typename T, typename U>
rocblas_status rocblas_her2_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_fill   uplo,
                                           int64_t        n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           int64_t        lda,
                                           rocblas_stride stride_a,
                                           U              x,
                                           rocblas_stride offset_x,
                                           int64_t        inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
                                           int64_t        inc_y,
                                           rocblas_stride stride_y,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{

    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          uplo,
                                                          rocblas_client_hermitian_matrix,
                                                          n,
                                                          n,
                                                          A,
                                                          offset_a,
                                                          lda,
                                                          stride_a,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    if(is_input)
    {
        check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
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
    }
    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *her2*.cpp

#ifdef INSTANTIATE_HER2_NUMERICS
#error INSTANTIATE_HER2_NUMERICS already defined
#endif

#define INSTANTIATE_HER2_NUMERICS(T_, U_)                                                      \
    template rocblas_status rocblas_her2_check_numerics<T_, U_>(const char*    function_name,  \
                                                                rocblas_handle handle,         \
                                                                rocblas_fill   uplo,           \
                                                                int64_t        n,              \
                                                                T_             A,              \
                                                                rocblas_stride offset_a,       \
                                                                int64_t        lda,            \
                                                                rocblas_stride stride_a,       \
                                                                U_             x,              \
                                                                rocblas_stride offset_x,       \
                                                                int64_t        inc_x,          \
                                                                rocblas_stride stride_x,       \
                                                                U_             y,              \
                                                                rocblas_stride offset_y,       \
                                                                int64_t        inc_y,          \
                                                                rocblas_stride stride_y,       \
                                                                int64_t        batch_count,    \
                                                                const int      check_numerics, \
                                                                bool           is_input);

INSTANTIATE_HER2_NUMERICS(rocblas_float_complex*, rocblas_float_complex const*)
INSTANTIATE_HER2_NUMERICS(rocblas_float_complex* const*, rocblas_float_complex const* const*)
INSTANTIATE_HER2_NUMERICS(rocblas_double_complex*, rocblas_double_complex const*)
INSTANTIATE_HER2_NUMERICS(rocblas_double_complex* const*, rocblas_double_complex const* const*)

#undef INSTANTIATE_HER2_NUMERICS

#ifdef INSTANTIATE_HER2_TEMPLATE
#error INSTANTIATE_HER2_TEMPLATE already defined
#endif

#define INSTANTIATE_HER2_TEMPLATE(T_)                                                            \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_her2_template<T_>( \
        rocblas_handle handle,                                                                   \
        rocblas_fill   uplo,                                                                     \
        rocblas_int    n,                                                                        \
        const T_*      alpha,                                                                    \
        const T_*      x,                                                                        \
        rocblas_stride offset_x,                                                                 \
        rocblas_int    incx,                                                                     \
        rocblas_stride stride_x,                                                                 \
        const T_*      y,                                                                        \
        rocblas_stride offset_y,                                                                 \
        rocblas_int    incy,                                                                     \
        rocblas_stride stride_y,                                                                 \
        T_*            A,                                                                        \
        rocblas_int    lda,                                                                      \
        rocblas_stride offset_A,                                                                 \
        rocblas_stride stride_A,                                                                 \
        rocblas_int    batch_count);

INSTANTIATE_HER2_TEMPLATE(rocblas_float_complex)
INSTANTIATE_HER2_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_HER2_TEMPLATE

#ifdef INSTANTIATE_HER2_BATCHED_TEMPLATE
#error INSTANTIATE_HER2_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_HER2_BATCHED_TEMPLATE(T_)                                 \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                  \
        rocblas_internal_her2_batched_template<T_>(rocblas_handle   handle,   \
                                                   rocblas_fill     uplo,     \
                                                   rocblas_int      n,        \
                                                   const T_*        alpha,    \
                                                   const T_* const* x,        \
                                                   rocblas_stride   offset_x, \
                                                   rocblas_int      incx,     \
                                                   rocblas_stride   stride_x, \
                                                   const T_* const* y,        \
                                                   rocblas_stride   offset_y, \
                                                   rocblas_int      incy,     \
                                                   rocblas_stride   stride_y, \
                                                   T_* const*       A,        \
                                                   rocblas_int      lda,      \
                                                   rocblas_stride   offset_A, \
                                                   rocblas_stride   stride_A, \
                                                   rocblas_int      batch_count);

INSTANTIATE_HER2_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_HER2_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_HER2_BATCHED_TEMPLATE
