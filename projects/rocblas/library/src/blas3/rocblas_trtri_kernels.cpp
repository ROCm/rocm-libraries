/* ************************************************************************
* Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "check_numerics_matrix.hpp"
#include "rocblas_trtri.hpp"

template <rocblas_int NB, bool BATCHED, typename T, typename U, typename V>
rocblas_status rocblas_internal_trtri_template(rocblas_handle   handle,
                                               rocblas_fill     uplo,
                                               rocblas_diagonal diag,
                                               rocblas_int      n,
                                               U                A,
                                               rocblas_stride   offset_A,
                                               rocblas_int      lda,
                                               rocblas_stride   stride_A,
                                               rocblas_stride   sub_stride_A,
                                               V                invA,
                                               rocblas_stride   offset_invA,
                                               rocblas_int      ldinvA,
                                               rocblas_stride   stride_invA,
                                               rocblas_stride   sub_stride_invA,
                                               rocblas_int      batch_count,
                                               rocblas_int      sub_batch_count,
                                               V                w_C_tmp)
{
    if(!n || !sub_batch_count)
        return rocblas_status_success;

    if(n <= NB)
    {
        return rocblas_trtri_small<NB, T>(handle,
                                          uplo,
                                          diag,
                                          n,
                                          A,
                                          offset_A,
                                          lda,
                                          stride_A,
                                          sub_stride_A,
                                          invA,
                                          offset_invA,
                                          ldinvA,
                                          stride_invA,
                                          sub_stride_invA,
                                          batch_count,
                                          sub_batch_count);
    }
    else
    {
        return rocblas_trtri_large<NB, BATCHED, T>(handle,
                                                   uplo,
                                                   diag,
                                                   n,
                                                   A,
                                                   offset_A,
                                                   lda,
                                                   stride_A,
                                                   sub_stride_A,
                                                   invA,
                                                   offset_invA,
                                                   ldinvA,
                                                   stride_invA,
                                                   sub_stride_invA,
                                                   batch_count,
                                                   sub_batch_count,
                                                   w_C_tmp);
    }
}

ROCBLAS_INTERNAL_EXPORT_NOINLINE size_t
    rocblas_internal_trtri_temp_elements(rocblas_int n, rocblas_int batch_count)
{
    rocblas_int IB   = ROCBLAS_TRTRI_NB * 2;
    size_t      size = 0;
    if(n > IB && batch_count > 0)
    {
        rocblas_int current_n = IB;
        while(current_n * 2 <= n)
            current_n *= 2;
        rocblas_int remainder = (n / IB) * IB - current_n;
        if(!rocblas_is_po2(remainder))
            remainder = rocblas_previous_po2(remainder);
        rocblas_int oddRemainder = n - current_n - remainder;

        size_t sizeRemainder = remainder ? remainder * current_n : 0;
        size_t sizeOdd       = 0;

        while(oddRemainder)
        {
            current_n         = n - oddRemainder;
            size_t curSizeOdd = oddRemainder * (n - oddRemainder);
            sizeOdd           = sizeOdd > curSizeOdd ? sizeOdd : curSizeOdd;

            if(!rocblas_is_po2(oddRemainder) && oddRemainder > IB)
            {
                oddRemainder = rocblas_previous_po2(oddRemainder);
                oddRemainder = n - current_n - oddRemainder;
            }
            else
            {
                oddRemainder = 0;
            }
        }

        if(sizeRemainder || sizeOdd)
            size = (sizeRemainder > sizeOdd ? sizeRemainder : sizeOdd) * batch_count;
    }
    return size;
}

#define TRTRI_TEMPLATE_PARAMS                                                                   \
    handle, uplo, diag, n, A, offset_A, lda, stride_A, sub_stride_A, invA, offset_invA, ldinvA, \
        stride_invA, sub_stride_invA, batch_count, sub_batch_count, w_C_tmp

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trtri_template(rocblas_handle   handle,
                                    rocblas_fill     uplo,
                                    rocblas_diagonal diag,
                                    rocblas_int      n,
                                    const T*         A,
                                    rocblas_stride   offset_A,
                                    rocblas_int      lda,
                                    rocblas_stride   stride_A,
                                    rocblas_stride   sub_stride_A,
                                    T*               invA,
                                    rocblas_stride   offset_invA,
                                    rocblas_int      ldinvA,
                                    rocblas_stride   stride_invA,
                                    rocblas_stride   sub_stride_invA,
                                    rocblas_int      batch_count,
                                    rocblas_int      sub_batch_count,
                                    T*               w_C_tmp)
{
    return rocblas_internal_trtri_template<ROCBLAS_TRTRI_NB, false, T>(TRTRI_TEMPLATE_PARAMS);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trtri_batched_template(rocblas_handle   handle,
                                            rocblas_fill     uplo,
                                            rocblas_diagonal diag,
                                            rocblas_int      n,
                                            const T* const*  A,
                                            rocblas_stride   offset_A,
                                            rocblas_int      lda,
                                            rocblas_stride   stride_A,
                                            rocblas_stride   sub_stride_A,
                                            T* const*        invA,
                                            rocblas_stride   offset_invA,
                                            rocblas_int      ldinvA,
                                            rocblas_stride   stride_invA,
                                            rocblas_stride   sub_stride_invA,
                                            rocblas_int      batch_count,
                                            rocblas_int      sub_batch_count,
                                            T* const*        w_C_tmp)
{
    return rocblas_internal_trtri_template<ROCBLAS_TRTRI_NB, true, T>(TRTRI_TEMPLATE_PARAMS);
}

#undef TRTRI_TEMPLATE_PARAMS

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_trtri_check_numerics(const char*    function_name,
                                            rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            rocblas_int    n,
                                            TConstPtr      A,
                                            rocblas_int    lda,
                                            rocblas_stride stride_a,
                                            TPtr           invA,
                                            rocblas_int    ldinvA,
                                            rocblas_stride stride_invA,
                                            rocblas_int    batch_count,
                                            const int      check_numerics,
                                            bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          uplo,
                                                          rocblas_client_triangular_matrix,
                                                          n,
                                                          n,
                                                          is_input ? A : invA,
                                                          0,
                                                          is_input ? lda : ldinvA,
                                                          is_input ? stride_a : stride_invA,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    return check_numerics_status;
}

// clang-format off
#ifdef INSTANTIATE_TRTRI_TEMPLATE
#error INSTANTIATE_TRTRI_TEMPLATE IS ALREADY DEFINED
#endif

#define INSTANTIATE_TRTRI_TEMPLATE(T_)                                                   \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                            \
    rocblas_status rocblas_internal_trtri_template<T_>(rocblas_handle   handle,          \
                                                       rocblas_fill     uplo,            \
                                                       rocblas_diagonal diag,            \
                                                       rocblas_int      n,               \
                                                       const T_*        A,               \
                                                       rocblas_stride   offset_A,        \
                                                       rocblas_int      lda,             \
                                                       rocblas_stride   stride_A,        \
                                                       rocblas_stride   sub_stride_A,    \
                                                       T_*              invA,            \
                                                       rocblas_stride   offset_invA,     \
                                                       rocblas_int      ldinvA,          \
                                                       rocblas_stride   stride_invA,     \
                                                       rocblas_stride   sub_stride_invA, \
                                                       rocblas_int      batch_count,     \
                                                       rocblas_int      sub_batch_count, \
                                                       T_*              w_C_tmp);

INSTANTIATE_TRTRI_TEMPLATE(float)
INSTANTIATE_TRTRI_TEMPLATE(double)
INSTANTIATE_TRTRI_TEMPLATE(rocblas_float_complex)
INSTANTIATE_TRTRI_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_TRTRI_TEMPLATE

#ifdef INSTANTIATE_TRTRI_BATCHED_TEMPLATE
#error INSTANTIATE_TRTRI_BATCHED_TEMPLATE IS ALREADY DEFINED
#endif

#define INSTANTIATE_TRTRI_BATCHED_TEMPLATE(T_)                                                   \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                                    \
    rocblas_status rocblas_internal_trtri_batched_template<T_>(rocblas_handle   handle,          \
                                                               rocblas_fill     uplo,            \
                                                               rocblas_diagonal diag,            \
                                                               rocblas_int      n,               \
                                                               const T_* const* A,               \
                                                               rocblas_stride   offset_A,        \
                                                               rocblas_int      lda,             \
                                                               rocblas_stride   stride_A,        \
                                                               rocblas_stride   sub_stride_A,    \
                                                               T_* const*       invA,            \
                                                               rocblas_stride   offset_invA,     \
                                                               rocblas_int      ldinvA,          \
                                                               rocblas_stride   stride_invA,     \
                                                               rocblas_stride   sub_stride_invA, \
                                                               rocblas_int      batch_count,     \
                                                               rocblas_int      sub_batch_count, \
                                                               T_* const*       w_C_tmp);

INSTANTIATE_TRTRI_BATCHED_TEMPLATE(float)
INSTANTIATE_TRTRI_BATCHED_TEMPLATE(double)
INSTANTIATE_TRTRI_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_TRTRI_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_TRTRI_BATCHED_TEMPLATE

#ifdef INSTANTIATE_TRTRI_CHECK_NUMERICS
#error INSTANTIATE_TRTRI_CHECK_NUMERICS IS ALREADY DEFINED
#endif

#define INSTANTIATE_TRTRI_CHECK_NUMERICS(T_, U_) template                          \
rocblas_status rocblas_trtri_check_numerics<T_, U_>(const char*    function_name,  \
                                                    rocblas_handle handle,         \
                                                    rocblas_fill   uplo,           \
                                                    rocblas_int    n,              \
                                                    T_             A,              \
                                                    rocblas_int    lda,            \
                                                    rocblas_stride stride_a,       \
                                                    U_             invA,           \
                                                    rocblas_int    ldinvA,         \
                                                    rocblas_stride stride_invA,    \
                                                    rocblas_int    batch_count,    \
                                                    const int      check_numerics, \
                                                    bool           is_input);

INSTANTIATE_TRTRI_CHECK_NUMERICS(const float*, float*)
INSTANTIATE_TRTRI_CHECK_NUMERICS(const double*, double*)
INSTANTIATE_TRTRI_CHECK_NUMERICS(const rocblas_float_complex*, rocblas_float_complex*)
INSTANTIATE_TRTRI_CHECK_NUMERICS(const rocblas_double_complex*, rocblas_double_complex*)

INSTANTIATE_TRTRI_CHECK_NUMERICS(const float* const*, float* const*)
INSTANTIATE_TRTRI_CHECK_NUMERICS(const double* const*, double* const*)
INSTANTIATE_TRTRI_CHECK_NUMERICS(const rocblas_float_complex* const*, rocblas_float_complex* const*)
INSTANTIATE_TRTRI_CHECK_NUMERICS(const rocblas_double_complex* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TRTRI_CHECK_NUMERICS
