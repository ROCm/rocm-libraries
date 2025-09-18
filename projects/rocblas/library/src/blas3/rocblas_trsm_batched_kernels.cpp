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

#include "rocblas_trsm_kernels.hpp"

#define TRSM_TEMPLATE_PARAMS                                                                     \
    handle, side, uplo, transA, diag, m, n, alpha, A, offset_A, lda, stride_A, B, offset_B, ldb, \
        stride_B, batch_count, optimal_mem, w_x_temp, w_x_temparr, invA, invAarr, supplied_invA, \
        supplied_invA_size, offset_invA, stride_invA

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_batched_template(rocblas_handle    handle,
                                           rocblas_side      side,
                                           rocblas_fill      uplo,
                                           rocblas_operation transA,
                                           rocblas_diagonal  diag,
                                           rocblas_int       m,
                                           rocblas_int       n,
                                           const T*          alpha,
                                           const T* const*   A,
                                           rocblas_stride    offset_A,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_A,
                                           T* const*         B,
                                           rocblas_stride    offset_B,
                                           rocblas_int       ldb,
                                           rocblas_stride    stride_B,
                                           rocblas_int       batch_count,
                                           bool              optimal_mem,
                                           void*             w_x_temp,
                                           void*             w_x_temparr,
                                           void*             invA,
                                           void*             invAarr,
                                           const T* const*   supplied_invA,
                                           rocblas_int       supplied_invA_size,
                                           rocblas_stride    offset_invA,
                                           rocblas_stride    stride_invA)
{
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_trsm_launcher<ROCBLAS_TRSM_NB, ROCBLAS_SDCTRSV_NB, true, T>(
            TRSM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_trsm_launcher<ROCBLAS_TRSM_NB, ROCBLAS_SDCTRSV_NB, true, T>(
            TRSM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_trsm_launcher<ROCBLAS_TRSM_NB, ROCBLAS_SDCTRSV_NB, true, T>(
            TRSM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_trsm_launcher<ROCBLAS_TRSM_NB, ROCBLAS_ZTRSV_NB, true, T>(
            TRSM_TEMPLATE_PARAMS);

    return rocblas_status_not_implemented;
}

#undef TRSM_TEMPLATE_PARAMS

#ifdef INSTANTIATE_TRSM_BATCHED_TEMPLATE
#error INSTANTIATE_TRSM_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_TRSM_BATCHED_TEMPLATE(T_)                                            \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                             \
        rocblas_internal_trsm_batched_template<T_>(rocblas_handle    handle,             \
                                                   rocblas_side      side,               \
                                                   rocblas_fill      uplo,               \
                                                   rocblas_operation transA,             \
                                                   rocblas_diagonal  diag,               \
                                                   rocblas_int       m,                  \
                                                   rocblas_int       n,                  \
                                                   const T_*         alpha,              \
                                                   const T_* const*  A,                  \
                                                   rocblas_stride    offset_A,           \
                                                   rocblas_int       lda,                \
                                                   rocblas_stride    stride_A,           \
                                                   T_* const*        B,                  \
                                                   rocblas_stride    offset_B,           \
                                                   rocblas_int       ldb,                \
                                                   rocblas_stride    stride_B,           \
                                                   rocblas_int       batch_count,        \
                                                   bool              optimal_mem,        \
                                                   void*             w_x_temp,           \
                                                   void*             w_x_temparr,        \
                                                   void*             invA,               \
                                                   void*             invAarr,            \
                                                   const T_* const*  supplied_invA,      \
                                                   rocblas_int       supplied_invA_size, \
                                                   rocblas_stride    offset_invA,        \
                                                   rocblas_stride    stride_invA);

INSTANTIATE_TRSM_BATCHED_TEMPLATE(float)
INSTANTIATE_TRSM_BATCHED_TEMPLATE(double)
INSTANTIATE_TRSM_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_TRSM_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_TRSM_BATCHED_TEMPLATE

#ifdef INSTANTIATE_TRSM_BATCHED_WORKSPACE_TEMPLATE
#error INSTANTIATE_TRSM_BATCHED_WORKSPACE_TEMPLATE already defined
#endif

#define INSTANTIATE_TRSM_BATCHED_WORKSPACE_TEMPLATE(T_)                                        \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                   \
        rocblas_internal_trsm_batched_workspace_size<T_>(rocblas_side      side,               \
                                                         rocblas_operation transA,             \
                                                         rocblas_int       m,                  \
                                                         rocblas_int       n,                  \
                                                         rocblas_int       batch_count,        \
                                                         rocblas_int       supplied_invA_size, \
                                                         size_t * w_x_tmp_size,                \
                                                         size_t * w_x_tmp_arr_size,            \
                                                         size_t * w_invA_size,                 \
                                                         size_t * w_invA_arr_size,             \
                                                         size_t * w_x_tmp_size_backup);

INSTANTIATE_TRSM_BATCHED_WORKSPACE_TEMPLATE(float)
INSTANTIATE_TRSM_BATCHED_WORKSPACE_TEMPLATE(double)
INSTANTIATE_TRSM_BATCHED_WORKSPACE_TEMPLATE(rocblas_float_complex)
INSTANTIATE_TRSM_BATCHED_WORKSPACE_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_TRSM_BATCHED_WORKSPACE_TEMPLATE

INSTANTIATE_TRSM_MEM_TEMPLATE(true, float, const float* const*)
INSTANTIATE_TRSM_MEM_TEMPLATE(true, double, const double* const*)
INSTANTIATE_TRSM_MEM_TEMPLATE(true, rocblas_float_complex, const rocblas_float_complex* const*)
INSTANTIATE_TRSM_MEM_TEMPLATE(true, rocblas_double_complex, const rocblas_double_complex* const*)
