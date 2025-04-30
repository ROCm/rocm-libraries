/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstring> // std::memcpy for graph capture use cases

#ifdef BUILD_WITH_TENSILE
#include "gemm_tensile.hpp"
#endif
#include "blas3/rocblas_gemm_source.hpp"

#include "blas3/rocblas_gemm.hpp"

#include "check_numerics_matrix.hpp"
#include "handle.hpp"
#include "logging.hpp"

/*
 * ===========================================================================
 *    template interface
 * ===========================================================================
 */
template <bool BATCHED, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_gemm(rocblas_handle    handle,
                                     rocblas_operation trans_a,
                                     rocblas_operation trans_b,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     rocblas_int       k,
                                     const TScal*      alpha,
                                     TConstPtr         A,
                                     rocblas_stride    offset_a,
                                     rocblas_int       lda,
                                     rocblas_stride    stride_a,
                                     TConstPtr         B,
                                     rocblas_stride    offset_b,
                                     rocblas_int       ldb,
                                     rocblas_stride    stride_b,
                                     const TScal*      beta,
                                     TPtr              C,
                                     rocblas_stride    offset_c,
                                     rocblas_int       ldc,
                                     rocblas_stride    stride_c,
                                     rocblas_int       batch_count)
{
    rocblas_status status = rocblas_status_success;

    // quick return 0 is valid in BLAS
    // Note: k==0 is not a quick return, because C must still be multiplied by beta
    if(!m || !n || !batch_count)
        return status;

    TScal alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        rocblas_copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));
    auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

#ifdef BUILD_WITH_TENSILE

    if(BATCHED)
    {
        status = rocblas_call_tensile(handle,
                                      alpha,
                                      beta,
                                      A,
                                      B,
                                      C,
                                      C, // gemm uses C matrix for output D
                                      trans_a,
                                      trans_b,
                                      ldc, // gemm uses C matrix for output D
                                      stride_c,
                                      offset_c,
                                      ldc,
                                      stride_c,
                                      offset_c,
                                      lda,
                                      stride_a,
                                      offset_a,
                                      ldb,
                                      stride_b,
                                      offset_b,
                                      m,
                                      n,
                                      k,
                                      batch_count);
    }
    else
    {
        status = rocblas_call_tensile(handle,
                                      alpha,
                                      beta,
                                      A + offset_a,
                                      B + offset_b,
                                      C + offset_c,
                                      C + offset_c,
                                      trans_a,
                                      trans_b,
                                      ldc,
                                      stride_c,
                                      0,
                                      ldc,
                                      stride_c,
                                      0,
                                      lda,
                                      stride_a,
                                      0,
                                      ldb,
                                      stride_b,
                                      0,
                                      m,
                                      n,
                                      k,
                                      batch_count);
    }

    // Return the current status if an exception is thrown
    // by other libraries(hipBLASLt or Tensile).
    // Otherwise, fall through to the rocBLAS source GEMM implementation.
    if(status != rocblas_status_not_implemented)
    {
        return status;
    }

#endif // BUILD_WITH_TENSILE

    bool backend_logging = handle->layer_mode & rocblas_layer_mode_log_internal;
    if(backend_logging)
    {
        rocblas_internal_logger logger;
        logger.log_trace(handle,
                         c_rocblas_internal,
                         "rocblas_gemm_source_backend",
                         trans_a,
                         trans_b,
                         m,
                         n,
                         k,
                         LOG_TRACE_SCALAR_VALUE(handle, alpha),
                         A,
                         lda,
                         B,
                         ldb,
                         LOG_TRACE_SCALAR_VALUE(handle, beta),
                         C,
                         ldc);
    }

    if(k == 0 || (alpha && *alpha == 0))
    {
        return rocblas_gemm_scale_launcher_64(
            handle, m, n, *beta, C, offset_c, ldc, stride_c, batch_count);
    }

    status = rocblas_status_success;
    for(int64_t n_base = 0; n_base < n; n_base += c_i64_grid_YZ_chunk)
    {
        // don't need to block through M as it's 32 bit and can use full 32-bits in X-dim of grid
        int32_t nblock = int32_t(std::min(n - n_base, c_i64_grid_YZ_chunk));

        status = rocblas_gemm_source_solution_64<BATCHED>(
            handle,
            trans_a,
            trans_b,
            m,
            nblock,
            k,
            *alpha,
            A,
            lda,
            stride_a,
            offset_a,
            B,
            ldb,
            stride_b,
            offset_b + (trans_b == rocblas_operation_none ? n_base * ldb : n_base),
            *beta,
            (TConstPtr)C,
            ldc,
            stride_c,
            offset_c + n_base * ldc,
            C,
            ldc,
            stride_c,
            offset_c + n_base * ldc,
            batch_count);

        if(status != rocblas_status_success)
            return status;
    }

    return status;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gemm_batched_template(rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           rocblas_int       m,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           const T*          alpha,
                                           const T* const*   A,
                                           rocblas_stride    offset_a,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_a,
                                           const T* const*   B,
                                           rocblas_stride    offset_b,
                                           rocblas_int       ldb,
                                           rocblas_stride    stride_b,
                                           const T*          beta,
                                           T* const*         C,
                                           rocblas_stride    offset_c,
                                           rocblas_int       ldc,
                                           rocblas_stride    stride_c,
                                           rocblas_int       batch_count)
{
    return rocblas_internal_gemm<true>(handle,
                                       trans_a,
                                       trans_b,
                                       m,
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       offset_a,
                                       lda,
                                       stride_a,
                                       B,
                                       offset_b,
                                       ldb,
                                       stride_b,
                                       beta,
                                       C,
                                       offset_c,
                                       ldc,
                                       stride_c,
                                       batch_count);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gemm_template(rocblas_handle    handle,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const T*          alpha,
                                   const T*          A,
                                   rocblas_stride    offset_a,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_a,
                                   const T*          B,
                                   rocblas_stride    offset_b,
                                   rocblas_int       ldb,
                                   rocblas_stride    stride_b,
                                   const T*          beta,
                                   T*                C,
                                   rocblas_stride    offset_c,
                                   rocblas_int       ldc,
                                   rocblas_stride    stride_c,
                                   rocblas_int       batch_count)
{
    return rocblas_internal_gemm<false>(handle,
                                        trans_a,
                                        trans_b,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        A,
                                        offset_a,
                                        lda,
                                        stride_a,
                                        B,
                                        offset_b,
                                        ldb,
                                        stride_b,
                                        beta,
                                        C,
                                        offset_c,
                                        ldc,
                                        stride_c,
                                        batch_count);
}

#ifdef INSTANTIATE_GEMM_TEMPLATE
#error INSTANTIATE_GEMM_TEMPLATE already defined
#endif

#define INSTANTIATE_GEMM_TEMPLATE(T_)                                                        \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_gemm_template( \
        rocblas_handle    handle,                                                            \
        rocblas_operation trans_a,                                                           \
        rocblas_operation trans_b,                                                           \
        rocblas_int       m,                                                                 \
        rocblas_int       n,                                                                 \
        rocblas_int       k,                                                                 \
        const T_*         alpha,                                                             \
        const T_*         A,                                                                 \
        rocblas_stride    offset_a,                                                          \
        rocblas_int       lda,                                                               \
        rocblas_stride    stride_a,                                                          \
        const T_*         B,                                                                 \
        rocblas_stride    offset_b,                                                          \
        rocblas_int       ldb,                                                               \
        rocblas_stride    stride_b,                                                          \
        const T_*         beta,                                                              \
        T_*               C,                                                                 \
        rocblas_stride    offset_c,                                                          \
        rocblas_int       ldc,                                                               \
        rocblas_stride    stride_c,                                                          \
        rocblas_int       batch_count);

INSTANTIATE_GEMM_TEMPLATE(rocblas_half)
INSTANTIATE_GEMM_TEMPLATE(float)
INSTANTIATE_GEMM_TEMPLATE(double)
INSTANTIATE_GEMM_TEMPLATE(rocblas_float_complex)
INSTANTIATE_GEMM_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_GEMM_TEMPLATE

#ifdef INSTANTIATE_GEMM_BATCHED_TEMPLATE
#error INSTANTIATE_GEMM_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_GEMM_BATCHED_TEMPLATE(T_)                              \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status               \
        rocblas_internal_gemm_batched_template(rocblas_handle    handle,   \
                                               rocblas_operation trans_a,  \
                                               rocblas_operation trans_b,  \
                                               rocblas_int       m,        \
                                               rocblas_int       n,        \
                                               rocblas_int       k,        \
                                               const T_*         alpha,    \
                                               const T_* const*  A,        \
                                               rocblas_stride    offset_a, \
                                               rocblas_int       lda,      \
                                               rocblas_stride    stride_a, \
                                               const T_* const*  B,        \
                                               rocblas_stride    offset_b, \
                                               rocblas_int       ldb,      \
                                               rocblas_stride    stride_b, \
                                               const T_*         beta,     \
                                               T_* const*        C,        \
                                               rocblas_stride    offset_c, \
                                               rocblas_int       ldc,      \
                                               rocblas_stride    stride_c, \
                                               rocblas_int       batch_count);

INSTANTIATE_GEMM_BATCHED_TEMPLATE(rocblas_half)
INSTANTIATE_GEMM_BATCHED_TEMPLATE(float)
INSTANTIATE_GEMM_BATCHED_TEMPLATE(double)
INSTANTIATE_GEMM_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_GEMM_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_GEMM_BATCHED_TEMPLATE
