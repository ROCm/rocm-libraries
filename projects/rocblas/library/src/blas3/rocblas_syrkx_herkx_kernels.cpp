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

#include "definitions.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_gemm.hpp"
#include "rocblas_syr2k_her2k.hpp"
#include "rocblas_syrkx_herkx.hpp"

#define OFFSET_A(i1) offset_a + i1* rocblas_stride(a_s1)
#define OFFSET_B(i1) offset_b + i1* rocblas_stride(b_s1)
#define OFFSET_C(i1, i2) offset_c + i1* rocblas_stride(c_s1) + i2* rocblas_stride(c_s2)

template <typename API_INT,
          int  MIN_NB,
          bool BATCHED,
          bool HERK,
          typename T,
          typename TScala,
          typename TScalb,
          typename TPtr,
          typename TConstPtr>
rocblas_status rocblas_internal_syrkx_herkx_template(rocblas_handle    handle,
                                                     rocblas_fill      uplo,
                                                     rocblas_operation trans,
                                                     rocblas_int       n,
                                                     API_INT           k,
                                                     const TScala*     alpha,
                                                     TConstPtr*        da,
                                                     rocblas_stride    offset_a,
                                                     API_INT           lda,
                                                     rocblas_stride    stride_a,
                                                     TConstPtr*        db,
                                                     rocblas_stride    offset_b,
                                                     API_INT           ldb,
                                                     rocblas_stride    stride_b,
                                                     const TScalb*     beta,
                                                     TPtr*             dc,
                                                     rocblas_stride    offset_c,
                                                     API_INT           ldc,
                                                     rocblas_stride    stride_c,
                                                     rocblas_int       batch_count)
{
    static constexpr bool TWOK = false;
    return rocblas_internal_syr2k_her2k_template<rocblas_int, MIN_NB, BATCHED, TWOK, HERK, T>(
        handle,
        uplo,
        trans,
        n,
        k,
        alpha,
        da,
        offset_a,
        lda,
        stride_a,
        db,
        offset_b,
        ldb,
        stride_b,
        beta,
        dc,
        offset_c,
        ldc,
        stride_c,
        batch_count);
}
#undef OFFSET_A
#undef OFFSET_B
#undef OFFSET_C

#define ROCBLAS_INTERNAL_SYRKX_HERKX_PARAMS                                                   \
    handle, uplo, trans, n, k, alpha, A, offset_a, lda, stride_a, B, offset_b, ldb, stride_b, \
        beta, C, offset_c, ldc, stride_c, batch_count

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrkx_template(rocblas_handle    handle,
                                    rocblas_fill      uplo,
                                    rocblas_operation trans,
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
    constexpr bool BATCHED = false;
    constexpr bool HERM    = false;
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_syrkx_herkx_template<rocblas_int,
                                                     ROCBLAS_SSYRKX_NB,
                                                     BATCHED,
                                                     HERM,
                                                     T>(ROCBLAS_INTERNAL_SYRKX_HERKX_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_syrkx_herkx_template<rocblas_int,
                                                     ROCBLAS_DCZSYRKX_NB,
                                                     BATCHED,
                                                     HERM,
                                                     T>(ROCBLAS_INTERNAL_SYRKX_HERKX_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syrkx_herkx_template<rocblas_int,
                                                     ROCBLAS_DCZSYRKX_NB,
                                                     BATCHED,
                                                     HERM,
                                                     T>(ROCBLAS_INTERNAL_SYRKX_HERKX_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syrkx_herkx_template<rocblas_int,
                                                     ROCBLAS_DCZSYRKX_NB,
                                                     BATCHED,
                                                     HERM,
                                                     T>(ROCBLAS_INTERNAL_SYRKX_HERKX_PARAMS);

    return rocblas_status_not_implemented;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrkx_batched_template(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation trans,
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
    constexpr bool BATCHED = true;
    constexpr bool HERM    = false;
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_syrkx_herkx_template<rocblas_int,
                                                     ROCBLAS_SDSYRKX_BATCHED_NB,
                                                     BATCHED,
                                                     HERM,
                                                     T>(ROCBLAS_INTERNAL_SYRKX_HERKX_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_syrkx_herkx_template<rocblas_int,
                                                     ROCBLAS_SDSYRKX_BATCHED_NB,
                                                     BATCHED,
                                                     HERM,
                                                     T>(ROCBLAS_INTERNAL_SYRKX_HERKX_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syrkx_herkx_template<rocblas_int,
                                                     ROCBLAS_CZSYRKX_BATCHED_NB,
                                                     BATCHED,
                                                     HERM,
                                                     T>(ROCBLAS_INTERNAL_SYRKX_HERKX_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syrkx_herkx_template<rocblas_int,
                                                     ROCBLAS_CZSYRKX_BATCHED_NB,
                                                     BATCHED,
                                                     HERM,
                                                     T>(ROCBLAS_INTERNAL_SYRKX_HERKX_PARAMS);

    return rocblas_status_not_implemented;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herkx_template(rocblas_handle    handle,
                                    rocblas_fill      uplo,
                                    rocblas_operation trans,
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
                                    const real_t<T>*  beta,
                                    T*                C,
                                    rocblas_stride    offset_c,
                                    rocblas_int       ldc,
                                    rocblas_stride    stride_c,
                                    rocblas_int       batch_count)
{
    constexpr bool BATCHED = false;
    constexpr bool HERM    = true;
    if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syrkx_herkx_template<rocblas_int,
                                                     ROCBLAS_HERKX_NB,
                                                     BATCHED,
                                                     HERM,
                                                     T>(ROCBLAS_INTERNAL_SYRKX_HERKX_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syrkx_herkx_template<rocblas_int,
                                                     ROCBLAS_HERKX_NB,
                                                     BATCHED,
                                                     HERM,
                                                     T>(ROCBLAS_INTERNAL_SYRKX_HERKX_PARAMS);

    return rocblas_status_not_implemented;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herkx_batched_template(rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation trans,
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
                                            const real_t<T>*  beta,
                                            T* const*         C,
                                            rocblas_stride    offset_c,
                                            rocblas_int       ldc,
                                            rocblas_stride    stride_c,
                                            rocblas_int       batch_count)
{
    constexpr bool BATCHED = true;
    constexpr bool HERM    = true;
    if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syrkx_herkx_template<rocblas_int,
                                                     ROCBLAS_HERKX_BATCHED_NB,
                                                     BATCHED,
                                                     HERM,
                                                     T>(ROCBLAS_INTERNAL_SYRKX_HERKX_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syrkx_herkx_template<rocblas_int,
                                                     ROCBLAS_HERKX_BATCHED_NB,
                                                     BATCHED,
                                                     HERM,
                                                     T>(ROCBLAS_INTERNAL_SYRKX_HERKX_PARAMS);

    return rocblas_status_not_implemented;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files syrkx*.cpp

#ifdef INSTANTIATE_SYRKX_TEMPLATE
#error INSTANTIATE_SYRKX_TEMPLATE already defined
#endif

#define INSTANTIATE_SYRKX_TEMPLATE(T_)                                                            \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_syrkx_template<T_>( \
        rocblas_handle    handle,                                                                 \
        rocblas_fill      uplo,                                                                   \
        rocblas_operation trans,                                                                  \
        rocblas_int       n,                                                                      \
        rocblas_int       k,                                                                      \
        const T_*         alpha,                                                                  \
        const T_*         A,                                                                      \
        rocblas_stride    offset_a,                                                               \
        rocblas_int       lda,                                                                    \
        rocblas_stride    stride_a,                                                               \
        const T_*         B,                                                                      \
        rocblas_stride    offset_b,                                                               \
        rocblas_int       ldb,                                                                    \
        rocblas_stride    stride_b,                                                               \
        const T_*         beta,                                                                   \
        T_*               C,                                                                      \
        rocblas_stride    offset_c,                                                               \
        rocblas_int       ldc,                                                                    \
        rocblas_stride    stride_c,                                                               \
        rocblas_int       batch_count);

INSTANTIATE_SYRKX_TEMPLATE(float)
INSTANTIATE_SYRKX_TEMPLATE(double)
INSTANTIATE_SYRKX_TEMPLATE(rocblas_float_complex)
INSTANTIATE_SYRKX_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_SYRKX_TEMPLATE

#ifdef INSTANTIATE_SYRKX_BATCHED_TEMPLATE
#error INSTANTIATE_SYRKX_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_SYRKX_BATCHED_TEMPLATE(T_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                    \
        rocblas_internal_syrkx_batched_template<T_>(rocblas_handle    handle,   \
                                                    rocblas_fill      uplo,     \
                                                    rocblas_operation trans,    \
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

INSTANTIATE_SYRKX_BATCHED_TEMPLATE(float)
INSTANTIATE_SYRKX_BATCHED_TEMPLATE(double)
INSTANTIATE_SYRKX_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_SYRKX_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_SYRKX_BATCHED_TEMPLATE

#ifdef INSTANTIATE_HERKX_TEMPLATE
#error INSTANTIATE_HERKX_TEMPLATE already defined
#endif

#define INSTANTIATE_HERKX_TEMPLATE(T_)                                                            \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_herkx_template<T_>( \
        rocblas_handle    handle,                                                                 \
        rocblas_fill      uplo,                                                                   \
        rocblas_operation trans,                                                                  \
        rocblas_int       n,                                                                      \
        rocblas_int       k,                                                                      \
        const T_*         alpha,                                                                  \
        const T_*         A,                                                                      \
        rocblas_stride    offset_a,                                                               \
        rocblas_int       lda,                                                                    \
        rocblas_stride    stride_a,                                                               \
        const T_*         B,                                                                      \
        rocblas_stride    offset_b,                                                               \
        rocblas_int       ldb,                                                                    \
        rocblas_stride    stride_b,                                                               \
        const real_t<T_>* beta,                                                                   \
        T_*               C,                                                                      \
        rocblas_stride    offset_c,                                                               \
        rocblas_int       ldc,                                                                    \
        rocblas_stride    stride_c,                                                               \
        rocblas_int       batch_count);

INSTANTIATE_HERKX_TEMPLATE(rocblas_float_complex)
INSTANTIATE_HERKX_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_HERKX_TEMPLATE

#ifdef INSTANTIATE_HERKX_BATCHED_TEMPLATE
#error INSTANTIATE_HERKX_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_HERKX_BATCHED_TEMPLATE(T_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                    \
        rocblas_internal_herkx_batched_template<T_>(rocblas_handle    handle,   \
                                                    rocblas_fill      uplo,     \
                                                    rocblas_operation trans,    \
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
                                                    const real_t<T_>* beta,     \
                                                    T_* const*        C,        \
                                                    rocblas_stride    offset_c, \
                                                    rocblas_int       ldc,      \
                                                    rocblas_stride    stride_c, \
                                                    rocblas_int       batch_count);

INSTANTIATE_HERKX_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_HERKX_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_HERKX_BATCHED_TEMPLATE
