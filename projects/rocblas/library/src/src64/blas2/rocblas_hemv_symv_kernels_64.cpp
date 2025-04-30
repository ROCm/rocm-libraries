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

#include "handle.hpp"
#include "int64_helpers.hpp"

// #include "rocblas_hemv_symv_64.hpp"

#include "blas2/rocblas_hemv_symv_kernels.hpp" // int32 API called

template <bool IS_HEMV, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_symv_hemv_launcher_64(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      int64_t        n_64,
                                                      TScal          alpha,
                                                      rocblas_stride stride_alpha,
                                                      TConstPtr      A,
                                                      rocblas_stride offsetA,
                                                      int64_t        lda,
                                                      rocblas_stride strideA,
                                                      TConstPtr      x,
                                                      rocblas_stride offsetx,
                                                      int64_t        incx,
                                                      rocblas_stride stridex,
                                                      TScal          beta,
                                                      rocblas_stride stride_beta,
                                                      TPtr           y,
                                                      rocblas_stride offsety,
                                                      int64_t        incy,
                                                      rocblas_stride stridey,
                                                      int64_t        batch_count_64,
                                                      T*             workspace)
{
    // Quick return if possible. Not Argument error
    if(!n_64 || !batch_count_64)
        return rocblas_status_success;

    if(n_64 > c_i32_max)
        return rocblas_status_invalid_size; // defer adding new kernels for sizes exceeding practical memory

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto    x_ptr       = adjust_ptr_batch(x, b_base, stridex);
        auto    y_ptr       = adjust_ptr_batch(y, b_base, stridey);
        auto    A_ptr       = adjust_ptr_batch(A, b_base, strideA);
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        rocblas_status status = rocblas_internal_symv_hemv_launcher<IS_HEMV>(handle,
                                                                             uplo,
                                                                             (rocblas_int)n_64,
                                                                             alpha,
                                                                             stride_alpha,
                                                                             A_ptr,
                                                                             offsetA,
                                                                             lda,
                                                                             strideA,
                                                                             x_ptr,
                                                                             offsetx,
                                                                             incx,
                                                                             stridex,
                                                                             beta,
                                                                             stride_beta,
                                                                             y_ptr,
                                                                             offsety,
                                                                             incy,
                                                                             stridey,
                                                                             batch_count,
                                                                             workspace);

        if(status != rocblas_status_success)
            return status;
    } // batch
    return rocblas_status_success;
}

template <typename T>
rocblas_status rocblas_internal_symv_template_64(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n_64,
                                                 const T*       alpha,
                                                 rocblas_stride stride_alpha,
                                                 const T*       A,
                                                 rocblas_stride offsetA,
                                                 int64_t        lda,
                                                 rocblas_stride strideA,
                                                 const T*       x,
                                                 rocblas_stride offsetx,
                                                 int64_t        incx,
                                                 rocblas_stride stridex,
                                                 const T*       beta,
                                                 rocblas_stride stride_beta,
                                                 T*             y,
                                                 rocblas_stride offsety,
                                                 int64_t        incy,
                                                 rocblas_stride stridey,
                                                 int64_t        batch_count_64,
                                                 T*             workspace)
{
    constexpr bool IS_HEMV = false;
    return rocblas_internal_symv_hemv_launcher_64<IS_HEMV>(handle,
                                                           uplo,
                                                           n_64,
                                                           alpha,
                                                           stride_alpha,
                                                           A,
                                                           offsetA,
                                                           lda,
                                                           strideA,
                                                           x,
                                                           offsetx,
                                                           incx,
                                                           stridex,
                                                           beta,
                                                           stride_beta,
                                                           y,
                                                           offsety,
                                                           incy,
                                                           stridey,
                                                           batch_count_64,
                                                           workspace);
}

template <typename T>
rocblas_status rocblas_internal_hemv_template_64(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n_64,
                                                 const T*       alpha,
                                                 rocblas_stride stride_alpha,
                                                 const T*       A,
                                                 rocblas_stride offsetA,
                                                 int64_t        lda,
                                                 rocblas_stride strideA,
                                                 const T*       x,
                                                 rocblas_stride offsetx,
                                                 int64_t        incx,
                                                 rocblas_stride stridex,
                                                 const T*       beta,
                                                 rocblas_stride stride_beta,
                                                 T*             y,
                                                 rocblas_stride offsety,
                                                 int64_t        incy,
                                                 rocblas_stride stridey,
                                                 int64_t        batch_count_64,
                                                 T*             workspace)
{
    constexpr bool IS_HEMV = true;
    return rocblas_internal_symv_hemv_launcher_64<IS_HEMV>(handle,
                                                           uplo,
                                                           n_64,
                                                           alpha,
                                                           stride_alpha,
                                                           A,
                                                           offsetA,
                                                           lda,
                                                           strideA,
                                                           x,
                                                           offsetx,
                                                           incx,
                                                           stridex,
                                                           beta,
                                                           stride_beta,
                                                           y,
                                                           offsety,
                                                           incy,
                                                           stridey,
                                                           batch_count_64,
                                                           workspace);
}

template <typename T>
rocblas_status rocblas_internal_symv_batched_template_64(rocblas_handle  handle,
                                                         rocblas_fill    uplo,
                                                         int64_t         n_64,
                                                         const T*        alpha,
                                                         rocblas_stride  stride_alpha,
                                                         const T* const* A,
                                                         rocblas_stride  offsetA,
                                                         int64_t         lda,
                                                         rocblas_stride  strideA,
                                                         const T* const* x,
                                                         rocblas_stride  offsetx,
                                                         int64_t         incx,
                                                         rocblas_stride  stridex,
                                                         const T*        beta,
                                                         rocblas_stride  stride_beta,
                                                         T* const*       y,
                                                         rocblas_stride  offsety,
                                                         int64_t         incy,
                                                         rocblas_stride  stridey,
                                                         int64_t         batch_count_64,
                                                         T*              workspace)
{
    constexpr bool IS_HEMV = false;
    return rocblas_internal_symv_hemv_launcher_64<IS_HEMV>(handle,
                                                           uplo,
                                                           n_64,
                                                           alpha,
                                                           stride_alpha,
                                                           A,
                                                           offsetA,
                                                           lda,
                                                           strideA,
                                                           x,
                                                           offsetx,
                                                           incx,
                                                           stridex,
                                                           beta,
                                                           stride_beta,
                                                           y,
                                                           offsety,
                                                           incy,
                                                           stridey,
                                                           batch_count_64,
                                                           workspace);
}

template <typename T>
rocblas_status rocblas_internal_hemv_batched_template_64(rocblas_handle  handle,
                                                         rocblas_fill    uplo,
                                                         int64_t         n_64,
                                                         const T*        alpha,
                                                         rocblas_stride  stride_alpha,
                                                         const T* const* A,
                                                         rocblas_stride  offsetA,
                                                         int64_t         lda,
                                                         rocblas_stride  strideA,
                                                         const T* const* x,
                                                         rocblas_stride  offsetx,
                                                         int64_t         incx,
                                                         rocblas_stride  stridex,
                                                         const T*        beta,
                                                         rocblas_stride  stride_beta,
                                                         T* const*       y,
                                                         rocblas_stride  offsety,
                                                         int64_t         incy,
                                                         rocblas_stride  stridey,
                                                         int64_t         batch_count_64,
                                                         T*              workspace)
{
    constexpr bool IS_HEMV = true;
    return rocblas_internal_symv_hemv_launcher_64<IS_HEMV>(handle,
                                                           uplo,
                                                           n_64,
                                                           alpha,
                                                           stride_alpha,
                                                           A,
                                                           offsetA,
                                                           lda,
                                                           strideA,
                                                           x,
                                                           offsetx,
                                                           incx,
                                                           stridex,
                                                           beta,
                                                           stride_beta,
                                                           y,
                                                           offsety,
                                                           incy,
                                                           stridey,
                                                           batch_count_64,
                                                           workspace);
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the .cpp files

#ifdef INSTANTIATE_HEMV_TEMPLATE_64
#error INSTANTIATE_HEMV_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_HEMV_TEMPLATE_64(T_)                                                       \
    template rocblas_status rocblas_internal_hemv_template_64<T_>(rocblas_handle handle,       \
                                                                  rocblas_fill   uplo,         \
                                                                  int64_t        n,            \
                                                                  const T_*      alpha,        \
                                                                  rocblas_stride stride_alpha, \
                                                                  const T_*      A,            \
                                                                  rocblas_stride offsetA,      \
                                                                  int64_t        lda,          \
                                                                  rocblas_stride strideA,      \
                                                                  const T_*      x,            \
                                                                  rocblas_stride offsetx,      \
                                                                  int64_t        incx,         \
                                                                  rocblas_stride stridex,      \
                                                                  const T_*      beta,         \
                                                                  rocblas_stride stride_beta,  \
                                                                  T_*            y,            \
                                                                  rocblas_stride offsety,      \
                                                                  int64_t        incy,         \
                                                                  rocblas_stride stridey,      \
                                                                  int64_t        batch_count,  \
                                                                  T_*            workspace);

INSTANTIATE_HEMV_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_HEMV_TEMPLATE_64(rocblas_double_complex)

#undef INSTANTIATE_HEMV_TEMPLATE_64

#ifdef INSTANTIATE_HEMV_BATCHED_TEMPLATE_64
#error INSTANTIATE_HEMV_BATCHED_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_HEMV_BATCHED_TEMPLATE_64(T_)                           \
    template rocblas_status rocblas_internal_hemv_batched_template_64<T_>( \
        rocblas_handle   handle,                                           \
        rocblas_fill     uplo,                                             \
        int64_t          n,                                                \
        const T_*        alpha,                                            \
        rocblas_stride   stride_alpha,                                     \
        const T_* const* A,                                                \
        rocblas_stride   offsetA,                                          \
        int64_t          lda,                                              \
        rocblas_stride   strideA,                                          \
        const T_* const* x,                                                \
        rocblas_stride   offsetx,                                          \
        int64_t          incx,                                             \
        rocblas_stride   stridex,                                          \
        const T_*        beta,                                             \
        rocblas_stride   stride_beta,                                      \
        T_* const*       y,                                                \
        rocblas_stride   offsety,                                          \
        int64_t          incy,                                             \
        rocblas_stride   stridey,                                          \
        int64_t          batch_count,                                      \
        T_*              workspace);

INSTANTIATE_HEMV_BATCHED_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_HEMV_BATCHED_TEMPLATE_64(rocblas_double_complex)

#undef INSTANTIATE_HEMV_BATCHED_TEMPLATE_64

#ifdef INSTANTIATE_SYMV_TEMPLATE_64
#error INSTANTIATE_SYMV_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_SYMV_TEMPLATE_64(T_)                                                       \
    template rocblas_status rocblas_internal_symv_template_64<T_>(rocblas_handle handle,       \
                                                                  rocblas_fill   uplo,         \
                                                                  int64_t        n,            \
                                                                  const T_*      alpha,        \
                                                                  rocblas_stride stride_alpha, \
                                                                  const T_*      A,            \
                                                                  rocblas_stride offsetA,      \
                                                                  int64_t        lda,          \
                                                                  rocblas_stride strideA,      \
                                                                  const T_*      x,            \
                                                                  rocblas_stride offsetx,      \
                                                                  int64_t        incx,         \
                                                                  rocblas_stride stridex,      \
                                                                  const T_*      beta,         \
                                                                  rocblas_stride stride_beta,  \
                                                                  T_*            y,            \
                                                                  rocblas_stride offsety,      \
                                                                  int64_t        incy,         \
                                                                  rocblas_stride stridey,      \
                                                                  int64_t        batch_count,  \
                                                                  T_*            workspace);

INSTANTIATE_SYMV_TEMPLATE_64(float)
INSTANTIATE_SYMV_TEMPLATE_64(double)
INSTANTIATE_SYMV_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_SYMV_TEMPLATE_64(rocblas_double_complex)

#undef INSTANTIATE_SYMV_TEMPLATE_64

#ifdef INSTANTIATE_SYMV_BATCHED_TEMPLATE_64
#error INSTANTIATE_SYMV_BATCHED_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_SYMV_BATCHED_TEMPLATE_64(T_)                           \
    template rocblas_status rocblas_internal_symv_batched_template_64<T_>( \
        rocblas_handle   handle,                                           \
        rocblas_fill     uplo,                                             \
        int64_t          n,                                                \
        const T_*        alpha,                                            \
        rocblas_stride   stride_alpha,                                     \
        const T_* const* A,                                                \
        rocblas_stride   offsetA,                                          \
        int64_t          lda,                                              \
        rocblas_stride   strideA,                                          \
        const T_* const* x,                                                \
        rocblas_stride   offsetx,                                          \
        int64_t          incx,                                             \
        rocblas_stride   stridex,                                          \
        const T_*        beta,                                             \
        rocblas_stride   stride_beta,                                      \
        T_* const*       y,                                                \
        rocblas_stride   offsety,                                          \
        int64_t          incy,                                             \
        rocblas_stride   stridey,                                          \
        int64_t          batch_count,                                      \
        T_*              workspace);

INSTANTIATE_SYMV_BATCHED_TEMPLATE_64(float)
INSTANTIATE_SYMV_BATCHED_TEMPLATE_64(double)
INSTANTIATE_SYMV_BATCHED_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_SYMV_BATCHED_TEMPLATE_64(rocblas_double_complex)

#undef INSTANTIATE_SYMV_BATCHED_TEMPLATE_64
