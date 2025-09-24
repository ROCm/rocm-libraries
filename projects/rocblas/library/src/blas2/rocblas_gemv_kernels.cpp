/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "check_numerics_vector.hpp"
#include "gemv_device.hpp"
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas_gemv.hpp"
#include "rocblas_level2_threshold.hpp"

// The warpSize * 2 corresponds to the number of x-dimension threads per block optimized for better performance in the double_buffered_kernels.
constexpr int rocblas_gemv_bx()
{
    return 64 * 2; // warpSize for gfx9xx: 64
}

// gemvt_sn is skinny n matrix optimizations
constexpr int rocblas_gemvt_sn_WIN()
{
    return 4;
}

constexpr int rocblas_gemvt_sn_NB()
{
    return 256;
}
inline size_t rocblas_gemvt_sn_kernel_block_count(rocblas_int m)
{
    if(m <= 0)
        m = 1; // avoid sign loss issues
    return size_t(m - 1) / (rocblas_gemvt_sn_NB() * rocblas_gemvt_sn_WIN()) + 1;
}

// gemvt_sn_crossover is n threshold to crossover back to normal (non-skinny) algorithm
template <typename T>
inline size_t rocblas_gemvt_sn_crossover()
{
    return 256;
}
template <>
inline size_t rocblas_gemvt_sn_crossover<rocblas_double>()
{
    return 128;
}
template <>
inline size_t rocblas_gemvt_sn_crossover<rocblas_float_complex>()
{
    return 64;
}
template <>
inline size_t rocblas_gemvt_sn_crossover<rocblas_double_complex>()
{
    return 16;
}

template <typename T>
inline bool rocblas_gemvt_skinny_n(rocblas_operation transA, rocblas_int m, rocblas_int n)
{
    size_t       cross_over_n    = rocblas_gemvt_sn_crossover<T>();
    const size_t skinny_constant = 2048;
    if(transA != rocblas_operation_none && n < cross_over_n && m >= skinny_constant * n)
        return true;
    else
        return false;
}

template <typename T>
inline bool rocblas_gemvt_fat_n(rocblas_int m, rocblas_int n, int gfx_arch)
{
    if(gfx_arch == 910)
    {
        bool fat = n / 4 >= m;
        return fat
               && ((std::is_same_v<T, float> && m <= 768)
                   || ((std::is_same_v<T, double> || std::is_same_v<T, rocblas_float_complex>)&&m
                       <= 384)
                   || (std::is_same_v<T, rocblas_double_complex> && m <= 128));
    }
    else if(gfx_arch == 942)
    {
        bool fat = n / 8 >= m;
        return fat && (std::is_same_v<T, float> && m <= 512);
    }
    else
        return false;
}

/*! \brief rocblas_internal_gemv_kernel_workspace_size
    Currently only transpose/conj skinny n matrices use workspace memory, so usually returns 0
    Work buffer for column reductions: number of blocks * cols * batch_count

    @param[in]
    outputType To*
        Type of output values
    @param[in]
    m rocblas_int
        Number of rows
    @param[in]
    n rocblas_int
        Number of columns
    @param[in]
    batch_count rocblas_int
        Number of batches
    ********************************************************************/
template <typename To>
ROCBLAS_INTERNAL_EXPORT_NOINLINE size_t rocblas_internal_gemv_kernel_workspace_size(
    rocblas_operation transA, rocblas_int m, rocblas_int n, rocblas_int batch_count)
{
    if(m <= 0 || n <= 0 || batch_count <= 0)
        return 0;

    if(!rocblas_gemvt_skinny_n<To>(transA, m, n))
        return 0; // workspace only used for skinny n kernel transpose/conj. transpose

    auto blocks = rocblas_gemvt_sn_kernel_block_count(m);
    return sizeof(To) * blocks * n * batch_count;
}

template <typename Ti, typename Tex, typename To>
rocblas_status rocblas_internal_gemv_launcher(rocblas_handle    handle,
                                              rocblas_operation transA,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const Tex*        alpha,
                                              rocblas_stride    stride_alpha,
                                              const Ti*         A,
                                              rocblas_stride    offseta,
                                              int64_t           lda,
                                              rocblas_stride    strideA,
                                              const Ti*         x,
                                              rocblas_stride    offsetx,
                                              int64_t           incx,
                                              rocblas_stride    stridex,
                                              const Tex*        beta,
                                              rocblas_stride    stride_beta,
                                              To*               y,
                                              rocblas_stride    offsety,
                                              int64_t           incy,
                                              rocblas_stride    stridey,
                                              rocblas_int       batch_count,
                                              Tex*              workspace)
{
    //quick return
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0
                      ? offsetx - int64_t(incx) * (transA == rocblas_operation_none ? n - 1 : m - 1)
                      : offsetx;
    auto shifty = incy < 0
                      ? offsety - int64_t(incy) * (transA == rocblas_operation_none ? m - 1 : n - 1)
                      : offsety;

    bool i64_incs = lda > c_i32_max || incx > c_i32_max || incx < c_i32_min || incy > c_i32_max
                    || incy < c_i32_min;

    bool i64_indices // i64_incs implies i64_indices
        = i64_incs || size_t(n) * lda > c_i32_max
          || size_t(transA == rocblas_operation_none ? n - 1 : m - 1) * std::abs(incx) >= c_i32_max
          || size_t(transA == rocblas_operation_none ? m - 1 : n - 1) * std::abs(incy) >= c_i32_max;

    //Identifying the precision to have an appropriate optimization
    static constexpr bool is_float = std::is_same_v<Ti, float> || std::is_same_v<Ti, float const*>;
    static constexpr bool is_double
        = std::is_same_v<Ti, double> || std::is_same_v<Ti, double const*>;
    static constexpr bool is_complex_float
        = std::is_same_v<Ti,
                         rocblas_float_complex> || std::is_same_v<Ti, rocblas_float_complex const*>;
    static constexpr bool is_complex_double
        = std::is_same_v<
              Ti,
              rocblas_double_complex> || std::is_same_v<Ti, rocblas_double_complex const*>;
    const bool is_atomics_allowed = handle->atomics_mode == rocblas_atomics_allowed ? true : false;

    //Identifying the architecture to have an appropriate optimization
    int  arch_major = handle->getArchMajor();
    bool is_arch_10_or_11_or_12
        = arch_major == 10 || arch_major == 11 || arch_major == 12 ? true : false;

    bool is_gfx11xx = arch_major == 11 ? true : false;
    bool is_gfx12xx = arch_major == 12 ? true : false;
    int  gfx_arch   = handle->getArch();
    bool is_gfx908  = gfx_arch == 908 ? true : false;
    bool is_gfx906  = gfx_arch == 906 ? true : false;
    bool is_gfx90a  = gfx_arch == 910 ? true : false;
    bool is_gfx942  = gfx_arch == 942 ? true : false;

    int batches = handle->getBatchGridDim((int)batch_count);

    if(transA == rocblas_operation_none)
    {
#define gemvn_KARGS(alpha_, beta_)                                                             \
    gemvn_grid, gemvn_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, A, offseta, lda, \
        strideA, x, shiftx, incx, stridex, beta_, stride_beta, y, shifty, incy, stridey,       \
        batch_count

        if(!i64_incs && m <= 32 && n <= 32
           && ((is_gfx90a && batch_count >= gemv_sm_mn_gfx90a_batch_min_threshold)
               || (is_gfx942 && batch_count >= gemv_sm_mn_gfx942_batch_min_threshold)))
        {
#define gemvn_sm_mn_batched_KARGS(alpha_, beta_)                                                 \
    gemvn_sm_mn_batched_grid, gemvn_sm_mn_batched_threads, 0, rocblas_stream, m, n, alpha_,      \
        stride_alpha, A, offseta, lda, strideA, x, shiftx, incx, stridex, beta_, stride_beta, y, \
        shifty, incy, stridey, batch_count

            // all rows && cols covered by DIM_X threads
            static constexpr int GEMVN_SM_MN_BATCHED_DIM_X      = 32;
            static constexpr int GEMVN_SM_MN_BATCHED_DIM_NBATCH = 24;

            dim3 gemvn_sm_mn_batched_grid((batch_count - 1) / GEMVN_SM_MN_BATCHED_DIM_NBATCH + 1);
            dim3 gemvn_sm_mn_batched_threads(GEMVN_SM_MN_BATCHED_DIM_X,
                                             GEMVN_SM_MN_BATCHED_DIM_NBATCH);

            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                ROCBLAS_LAUNCH_KERNEL(
                    (rocblas_gemvn_sm_mn_batched_kernel<GEMVN_SM_MN_BATCHED_DIM_X,
                                                        GEMVN_SM_MN_BATCHED_DIM_NBATCH>),
                    gemvn_sm_mn_batched_KARGS(alpha, beta));
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;

                ROCBLAS_LAUNCH_KERNEL(
                    (rocblas_gemvn_sm_mn_batched_kernel<GEMVN_SM_MN_BATCHED_DIM_X,
                                                        GEMVN_SM_MN_BATCHED_DIM_NBATCH>),
                    gemvn_sm_mn_batched_KARGS(*alpha, *beta));
            }
#undef gemvn_sm_mn_batched_KARGS
        }
        else if(n <= 128 && m >= 2048 * n)
        {
            // skinny tuned block size

            static constexpr int GEMVN_DIM_X = 64;
            static constexpr int GEMVN_DIM_Y = 4;
            rocblas_int          blocks      = (m - 1) / (GEMVN_DIM_X * 4) + 1;
            if(std::is_same_v<Tex, rocblas_double_complex>)
                blocks = (m - 1) / (GEMVN_DIM_X) + 1;
            dim3 gemvn_grid(blocks, 1, batches);
            dim3 gemvn_threads(GEMVN_DIM_X, GEMVN_DIM_Y);

            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, rocblas_int>),
                        gemvn_KARGS(alpha, beta));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, int64_t>),
                                          gemvn_KARGS(alpha, beta));
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;

                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, rocblas_int>),
                        gemvn_KARGS(*alpha, *beta));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, int64_t>),
                                          gemvn_KARGS(*alpha, *beta));
            }
        }
        //optimized gemvn kernel with double buffered loads for gfx90a or gfx942
        else if(!i64_incs && is_atomics_allowed && (is_float || is_double) && (m == n)
                && (m % rocblas_gemv_bx() == 0)
                && (is_gfx90a
                    || (is_gfx942
                        && ((is_float && m < sgemvn_gfx942_double_buffered_higher_threshold)
                            || (is_double && n < dgemvn_gfx942_double_buffered_higher_threshold)))))
        {
            if constexpr(is_float || is_double)
            {
                // The following rocblas_gemv_scal_kernel does the `y = y*beta` computation
                static constexpr int NB               = 256;
                const int            gemv_scal_blocks = (m - 1) / NB + 1;
                dim3                 grid(gemv_scal_blocks, 1, batches);
                dim3                 threads(NB);
                if(handle->pointer_mode == rocblas_pointer_mode_device)
                {
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemv_scal_kernel<NB>),
                                          grid,
                                          threads,
                                          0,
                                          rocblas_stream,
                                          m,
                                          beta,
                                          stride_beta,
                                          y,
                                          shifty,
                                          incy,
                                          stridey,
                                          batch_count);
                }
                else
                {
                    if(*beta != 1)
                        ROCBLAS_LAUNCH_KERNEL((rocblas_gemv_scal_kernel<NB>),
                                              grid,
                                              threads,
                                              0,
                                              rocblas_stream,
                                              m,
                                              *beta,
                                              stride_beta,
                                              y,
                                              shifty,
                                              incy,
                                              stridey,
                                              batch_count);
                }

#define gemvn_double_buffered_KARGS(alpha_)                                                    \
    gemvn_grid, gemvn_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, A, offseta, lda, \
        strideA, x, shiftx, incx, stridex, y, shifty, incy, stridey, batch_count

                // The following kernel does the `y += A * x` computation
                static constexpr int thread_x            = rocblas_gemv_bx();
                static constexpr int block_y             = 8;
                static constexpr int thread_y            = is_float ? 8 : 4;
                static constexpr int elements_per_thread = thread_x / (2 * thread_y);

                const int block_x = m / thread_x;
                dim3      gemvn_threads(thread_x, thread_y);
                dim3      gemvn_grid(block_x, block_y, batches);

                if(handle->pointer_mode == rocblas_pointer_mode_device)
                {
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvn_double_buffered_kernel<thread_x,
                                                              thread_y,
                                                              elements_per_thread>),
                        gemvn_double_buffered_KARGS(alpha));
                }
                else
                {
                    if(!*alpha)
                        return rocblas_status_success;

                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvn_double_buffered_kernel<thread_x,
                                                              thread_y,
                                                              elements_per_thread>),
                        gemvn_double_buffered_KARGS(*alpha));
                }
            }
#undef gemvn_double_buffered_KARGS
        }
        //optimized gemvn kernel with 512 threads/block for gfx906, gfx908, gfx90a, gfx942, gfx11xx and gfx12xx.
        // When(m < 2*n) should use 512 threads/block for gfx90a, gfx942, gfx11xx and gfx12xx.
        else if(((is_gfx11xx || is_gfx12xx || is_gfx90a || is_gfx942) && (m < 2 * n))
                || (is_gfx908
                    && (((is_float || is_double || is_complex_float) && m <= gemvn_gfx908_threshold
                         && n <= gemvn_gfx908_threshold)
                        || (is_complex_double && m <= zgemvn_gfx908_threshold
                            && n <= zgemvn_gfx908_threshold)))

                || (is_gfx906
                    && (is_complex_float
                        || ((is_float || is_double) && m <= gemvn_gfx906_threshold
                            && n <= gemvn_gfx906_threshold)
                        || (is_double
                            && ((m >= dgemvn_gfx906_lower_threshold
                                 && n >= dgemvn_gfx906_lower_threshold)
                                || (m <= dgemvn_gfx906_upper_threshold
                                    && n <= dgemvn_gfx906_upper_threshold))))))
        {
            static constexpr int GEMVN_DIM_X = 32;
            static constexpr int GEMVN_DIM_Y = 16;
            rocblas_int          blocks      = (m - 1) / (GEMVN_DIM_X * 4) + 1;
            if(std::is_same_v<Tex, rocblas_double_complex>)
                blocks = (m - 1) / (GEMVN_DIM_X) + 1;
            dim3 gemvn_grid(blocks, 1, batches);
            dim3 gemvn_threads(GEMVN_DIM_X, GEMVN_DIM_Y);

            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, rocblas_int>),
                        gemvn_KARGS(alpha, beta));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, int64_t>),
                                          gemvn_KARGS(alpha, beta));
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;

                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, rocblas_int>),
                        gemvn_KARGS(*alpha, *beta));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, int64_t>),
                                          gemvn_KARGS(*alpha, *beta));
            }
        }
        else
        {
            // GEMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
            static constexpr int GEMVN_DIM_X = 64;
            static constexpr int GEMVN_DIM_Y = 16;
            rocblas_int          blocks      = (m - 1) / (GEMVN_DIM_X * 4) + 1;
            if(std::is_same_v<Tex, rocblas_double_complex>)
                blocks = (m - 1) / (GEMVN_DIM_X) + 1;
            dim3 gemvn_grid(blocks, 1, batches);
            dim3 gemvn_threads(GEMVN_DIM_X, GEMVN_DIM_Y);

            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, rocblas_int>),
                        gemvn_KARGS(alpha, beta));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, int64_t>),
                                          gemvn_KARGS(alpha, beta));
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;

                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, rocblas_int>),
                        gemvn_KARGS(*alpha, *beta));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, int64_t>),
                                          gemvn_KARGS(*alpha, *beta));
            }
        }
#undef gemvn_KARGS
    }
    else if(transA == rocblas_operation_transpose)
    {
        // transpose
        static constexpr bool CONJ = false;

        if(!i64_incs && m <= 64 && batch_count > 8) // few rows, e.g. qmcpack
        {
            // number of columns on the y-dim of the grid
            static constexpr int NB = 256;
            dim3                 gemvtsm_grid(batch_count);
            dim3                 gemvtsm_threads(NB);
            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemvtsm_kernel<CONJ, NB>),
                                      gemvtsm_grid,
                                      gemvtsm_threads,
                                      0,
                                      rocblas_stream,
                                      m,
                                      n,
                                      alpha,
                                      stride_alpha,
                                      A,
                                      offseta,
                                      lda,
                                      strideA,
                                      x,
                                      shiftx,
                                      incx,
                                      stridex,
                                      beta,
                                      stride_beta,
                                      y,
                                      shifty,
                                      incy,
                                      stridey);
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;

                ROCBLAS_LAUNCH_KERNEL((rocblas_gemvtsm_kernel<CONJ, NB>),
                                      gemvtsm_grid,
                                      gemvtsm_threads,
                                      0,
                                      rocblas_stream,
                                      m,
                                      n,
                                      *alpha,
                                      stride_alpha,
                                      A,
                                      offseta,
                                      lda,
                                      strideA,
                                      x,
                                      shiftx,
                                      incx,
                                      stridex,
                                      *beta,
                                      stride_beta,
                                      y,
                                      shifty,
                                      incy,
                                      stridey);
            }
        }
        else if(!i64_incs && rocblas_gemvt_fat_n<Ti>(m, n, gfx_arch))
        {
#define gemvt_row_vectorized_KARGS(alpha_, beta_)                                              \
    gemvt_grid, gemvt_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, A, offseta, lda, \
        strideA, x, shiftx, incx, stridex, beta_, stride_beta, y, shifty, incy, stridey,       \
        batch_count

            if constexpr(is_float)
            {
                const int TILE_DIM_X = 16;
                const int TILE_DIM_Y = 64;
                dim3      gemvt_threads(TILE_DIM_X, TILE_DIM_Y);
                dim3      gemvt_grid((n - 1) / TILE_DIM_Y + 1, 1, batches);
                if(handle->pointer_mode == rocblas_pointer_mode_device)
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_row_vectorized_kernel<CONJ, TILE_DIM_X, TILE_DIM_Y>),
                        gemvt_row_vectorized_KARGS(alpha, beta));
                else
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_row_vectorized_kernel<CONJ, TILE_DIM_X, TILE_DIM_Y>),
                        gemvt_row_vectorized_KARGS(*alpha, *beta));
            }
            else if constexpr(is_double || is_complex_float)
            {
                const int TILE_DIM_X = 16;
                const int TILE_DIM_Y = 32;
                dim3      gemvt_threads(TILE_DIM_X, TILE_DIM_Y);
                dim3      gemvt_grid((n - 1) / TILE_DIM_Y + 1, 1, batches);
                if(handle->pointer_mode == rocblas_pointer_mode_device)
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_row_vectorized_kernel<CONJ, TILE_DIM_X, TILE_DIM_Y>),
                        gemvt_row_vectorized_KARGS(alpha, beta));
                else
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_row_vectorized_kernel<CONJ, TILE_DIM_X, TILE_DIM_Y>),
                        gemvt_row_vectorized_KARGS(*alpha, *beta));
            }
            else if constexpr(is_complex_double)
            {
                const int TILE_DIM_X = 16;
                const int TILE_DIM_Y = 8;
                dim3      gemvt_threads(TILE_DIM_X, TILE_DIM_Y);
                dim3      gemvt_grid((n - 1) / TILE_DIM_Y + 1, 1, batches);
                if(handle->pointer_mode == rocblas_pointer_mode_device)
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_row_vectorized_kernel<CONJ, TILE_DIM_X, TILE_DIM_Y>),
                        gemvt_row_vectorized_KARGS(alpha, beta));
                else
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_row_vectorized_kernel<CONJ, TILE_DIM_X, TILE_DIM_Y>),
                        gemvt_row_vectorized_KARGS(*alpha, *beta));
            }
#undef gemvt_row_vectorized_KARGS
        }
        else if(workspace && rocblas_gemvt_skinny_n<Ti>(transA, m, n))
        {
            static constexpr int NB     = rocblas_gemvt_sn_NB();
            static constexpr int WIN    = rocblas_gemvt_sn_WIN();
            int                  blocks = rocblas_gemvt_sn_kernel_block_count(m);
            dim3                 gemvt_grid(blocks, 1, batches);
            dim3                 gemvt_threads(NB);

#define gemvt_sn_KARGS(alpha_)                                                                 \
    gemvt_grid, gemvt_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, A, offseta, lda, \
        strideA, x, shiftx, incx, stridex, (Tex*)workspace, batch_count

            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_sn_kernel<CONJ, NB, WIN, rocblas_int>),
                                          gemvt_sn_KARGS(alpha));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_sn_kernel<CONJ, NB, WIN, int64_t>),
                                          gemvt_sn_KARGS(alpha));

                ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_sn_reduce<NB, 8>),
                                      dim3(1, n, batches),
                                      gemvt_threads,
                                      0,
                                      rocblas_stream,
                                      blocks,
                                      beta,
                                      stride_beta,
                                      y,
                                      shifty,
                                      incy,
                                      stridey,
                                      (Tex*)workspace,
                                      batch_count);
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;

                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_sn_kernel<CONJ, NB, WIN, rocblas_int>),
                                          gemvt_sn_KARGS(*alpha));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_sn_kernel<CONJ, NB, WIN, int64_t>),
                                          gemvt_sn_KARGS(*alpha));

                ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_sn_reduce<NB, 8>),
                                      dim3(1, n, batches),
                                      gemvt_threads,
                                      0,
                                      rocblas_stream,
                                      blocks,
                                      *beta,
                                      stride_beta,
                                      y,
                                      shifty,
                                      incy,
                                      stridey,
                                      workspace,
                                      batch_count);
            }

#undef gemvt_sn_KARGS
        }
        //optimized gemvt kernel with double buffered loads for gfx908 and gfx942.
        else if(!i64_incs && is_atomics_allowed && (m == n) && (m % rocblas_gemv_bx() == 0)
                && ((is_gfx908
                     && ((is_float && m > sgemvt_gfx908_lower_threshold)
                         || (is_double && m > dgemvt_gfx908_lower_threshold)))
                    || (is_gfx942
                        && ((is_float && m > gemvt_threshold)
                            || (is_double && m > gemvt_threshold)))))
        {
            if constexpr(is_float || is_double)
            {
                // The following rocblas_gemv_scal_kernel does the `y = y*beta` computation
                static constexpr int NB               = 256;
                const int            gemv_scal_blocks = (n - 1) / NB + 1;
                dim3                 grid(gemv_scal_blocks, 1, batches);
                dim3                 threads(NB);
                if(handle->pointer_mode == rocblas_pointer_mode_device)
                {
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemv_scal_kernel<NB>),
                                          grid,
                                          threads,
                                          0,
                                          rocblas_stream,
                                          n,
                                          beta,
                                          stride_beta,
                                          y,
                                          shifty,
                                          incy,
                                          stridey,
                                          batch_count);
                }
                else
                {
                    if(*beta != 1)
                        ROCBLAS_LAUNCH_KERNEL((rocblas_gemv_scal_kernel<NB>),
                                              grid,
                                              threads,
                                              0,
                                              rocblas_stream,
                                              n,
                                              *beta,
                                              stride_beta,
                                              y,
                                              shifty,
                                              incy,
                                              stridey,
                                              batch_count);
                }
                // The following kernel does the `y += A * x` computation
                static constexpr int thread_x            = rocblas_gemv_bx();
                static constexpr int block_y             = is_float ? 8 : 16;
                static constexpr int thread_y            = is_float ? 8 : 4;
                static constexpr int elements_per_thread = thread_x / (2 * thread_y);

                const int block_x = n / thread_x;
                dim3      gemvt_threads(thread_x, thread_y);
                dim3      gemvt_grid(block_x, block_y, batches);

#define gemvt_double_buffered_KARGS(alpha_)                                                    \
    gemvt_grid, gemvt_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, A, offseta, lda, \
        strideA, x, shiftx, incx, stridex, y, shifty, incy, stridey, batch_count

                if(handle->pointer_mode == rocblas_pointer_mode_device)
                {
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_double_buffered_kernel<CONJ,
                                                              thread_x,
                                                              thread_y,
                                                              elements_per_thread>),
                        gemvt_double_buffered_KARGS(alpha));
                }
                else
                {
                    if(!*alpha)
                        return rocblas_status_success;

                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_double_buffered_kernel<CONJ,
                                                              thread_x,
                                                              thread_y,
                                                              elements_per_thread>),
                        gemvt_double_buffered_KARGS(*alpha));
                }
            }
#undef gemvt_double_buffered_KARGS
        }

#define gemvt_KARGS(alpha_, beta_)                                                             \
    gemvt_grid, gemvt_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, A, offseta, lda, \
        strideA, x, shiftx, incx, stridex, beta_, stride_beta, y, shifty, incy, stridey,       \
        batch_count

        //Using kernel code with warp reduction for gfx1030.
        else if(is_arch_10_or_11_or_12
                && (is_double || is_complex_float
                    || (is_float
                        && (m < sgemvt_gfx_arch_10_11_12_threshold
                            || n < sgemvt_gfx_arch_10_11_12_threshold))))
        {
            //Number of threads per block
            static constexpr int NB = 256;
            dim3                 gemvt_grid(n, 1, batches);
            dim3                 gemvt_threads(NB);

            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_warp_reduce_kernel<CONJ, NB, rocblas_int>),
                                          gemvt_KARGS(alpha, beta));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_warp_reduce_kernel<CONJ, NB, int64_t>),
                                          gemvt_KARGS(alpha, beta));
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;

                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_warp_reduce_kernel<CONJ, NB, rocblas_int>),
                                          gemvt_KARGS(*alpha, *beta));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_warp_reduce_kernel<CONJ, NB, int64_t>),
                                          gemvt_KARGS(*alpha, *beta));
            }
        }
        //Using kernel code with shared memory reduction for single precision as well as for other precisions when m or n is less than gemvt_threshold and for complex double in gfx1030.
        else if(!i64_incs
                && ((is_float || m < gemvt_threshold || n < gemvt_threshold)
                    || (is_arch_10_or_11_or_12 && is_complex_double)))
        {
            //Number of threads per block
            static constexpr int NB = 256;
            dim3                 gemvt_grid(n, 1, batches);
            dim3                 gemvt_threads(NB);

            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_kernel<CONJ, NB>), gemvt_KARGS(alpha, beta));
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;

                ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_kernel<CONJ, NB>), gemvt_KARGS(*alpha, *beta));
            }
        }

        //Using kernel code with warp reduction.
        //Having 1024 threads per block for double, complex-float and complex-double precision GEMV (transpose) for better performance.
        else
        {
            //Number of threads per block
            static constexpr int NB = 1024;
            dim3                 gemvt_grid(n, 1, batches);
            dim3                 gemvt_threads(NB);

            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {

                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_warp_reduce_kernel<CONJ, NB, rocblas_int>),
                                          gemvt_KARGS(alpha, beta));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_warp_reduce_kernel<CONJ, NB, int64_t>),
                                          gemvt_KARGS(alpha, beta));
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;

                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_warp_reduce_kernel<CONJ, NB, rocblas_int>),
                                          gemvt_KARGS(*alpha, *beta));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_warp_reduce_kernel<CONJ, NB, int64_t>),
                                          gemvt_KARGS(*alpha, *beta));
            }
        }
#undef gemvt_KARGS
    }
    else
    {
        // conjugate transpose
        static constexpr bool CONJ = true;

        if(!i64_incs && m <= 64 && batch_count > 8) // few rows, e.g. qmcpack
        {
            // number of columns on the y-dim of the grid
            static constexpr int NB = 256;
            dim3                 gemvtsm_grid(batch_count);
            dim3                 gemvtsm_threads(NB);
            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemvtsm_kernel<CONJ, NB>),
                                      gemvtsm_grid,
                                      gemvtsm_threads,
                                      0,
                                      rocblas_stream,
                                      m,
                                      n,
                                      alpha,
                                      stride_alpha,
                                      A,
                                      offseta,
                                      lda,
                                      strideA,
                                      x,
                                      shiftx,
                                      incx,
                                      stridex,
                                      beta,
                                      stride_beta,
                                      y,
                                      shifty,
                                      incy,
                                      stridey);
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;

                ROCBLAS_LAUNCH_KERNEL((rocblas_gemvtsm_kernel<CONJ, NB>),
                                      gemvtsm_grid,
                                      gemvtsm_threads,
                                      0,
                                      rocblas_stream,
                                      m,
                                      n,
                                      *alpha,
                                      stride_alpha,
                                      A,
                                      offseta,
                                      lda,
                                      strideA,
                                      x,
                                      shiftx,
                                      incx,
                                      stridex,
                                      *beta,
                                      stride_beta,
                                      y,
                                      shifty,
                                      incy,
                                      stridey);
            }
        }
        else if(!i64_incs && rocblas_gemvt_fat_n<Ti>(m, n, gfx_arch))
        {
#define gemvt_row_vectorized_KARGS(alpha_, beta_)                                              \
    gemvt_grid, gemvt_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, A, offseta, lda, \
        strideA, x, shiftx, incx, stridex, beta_, stride_beta, y, shifty, incy, stridey,       \
        batch_count

            if constexpr(is_float)
            {
                const int TILE_DIM_X = 16;
                const int TILE_DIM_Y = 64;
                dim3      gemvt_threads(TILE_DIM_X, TILE_DIM_Y);
                dim3      gemvt_grid((n - 1) / TILE_DIM_Y + 1, 1, batches);
                if(handle->pointer_mode == rocblas_pointer_mode_device)
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_row_vectorized_kernel<CONJ, TILE_DIM_X, TILE_DIM_Y>),
                        gemvt_row_vectorized_KARGS(alpha, beta));
                else
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_row_vectorized_kernel<CONJ, TILE_DIM_X, TILE_DIM_Y>),
                        gemvt_row_vectorized_KARGS(*alpha, *beta));
            }
            else if constexpr(is_double || is_complex_float)
            {
                const int TILE_DIM_X = 16;
                const int TILE_DIM_Y = 32;
                dim3      gemvt_threads(TILE_DIM_X, TILE_DIM_Y);
                dim3      gemvt_grid((n - 1) / TILE_DIM_Y + 1, 1, batches);
                if(handle->pointer_mode == rocblas_pointer_mode_device)
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_row_vectorized_kernel<CONJ, TILE_DIM_X, TILE_DIM_Y>),
                        gemvt_row_vectorized_KARGS(alpha, beta));
                else
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_row_vectorized_kernel<CONJ, TILE_DIM_X, TILE_DIM_Y>),
                        gemvt_row_vectorized_KARGS(*alpha, *beta));
            }
            else if constexpr(is_complex_double)
            {
                const int TILE_DIM_X = 16;
                const int TILE_DIM_Y = 8;
                dim3      gemvt_threads(TILE_DIM_X, TILE_DIM_Y);
                dim3      gemvt_grid((n - 1) / TILE_DIM_Y + 1, 1, batches);
                if(handle->pointer_mode == rocblas_pointer_mode_device)
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_row_vectorized_kernel<CONJ, TILE_DIM_X, TILE_DIM_Y>),
                        gemvt_row_vectorized_KARGS(alpha, beta));
                else
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_row_vectorized_kernel<CONJ, TILE_DIM_X, TILE_DIM_Y>),
                        gemvt_row_vectorized_KARGS(*alpha, *beta));
            }
#undef gemvt_row_vectorized_KARGS
        }
        else if(workspace && rocblas_gemvt_skinny_n<Ti>(transA, m, n))
        {
            static constexpr int NB     = rocblas_gemvt_sn_NB();
            static constexpr int WIN    = rocblas_gemvt_sn_WIN();
            int                  blocks = rocblas_gemvt_sn_kernel_block_count(m);
            dim3                 gemvt_grid(blocks, 1, batches);
            dim3                 gemvt_threads(NB);

#define gemvt_sn_KARGS(alpha_)                                                                 \
    gemvt_grid, gemvt_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, A, offseta, lda, \
        strideA, x, shiftx, incx, stridex, (Tex*)workspace, batch_count

            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_sn_kernel<CONJ, NB, WIN, rocblas_int>),
                                          gemvt_sn_KARGS(alpha));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_sn_kernel<CONJ, NB, WIN, int64_t>),
                                          gemvt_sn_KARGS(alpha));

                ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_sn_reduce<NB, 8>),
                                      dim3(1, n, batches),
                                      gemvt_threads,
                                      0,
                                      rocblas_stream,
                                      blocks,
                                      beta,
                                      stride_beta,
                                      y,
                                      shifty,
                                      incy,
                                      stridey,
                                      (Tex*)workspace,
                                      batch_count);
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;

                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_sn_kernel<CONJ, NB, WIN, rocblas_int>),
                                          gemvt_sn_KARGS(*alpha));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_sn_kernel<CONJ, NB, WIN, int64_t>),
                                          gemvt_sn_KARGS(*alpha));

                ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_sn_reduce<NB, 8>),
                                      dim3(1, n, batches),
                                      gemvt_threads,
                                      0,
                                      rocblas_stream,
                                      blocks,
                                      *beta,
                                      stride_beta,
                                      y,
                                      shifty,
                                      incy,
                                      stridey,
                                      workspace,
                                      batch_count);
            }

#undef gemvt_sn_KARGS
        }
        //optimized gemvt kernel with double buffered loads for gfx908.
        else if(!i64_incs && is_atomics_allowed && (m == n) && (m % rocblas_gemv_bx() == 0)
                && (is_gfx908
                    && ((is_float && m > sgemvt_gfx908_lower_threshold)
                        || (is_double && m > dgemvt_gfx908_lower_threshold))))
        {
            if constexpr(is_float || is_double)
            {
                // The following rocblas_gemv_scal_kernel does the `y = y*beta` computation
                static constexpr int NB               = 256;
                const int            gemv_scal_blocks = (n - 1) / NB + 1;
                dim3                 grid(gemv_scal_blocks, 1, batches);
                dim3                 threads(NB);
                if(handle->pointer_mode == rocblas_pointer_mode_device)
                {
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemv_scal_kernel<NB>),
                                          grid,
                                          threads,
                                          0,
                                          rocblas_stream,
                                          n,
                                          beta,
                                          stride_beta,
                                          y,
                                          shifty,
                                          incy,
                                          stridey,
                                          batch_count);
                }
                else
                {
                    if(*beta != 1)
                        ROCBLAS_LAUNCH_KERNEL((rocblas_gemv_scal_kernel<NB>),
                                              grid,
                                              threads,
                                              0,
                                              rocblas_stream,
                                              n,
                                              *beta,
                                              stride_beta,
                                              y,
                                              shifty,
                                              incy,
                                              stridey,
                                              batch_count);
                }
                // The following kernel does the `y += A * x` computation
                static constexpr int thread_x            = rocblas_gemv_bx();
                static constexpr int block_y             = is_float ? 8 : 16;
                static constexpr int thread_y            = is_float ? 8 : 4;
                static constexpr int elements_per_thread = thread_x / (2 * thread_y);

                const int block_x = n / thread_x;
                dim3      gemvt_threads(thread_x, thread_y);
                dim3      gemvt_grid(block_x, block_y, batches);

#define gemvt_double_buffered_KARGS(alpha_)                                                    \
    gemvt_grid, gemvt_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, A, offseta, lda, \
        strideA, x, shiftx, incx, stridex, y, shifty, incy, stridey, batch_count

                if(handle->pointer_mode == rocblas_pointer_mode_device)
                {
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_double_buffered_kernel<CONJ,
                                                              thread_x,
                                                              thread_y,
                                                              elements_per_thread>),
                        gemvt_double_buffered_KARGS(alpha));
                }
                else
                {
                    if(!*alpha)
                        return rocblas_status_success;

                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_gemvt_double_buffered_kernel<CONJ,
                                                              thread_x,
                                                              thread_y,
                                                              elements_per_thread>),
                        gemvt_double_buffered_KARGS(*alpha));
                }
            }
#undef gemvt_double_buffered_KARGS
        }

#define gemvt_KARGS(alpha_, beta_)                                                             \
    gemvt_grid, gemvt_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, A, offseta, lda, \
        strideA, x, shiftx, incx, stridex, beta_, stride_beta, y, shifty, incy, stridey,       \
        batch_count
        //Using kernel code with shared memory reduction for single precision and all other precision when m or n is less than gemvt_threshold.
        else if(!i64_incs && (is_float || m < gemvt_threshold || n < gemvt_threshold))
        {
            //Number of threads per block
            static constexpr int NB = 256;
            dim3                 gemvt_grid(n, 1, batches);
            dim3                 gemvt_threads(NB);
            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_kernel<CONJ, NB>), gemvt_KARGS(alpha, beta));
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;

                ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_kernel<CONJ, NB>), gemvt_KARGS(*alpha, *beta));
            }
        }
        //Using kernel code with warp reduction.
        //Having 1024 threads per block for double, complex-float and complex-double precision GEMV (transpose) for better performance.
        else
        {
            //Number of threads per block
            static constexpr int NB = 1024;
            dim3                 gemvt_grid(n, 1, batches);
            dim3                 gemvt_threads(NB);
            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_warp_reduce_kernel<CONJ, NB, rocblas_int>),
                                          gemvt_KARGS(alpha, beta));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_warp_reduce_kernel<CONJ, NB, int64_t>),
                                          gemvt_KARGS(alpha, beta));
            }
            else
            {
                if(!*alpha && *beta == 1)
                    return rocblas_status_success;

                if(!i64_indices)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_warp_reduce_kernel<CONJ, NB, rocblas_int>),
                                          gemvt_KARGS(*alpha, *beta));
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_gemvt_warp_reduce_kernel<CONJ, NB, int64_t>),
                                          gemvt_KARGS(*alpha, *beta));
            }
        }
#undef gemvt_KARGS
    }
    return rocblas_status_success;
}

template <typename T>
rocblas_status rocblas_internal_gemv_template(rocblas_handle    handle,
                                              rocblas_operation transA,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const T*          alpha,
                                              rocblas_stride    stride_alpha,
                                              const T*          A,
                                              rocblas_stride    offseta,
                                              rocblas_int       lda,
                                              rocblas_stride    strideA,
                                              const T*          x,
                                              rocblas_stride    offsetx,
                                              rocblas_int       incx,
                                              rocblas_stride    stridex,
                                              const T*          beta,
                                              rocblas_stride    stride_beta,
                                              T*                y,
                                              rocblas_stride    offsety,
                                              rocblas_int       incy,
                                              rocblas_stride    stridey,
                                              rocblas_int       batch_count,
                                              T*                workspace)
{
    return rocblas_internal_gemv_launcher<T, T, T>(handle,
                                                   transA,
                                                   m,
                                                   n,
                                                   alpha,
                                                   stride_alpha,
                                                   A,
                                                   offseta,
                                                   (int64_t)lda,
                                                   strideA,
                                                   x,
                                                   offsetx,
                                                   (int64_t)incx,
                                                   stridex,
                                                   beta,
                                                   stride_beta,
                                                   y,
                                                   offsety,
                                                   (int64_t)incy,
                                                   stridey,
                                                   batch_count,
                                                   workspace);
}

template <typename T>
rocblas_status rocblas_internal_gemv_batched_template(rocblas_handle    handle,
                                                      rocblas_operation transA,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      const T*          alpha,
                                                      rocblas_stride    stride_alpha,
                                                      const T* const*   A,
                                                      rocblas_stride    offseta,
                                                      rocblas_int       lda,
                                                      rocblas_stride    strideA,
                                                      const T* const*   x,
                                                      rocblas_stride    offsetx,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stridex,
                                                      const T*          beta,
                                                      rocblas_stride    stride_beta,
                                                      T* const*         y,
                                                      rocblas_stride    offsety,
                                                      rocblas_int       incy,
                                                      rocblas_stride    stridey,
                                                      rocblas_int       batch_count,
                                                      T*                workspace)
{
    return rocblas_internal_gemv_launcher(handle,
                                          transA,
                                          m,
                                          n,
                                          alpha,
                                          stride_alpha,
                                          A,
                                          offseta,
                                          (int64_t)lda,
                                          strideA,
                                          x,
                                          offsetx,
                                          (int64_t)incx,
                                          stridex,
                                          beta,
                                          stride_beta,
                                          y,
                                          offsety,
                                          (int64_t)incy,
                                          stridey,
                                          batch_count,
                                          workspace);
}

template <typename Ti, typename To>
rocblas_status rocblas_gemv_check_numerics(const char*       function_name,
                                           rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           int64_t           m,
                                           int64_t           n,
                                           Ti                A,
                                           rocblas_stride    offset_a,
                                           int64_t           lda,
                                           rocblas_stride    stride_a,
                                           Ti                x,
                                           rocblas_stride    offset_x,
                                           int64_t           inc_x,
                                           rocblas_stride    stride_x,
                                           To                y,
                                           rocblas_stride    offset_y,
                                           int64_t           inc_y,
                                           rocblas_stride    stride_y,
                                           int64_t           batch_count,
                                           const int         check_numerics,
                                           bool              is_input)
{
    rocblas_status check_numerics_status = rocblas_status_success;
    if(is_input)
    {
        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              rocblas_operation_none,
                                                              rocblas_fill_full,
                                                              rocblas_client_general_matrix,
                                                              m,
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

        //Checking trans_a to transpose a vector 'x'
        rocblas_int n_x = trans_a == rocblas_operation_none ? n : m;

        check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                                handle,
                                                                                n_x,
                                                                                x,
                                                                                offset_x,
                                                                                inc_x,
                                                                                stride_x,
                                                                                batch_count,
                                                                                check_numerics,
                                                                                is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }

    //Checking trans_a to transpose a vector 'y'
    rocblas_int n_y       = trans_a == rocblas_operation_none ? m : n;
    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n_y,
                                                                            y,
                                                                            offset_y,
                                                                            inc_y,
                                                                            stride_y,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *gemv*.cpp

#ifdef INSTANTIATE_GEMV_WORKSPACE
#error INSTANTIATE_GEMV_WORKSPACE already defined
#endif

#define INSTANTIATE_GEMV_WORKSPACE(To_)                   \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE size_t      \
        rocblas_internal_gemv_kernel_workspace_size<To_>( \
            rocblas_operation transA, rocblas_int m, rocblas_int n, rocblas_int batch_count);

INSTANTIATE_GEMV_WORKSPACE(float)
INSTANTIATE_GEMV_WORKSPACE(double)
INSTANTIATE_GEMV_WORKSPACE(rocblas_float_complex)
INSTANTIATE_GEMV_WORKSPACE(rocblas_double_complex)
INSTANTIATE_GEMV_WORKSPACE(rocblas_half)
INSTANTIATE_GEMV_WORKSPACE(rocblas_bfloat16)
#undef INSTANTIATE_GEMV_WORKSPACE

#ifdef INSTANTIATE_GEMV_TEMPLATE
#error INSTANTIATE_GEMV_TEMPLATE already defined
#endif

#define INSTANTIATE_GEMV_TEMPLATE(T_)                                                            \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_gemv_template<T_>( \
        rocblas_handle    handle,                                                                \
        rocblas_operation transA,                                                                \
        rocblas_int       m,                                                                     \
        rocblas_int       n,                                                                     \
        const T_*         alpha,                                                                 \
        rocblas_stride    stride_alpha,                                                          \
        const T_*         A,                                                                     \
        rocblas_stride    offseta,                                                               \
        rocblas_int       lda,                                                                   \
        rocblas_stride    strideA,                                                               \
        const T_*         x,                                                                     \
        rocblas_stride    offsetx,                                                               \
        rocblas_int       incx,                                                                  \
        rocblas_stride    stridex,                                                               \
        const T_*         beta,                                                                  \
        rocblas_stride    stride_beta,                                                           \
        T_*               y,                                                                     \
        rocblas_stride    offsety,                                                               \
        rocblas_int       incy,                                                                  \
        rocblas_stride    stridey,                                                               \
        rocblas_int       batch_count,                                                           \
        T_*               workspace);

INSTANTIATE_GEMV_TEMPLATE(float)
INSTANTIATE_GEMV_TEMPLATE(double)
INSTANTIATE_GEMV_TEMPLATE(rocblas_float_complex)
INSTANTIATE_GEMV_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_GEMV_TEMPLATE

#ifdef INSTANTIATE_GEMV_BATCHED_TEMPLATE
#error INSTANTIATE_GEMV_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_GEMV_BATCHED_TEMPLATE(T_)                                      \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                       \
        rocblas_internal_gemv_batched_template<T_>(rocblas_handle    handle,       \
                                                   rocblas_operation transA,       \
                                                   rocblas_int       m,            \
                                                   rocblas_int       n,            \
                                                   const T_*         alpha,        \
                                                   rocblas_stride    stride_alpha, \
                                                   const T_* const*  A,            \
                                                   rocblas_stride    offseta,      \
                                                   rocblas_int       lda,          \
                                                   rocblas_stride    strideA,      \
                                                   const T_* const*  x,            \
                                                   rocblas_stride    offsetx,      \
                                                   rocblas_int       incx,         \
                                                   rocblas_stride    stridex,      \
                                                   const T_*         beta,         \
                                                   rocblas_stride    stride_beta,  \
                                                   T_* const*        y,            \
                                                   rocblas_stride    offsety,      \
                                                   rocblas_int       incy,         \
                                                   rocblas_stride    stridey,      \
                                                   rocblas_int       batch_count,  \
                                                   T_*               workspace);

INSTANTIATE_GEMV_BATCHED_TEMPLATE(float)
INSTANTIATE_GEMV_BATCHED_TEMPLATE(double)
INSTANTIATE_GEMV_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_GEMV_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_GEMV_BATCHED_TEMPLATE

// For mixed-precision gemv
#ifdef INST_GEMV_MIXED_LAUNCHER
#error INST_GEMV_MIXED_LAUNCHER already defined
#endif

#define INST_GEMV_MIXED_LAUNCHER(Ti_, Tex_, To_)                                       \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                           \
        rocblas_internal_gemv_launcher<Ti_, Tex_, To_>(rocblas_handle    handle,       \
                                                       rocblas_operation transA,       \
                                                       rocblas_int       m,            \
                                                       rocblas_int       n,            \
                                                       const Tex_*       alpha,        \
                                                       rocblas_stride    stride_alpha, \
                                                       Ti_ const*        A,            \
                                                       rocblas_stride    offseta,      \
                                                       int64_t           lda,          \
                                                       rocblas_stride    strideA,      \
                                                       Ti_ const*        x,            \
                                                       rocblas_stride    offsetx,      \
                                                       int64_t           incx,         \
                                                       rocblas_stride    stridex,      \
                                                       const Tex_*       beta,         \
                                                       rocblas_stride    stride_beta,  \
                                                       To_*              y,            \
                                                       rocblas_stride    offsety,      \
                                                       int64_t           incy,         \
                                                       rocblas_stride    stridey,      \
                                                       rocblas_int       batch_count,  \
                                                       Tex_*             workspace);

INST_GEMV_MIXED_LAUNCHER(rocblas_half, float, rocblas_half)
INST_GEMV_MIXED_LAUNCHER(rocblas_half const*, float, rocblas_half* const)
INST_GEMV_MIXED_LAUNCHER(rocblas_half, float, float)
INST_GEMV_MIXED_LAUNCHER(rocblas_half const*, float, float* const)
INST_GEMV_MIXED_LAUNCHER(rocblas_bfloat16, float, rocblas_bfloat16)
INST_GEMV_MIXED_LAUNCHER(rocblas_bfloat16 const*, float, rocblas_bfloat16* const)
INST_GEMV_MIXED_LAUNCHER(rocblas_bfloat16, float, float)
INST_GEMV_MIXED_LAUNCHER(rocblas_bfloat16 const*, float, float* const)

#undef INST_GEMV_MIXED_LAUNCHER

#ifdef INSTANTIATE_GEMV_NUMERICS
#error INSTANTIATE_GEMV_NUMERICS already defined
#endif

#define INSTANTIATE_GEMV_NUMERICS(Ti_, To_)                                                         \
    template rocblas_status rocblas_gemv_check_numerics<Ti_, To_>(const char*       function_name,  \
                                                                  rocblas_handle    handle,         \
                                                                  rocblas_operation trans_a,        \
                                                                  int64_t           m,              \
                                                                  int64_t           n,              \
                                                                  Ti_               A,              \
                                                                  rocblas_stride    offset_a,       \
                                                                  int64_t           lda,            \
                                                                  rocblas_stride    stride_a,       \
                                                                  Ti_               x,              \
                                                                  rocblas_stride    offset_x,       \
                                                                  int64_t           inc_x,          \
                                                                  rocblas_stride    stride_x,       \
                                                                  To_               y,              \
                                                                  rocblas_stride    offset_y,       \
                                                                  int64_t           inc_y,          \
                                                                  rocblas_stride    stride_y,       \
                                                                  int64_t           batch_count,    \
                                                                  const int         check_numerics, \
                                                                  bool              is_input);

INSTANTIATE_GEMV_NUMERICS(float const*, float*)
INSTANTIATE_GEMV_NUMERICS(double const*, double*)
INSTANTIATE_GEMV_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_GEMV_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_GEMV_NUMERICS(float const* const*, float* const*)
INSTANTIATE_GEMV_NUMERICS(double const* const*, double* const*)
INSTANTIATE_GEMV_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_GEMV_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)
INSTANTIATE_GEMV_NUMERICS(rocblas_half const*, rocblas_half*)
INSTANTIATE_GEMV_NUMERICS(rocblas_half const* const*, rocblas_half* const*)
INSTANTIATE_GEMV_NUMERICS(rocblas_half const*, float*)
INSTANTIATE_GEMV_NUMERICS(rocblas_half const* const*, float* const*)
INSTANTIATE_GEMV_NUMERICS(rocblas_bfloat16 const*, rocblas_bfloat16*)
INSTANTIATE_GEMV_NUMERICS(rocblas_bfloat16 const* const*, rocblas_bfloat16* const*)
INSTANTIATE_GEMV_NUMERICS(rocblas_bfloat16 const*, float*)
INSTANTIATE_GEMV_NUMERICS(rocblas_bfloat16 const* const*, float* const*)

#undef INSTANTIATE_GEMV_NUMERICS
