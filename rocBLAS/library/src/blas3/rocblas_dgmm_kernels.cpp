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

#include "device_macros.hpp"
#include "handle.hpp"
#include "rocblas_dgmm.hpp"
#include "rocblas_level3_threshold.hpp"

template <int DIM_X, int DIM_Y, bool side_right, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_dgmm_kernel(rocblas_int    m,
                    rocblas_int    n,
                    TConstPtr      Aa,
                    rocblas_stride offset_A,
                    int64_t        lda,
                    rocblas_stride stride_A,
                    TConstPtr      Xa,
                    int64_t        shift_x,
                    int64_t        incx,
                    rocblas_stride stride_x,
                    TPtr           Ca,
                    rocblas_stride offset_C,
                    int64_t        ldc,
                    rocblas_stride stride_C,
                    rocblas_int    batch_count)
{
    rocblas_int tx    = blockIdx.x * DIM_X + threadIdx.x;
    uint32_t    batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        //looping over ty
        for(rocblas_int ty = blockIdx.y * DIM_Y + threadIdx.y; ty < n && tx < m;
            ty += DIM_Y * gridDim.y)
        {
            auto* A = load_ptr_batch(Aa, batch, offset_A, stride_A);
            auto* X = load_ptr_batch(Xa, batch, shift_x, stride_x);
            auto* C = load_ptr_batch(Ca, batch, offset_C, stride_C);

            if constexpr(side_right)
            {
                C[tx + ldc * ty] = A[tx + lda * ty] * X[ty * incx];
            }
            else
            {
                C[tx + ldc * ty] = A[tx + lda * ty] * X[tx * incx];
            }
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <int DIM_X, int DIM_Y, bool side_right, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_dgmm_gfx942_kernel(rocblas_int    m,
                           rocblas_int    n,
                           TConstPtr      Aa,
                           rocblas_stride offset_A,
                           int64_t        lda,
                           rocblas_stride stride_A,
                           TConstPtr      Xa,
                           int64_t        shift_x,
                           int64_t        incx,
                           rocblas_stride stride_x,
                           TPtr           Ca,
                           rocblas_stride offset_C,
                           int64_t        ldc,
                           rocblas_stride stride_C)
{
// gfx942 kernels
#if defined(__gfx942__)

    rocblas_int tx = (blockIdx.x * DIM_X + threadIdx.x) * 2;
    rocblas_int ty = blockIdx.y * DIM_Y + threadIdx.y;

    const auto* A = load_ptr_batch(Aa, blockIdx.z, offset_A, stride_A);
    const auto* x = load_ptr_batch(Xa, blockIdx.z, shift_x, stride_x);
    auto*       C = load_ptr_batch(Ca, blockIdx.z, offset_C, stride_C);

    if constexpr(side_right)
    {
        if(ty < n)
        {
            const auto reg_x = x[ty * incx];

            const auto reg_A_1 = (tx < m) ? A[tx + ty * lda] : 0;
            const auto reg_A_2 = ((tx + 1) < m) ? A[(tx + 1) + ty * lda] : 0;

            //scalar-Matrix-vector product and put the result to the last element of 'C'.
            if((m & 1) != 0 && (tx + 1) == m)
                C[tx + ldc * ty] = A[tx + ty * lda] * reg_x;

            //scalar-Matrix-vector product and put the result to the matrix 'C'.
            if((tx + 1) < m)
            {
                C[tx + ldc * ty]       = A[tx + ty * lda] * reg_x;
                C[(tx + 1) + ldc * ty] = A[(tx + 1) + ty * lda] * reg_x;
            }
        }
    }
    else
    {
        if(ty < n)
        {
            const auto reg_x_1 = (tx < m) ? x[tx * incx] : 0;
            const auto reg_x_2 = ((tx + 1) < m) ? x[(tx + 1) * incx] : 0;

            const auto reg_A_1 = (tx < m) ? A[tx + ty * lda] : 0;
            const auto reg_A_2 = ((tx + 1) < m) ? A[(tx + 1) + ty * lda] : 0;

            //scalar-matrix-vector product and put the result to the last element of 'C'.
            if((m & 1) != 0 && (tx + 1) == m)
                C[tx + ldc * ty] = reg_A_1 * reg_x_1;

            //scalar-matrix-vector product and put the result to the matrix 'C'.
            if((tx + 1) < m)
            {
                C[tx + ldc * ty]       = reg_A_1 * reg_x_1;
                C[(tx + 1) + ldc * ty] = reg_A_2 * reg_x_2;
            }
        }
    }
#endif
}

/*
 * ===========================================================================
 *    template interface
 *    template specialization
 *    call DGMM C interfaces (see rocblas_dgmm*.cpp in the same dir)
 * ===========================================================================
 */

/**
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (float, double, rocblas_complex, or rocblas_double_complex)
 */

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_dgmm_launcher(rocblas_handle handle,
                                              rocblas_side   side,
                                              rocblas_int    m,
                                              rocblas_int    n,
                                              TConstPtr      A,
                                              rocblas_stride offset_A,
                                              int64_t        lda,
                                              rocblas_stride stride_A,
                                              TConstPtr      x,
                                              rocblas_stride offset_x,
                                              int64_t        incx,
                                              rocblas_stride stride_x,
                                              TPtr           C,
                                              rocblas_stride offset_C,
                                              int64_t        ldc,
                                              rocblas_stride stride_C,
                                              rocblas_int    batch_count)

{
    hipStream_t rocblas_stream = handle->get_stream();
    int         batches        = handle->getBatchGridDim((int)batch_count);

    // in case of negative incx shift pointer to end of data for negative indexing
    rocblas_int k       = side == rocblas_side_left ? m : n;
    ptrdiff_t   shift_x = offset_x - ((incx < 0) ? ptrdiff_t(incx) * (k - 1) : 0);

    bool is_gfx942 = handle->getArch() == 942 ? true : false;

    //Identifying the precision to have an appropriate optimization
    static constexpr bool is_float
        = std::is_same_v<TPtr, float*> || std::is_same_v<TPtr, float* const*>;
    static constexpr bool is_double
        = std::is_same_v<TPtr, double*> || std::is_same_v<TPtr, double* const*>;
    static constexpr bool is_complex_float
        = std::is_same_v<
              TPtr,
              rocblas_float_complex*> || std::is_same_v<TPtr, rocblas_float_complex* const*>;

#define dgmm_KARGS                                                                            \
    dgmm_grid, dgmm_threads, 0, rocblas_stream, m, n, A, offset_A, lda, stride_A, x, shift_x, \
        incx, stride_x, C, offset_C, ldc, stride_C, batch_count

#define dgmm_gfx942_KARGS                                                                     \
    dgmm_grid, dgmm_threads, 0, rocblas_stream, m, n, A, offset_A, lda, stride_A, x, shift_x, \
        incx, stride_x, C, offset_C, ldc, stride_C

    if(rocblas_side_left == side)
    {
        if(is_gfx942
           && (is_float
               || ((is_double || is_complex_float) && m > dcdgmm_gfx942_m_lower_threshold)))
        {
            static constexpr int DGMM_DIM_X = 32;
            static constexpr int DGMM_DIM_Y = 32;

            rocblas_int blocksX = (m - 1) / (DGMM_DIM_X * 2) + 1;
            rocblas_int blocksY = (n - 1) / DGMM_DIM_Y + 1;

            dim3 dgmm_grid(blocksX, blocksY, batch_count);
            dim3 dgmm_threads(DGMM_DIM_X, DGMM_DIM_Y);

            ROCBLAS_LAUNCH_KERNEL((rocblas_dgmm_gfx942_kernel<DGMM_DIM_X, DGMM_DIM_Y, false>),
                                  dgmm_gfx942_KARGS);
        }
        else
        {
            static constexpr int DGMM_DIM_X = 16;
            static constexpr int DGMM_DIM_Y = 16;

            rocblas_int blocksX = (m - 1) / DGMM_DIM_X + 1;
            //blocksY should be <= 2^16 (65536) to avoid overflow as grid y and z block indices support only 16-bit values on some gfx
            rocblas_int blocksY = std::min(c_YZ_grid_launch_limit, (n - 1) / DGMM_DIM_Y + 1);

            dim3 dgmm_grid(blocksX, blocksY, batches);
            dim3 dgmm_threads(DGMM_DIM_X, DGMM_DIM_Y);

            ROCBLAS_LAUNCH_KERNEL((rocblas_dgmm_kernel<DGMM_DIM_X, DGMM_DIM_Y, false>), dgmm_KARGS);
        }
    }
    else
    {
        if(is_gfx942
           && (is_float
               || ((is_double || is_complex_float) && m > dcdgmm_gfx942_m_lower_threshold)))
        {
            static constexpr int DGMM_DIM_X = 32;
            static constexpr int DGMM_DIM_Y = 32;

            rocblas_int blocksX = (m - 1) / (DGMM_DIM_X * 2) + 1;
            rocblas_int blocksY = (n - 1) / DGMM_DIM_Y + 1;

            dim3 dgmm_grid(blocksX, blocksY, batch_count);
            dim3 dgmm_threads(DGMM_DIM_X, DGMM_DIM_Y);

            ROCBLAS_LAUNCH_KERNEL((rocblas_dgmm_gfx942_kernel<DGMM_DIM_X, DGMM_DIM_Y, true>),
                                  dgmm_gfx942_KARGS);
        }
        else
        {
            static constexpr int DGMM_DIM_X = 16;
            static constexpr int DGMM_DIM_Y = 16;

            rocblas_int blocksX = (m - 1) / DGMM_DIM_X + 1;
            //blocksY should be <= 2^16 (65536) to avoid overflow as grid y and z block indices support only 16-bit values on some gfx
            rocblas_int blocksY = std::min(c_YZ_grid_launch_limit, (n - 1) / DGMM_DIM_Y + 1);

            dim3 dgmm_grid(blocksX, blocksY, batches);
            dim3 dgmm_threads(DGMM_DIM_X, DGMM_DIM_Y);

            ROCBLAS_LAUNCH_KERNEL((rocblas_dgmm_kernel<DGMM_DIM_X, DGMM_DIM_Y, true>), dgmm_KARGS);
        }
    }
    return rocblas_status_success;
}

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_dgmm_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_side   side,
                                           int64_t        m,
                                           int64_t        n,
                                           TConstPtr      A,
                                           int64_t        lda,
                                           rocblas_stride stride_A,
                                           TConstPtr      x,
                                           int64_t        incx,
                                           rocblas_stride stride_x,
                                           TPtr           C,
                                           int64_t        ldc,
                                           rocblas_stride stride_C,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{

    rocblas_status check_numerics_status = rocblas_status_success;
    if(is_input)
    {
        rocblas_int dim_x = (side == rocblas_side_left) ? m : n;
        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              rocblas_operation_none,
                                                              rocblas_fill_full,
                                                              rocblas_client_general_matrix,
                                                              m,
                                                              n,
                                                              A,
                                                              0,
                                                              lda,
                                                              stride_A,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;

        check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                                handle,
                                                                                dim_x,
                                                                                x,
                                                                                0,
                                                                                incx,
                                                                                stride_x,
                                                                                batch_count,
                                                                                check_numerics,
                                                                                is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }
    check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          m,
                                                          n,
                                                          C,
                                                          0,
                                                          ldc,
                                                          stride_C,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files dgmm*.cpp

#ifdef INSTANTIATE_DGMM_LAUNCHER
#error INSTANTIATE_DGMM_LAUNCHER already defined
#endif

#define INSTANTIATE_DGMM_LAUNCHER(TConstPtr_, TPtr_)                           \
    template rocblas_status rocblas_internal_dgmm_launcher<TConstPtr_, TPtr_>( \
        rocblas_handle handle,                                                 \
        rocblas_side   side,                                                   \
        rocblas_int    m,                                                      \
        rocblas_int    n,                                                      \
        TConstPtr_     A,                                                      \
        rocblas_stride offset_A,                                               \
        int64_t        lda,                                                    \
        rocblas_stride stride_A,                                               \
        TConstPtr_     X,                                                      \
        rocblas_stride offset_x,                                               \
        int64_t        incx,                                                   \
        rocblas_stride stride_x,                                               \
        TPtr_          C,                                                      \
        rocblas_stride offset_C,                                               \
        int64_t        ldc,                                                    \
        rocblas_stride stride_C,                                               \
        rocblas_int    batch_count);

// instantiate for rocblas_Xdgmm and rocblas_Xdgmm_strided_batched
INSTANTIATE_DGMM_LAUNCHER(float const*, float*)
INSTANTIATE_DGMM_LAUNCHER(double const*, double*)
INSTANTIATE_DGMM_LAUNCHER(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_DGMM_LAUNCHER(rocblas_double_complex const*, rocblas_double_complex*)

// instantiate for rocblas_Xdgmm_batched
INSTANTIATE_DGMM_LAUNCHER(float const* const*, float* const*)
INSTANTIATE_DGMM_LAUNCHER(double const* const*, double* const*)
INSTANTIATE_DGMM_LAUNCHER(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_DGMM_LAUNCHER(rocblas_double_complex const* const*, rocblas_double_complex* const*)
#undef INSTANTIATE_DGMM_LAUNCHER

#ifdef INSTANTIATE_DGMM_NUMERICS
#error INSTANTIATE_DGMM_NUMERICS already defined
#endif

#define INSTANTIATE_DGMM_NUMERICS(TConstPtr_, TPtr_)                        \
    template rocblas_status rocblas_dgmm_check_numerics<TConstPtr_, TPtr_>( \
        const char*    function_name,                                       \
        rocblas_handle handle,                                              \
        rocblas_side   side,                                                \
        int64_t        m,                                                   \
        int64_t        n,                                                   \
        TConstPtr_     A,                                                   \
        int64_t        lda,                                                 \
        rocblas_stride stride_A,                                            \
        TConstPtr_     x,                                                   \
        int64_t        inc,                                                 \
        rocblas_stride stride_x,                                            \
        TPtr_          C,                                                   \
        int64_t        ldc,                                                 \
        rocblas_stride stride_C,                                            \
        int64_t        batch_count,                                         \
        const int      check_numerics,                                      \
        bool           is_input);

// instantiate for rocblas_Xdgmm and rocblas_Xdgmm_strided_batched
INSTANTIATE_DGMM_NUMERICS(float const*, float*)
INSTANTIATE_DGMM_NUMERICS(double const*, double*)
INSTANTIATE_DGMM_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_DGMM_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)

// instantiate for rocblas_Xdgmm_batched
INSTANTIATE_DGMM_NUMERICS(float const* const*, float* const*)
INSTANTIATE_DGMM_NUMERICS(double const* const*, double* const*)
INSTANTIATE_DGMM_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_DGMM_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_DGMM_NUMERICS
