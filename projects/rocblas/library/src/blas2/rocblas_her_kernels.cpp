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

#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"
#include "device_macros.hpp"
#include "handle.hpp"
#include "rocblas_her.hpp"

template <int DIM_X, typename T, typename U>
ROCBLAS_KERNEL_ILF void rocblas_her_kernel_calc(bool        is_upper,
                                                rocblas_int n,
                                                U           alpha,
                                                const T* __restrict__ x,
                                                int64_t incx,
                                                T* __restrict__ A,
                                                int64_t lda)
{
    int32_t tx  = threadIdx.x;
    int32_t col = blockIdx.x;

    if(tx < n)
        A += tx;

    //Each BlockIdx.x takes care of each column of matrix A
    A += col * int64_t(lda);

    const T res_x = conj(x[col * int64_t(incx)]) * alpha;

    if(is_upper)
    {
        //scalar-vector-vector product and add the result to a Hermitian matrix 'A'.
        //If n > DIM_X, then the threads are reused and the multiplied values will be accumulated with matrix A.
        int32_t i = 0;
        for(; tx + i < col; i += DIM_X)
        {
            A[i] += res_x * x[(tx + i) * int64_t(incx)];
        }
        //Diagonal elements must be real
        if(tx + i == col)
        {
            A[i] = std::real(A[i]) + std::real(x[col * int64_t(incx)] * res_x);
        }
    }
    else
    {
        int32_t i = col + 1;
        //Diagonal elements must be real
        if(tx == 0)
        {
            A[i - 1] = std::real(A[i - 1]) + std::real(x[col * int64_t(incx)] * res_x);
        }
        //scalar-vector-vector product and add the result to a Hermitian matrix 'A'.
        //If n > DIM_X, then the threads are reused and the multiplied values will be accumulated with matrix A.
        for(; tx + i < n; i += DIM_X)
        {
            A[i] += res_x * x[(tx + i) * int64_t(incx)];
        }
    }
}

template <int DIM_X, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X)
rocblas_her_kernel(bool           is_upper,
                   rocblas_int    n,
                   TScal          alpha_device_host,
                   TConstPtr      xa,
                   rocblas_stride shift_x,
                   int64_t        incx,
                   rocblas_stride stride_x,
                   TPtr           Aa,
                   int64_t        lda,
                   rocblas_stride shift_A,
                   rocblas_stride stride_A,
                   rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, 0); //stride_alpha is always 0
        if(!alpha)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        auto*       A = load_ptr_batch(Aa, batch, shift_A, stride_A);
        const auto* x = load_ptr_batch(xa, batch, shift_x, stride_x);

        rocblas_her_kernel_calc<DIM_X>(is_upper, n, alpha, x, incx, A, lda);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/**
 * TScal     is always: const U* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (rocblas_float_complex or rocblas_double_complex)
 * and U is the scalar type (float or double)
 */
template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_her_launcher(rocblas_handle handle,
                                    rocblas_fill   uplo,
                                    API_INT        n,
                                    TScal          alpha,
                                    TConstPtr      x,
                                    rocblas_stride offset_x,
                                    int64_t        incx,
                                    rocblas_stride stride_x,
                                    TPtr           A,
                                    rocblas_stride offset_A,
                                    int64_t        lda,
                                    rocblas_stride stride_A,
                                    API_INT        batch_count)
{
    // Quick return if possible. Not Argument error
    if(!n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();

    // in case of negative inc, shift pointer to end of data for negative indexing tid*inc
    ptrdiff_t shift_x = incx < 0 ? offset_x - ptrdiff_t(incx) * (n - 1) : offset_x;

    int batches = handle->getBatchGridDim((int)batch_count);

#define her_KARGS(alpha_)                                                                        \
    her_grid, her_threads, 0, rocblas_stream, uplo == rocblas_fill_upper, n, alpha_, x, shift_x, \
        incx, stride_x, A, lda, offset_A, stride_A, batch_count

    static constexpr int HER_DIM_X = 1024;

    dim3 her_grid(n, 1, batches);
    dim3 her_threads(HER_DIM_X);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        ROCBLAS_LAUNCH_KERNEL((rocblas_her_kernel<HER_DIM_X>), her_KARGS(alpha));
    }
    else
    {
        ROCBLAS_LAUNCH_KERNEL((rocblas_her_kernel<HER_DIM_X>), her_KARGS(*alpha));
    }
#undef her_KARGS
    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_he_matrix_template for checking Matrix `A` which is a Hermitian Matrix
template <typename T, typename U>
rocblas_status rocblas_her_check_numerics(const char*    function_name,
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
    }

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *her*.cpp

#ifdef INST_HER_LAUNCHER
#error INST_HER_LAUNCHER  already defined
#endif

#define INST_HER_LAUNCHER(TI_, Tscal_, TConstPtr_, TPtr_)                         \
    template rocblas_status rocblas_her_launcher<TI_, Tscal_, TConstPtr_, TPtr_>( \
        rocblas_handle handle,                                                    \
        rocblas_fill   uplo,                                                      \
        TI_            n,                                                         \
        Tscal_         alpha,                                                     \
        TConstPtr_     x,                                                         \
        rocblas_stride offset_x,                                                  \
        int64_t        incx,                                                      \
        rocblas_stride stride_x,                                                  \
        TPtr_          A,                                                         \
        int64_t        lda,                                                       \
        rocblas_stride offset_A,                                                  \
        rocblas_stride stride_A,                                                  \
        TI_            batch_count);

INST_HER_LAUNCHER(rocblas_int, float const*, rocblas_float_complex const*, rocblas_float_complex*)
INST_HER_LAUNCHER(rocblas_int,
                  double const*,
                  rocblas_double_complex const*,
                  rocblas_double_complex*)
INST_HER_LAUNCHER(rocblas_int,
                  float const*,
                  rocblas_float_complex const* const*,
                  rocblas_float_complex* const*)
INST_HER_LAUNCHER(rocblas_int,
                  double const*,
                  rocblas_double_complex const* const*,
                  rocblas_double_complex* const*)

#undef INST_HER_LAUNCHER

#ifdef INST_HER_NUMERICS
#error INST_HER_NUMERICS already defined
#endif

#define INST_HER_NUMERICS(T_, U_)                                                             \
    template rocblas_status rocblas_her_check_numerics<T_, U_>(const char*    function_name,  \
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
                                                               int64_t        batch_count,    \
                                                               const int      check_numerics, \
                                                               bool           is_input);

INST_HER_NUMERICS(rocblas_float_complex*, rocblas_float_complex const*);
INST_HER_NUMERICS(rocblas_double_complex*, rocblas_double_complex const*);
INST_HER_NUMERICS(rocblas_float_complex* const*, rocblas_float_complex const* const*);
INST_HER_NUMERICS(rocblas_double_complex* const*, rocblas_double_complex const* const*);
