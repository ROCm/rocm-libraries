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

#include "../blas1/rocblas_copy.hpp"
#include "check_numerics_vector.hpp"
#include "rocblas_tbsv.hpp"

template <bool UPPER, bool TRANS>
ROCBLAS_KERNEL_ILF inline size_t rocblas_banded_matrix_index(
    rocblas_int n, int64_t lda, rocblas_int k, rocblas_int row, rocblas_int col)
{
    return UPPER ? (TRANS ? ((row * lda + col + (k - row))) : (col * lda + row + (k - col)))
                 : (TRANS ? ((row * lda + (col - row))) : (col * lda + (row - col)));
}

// Uses forward substitution to solve Ax = b. Used for a non-transposed lower-triangular matrix
// or a transposed upper-triangular matrix.
template <bool CONJ, bool TRANS, rocblas_int BLK_SIZE, typename T>
ROCBLAS_KERNEL_ILF void rocblas_tbsv_forward_substitution_calc(
    bool is_unit_diag, int n, int k, const T* A, int64_t lda, T* x, int64_t incx)
{
    __shared__ T xshared[BLK_SIZE];
    int          tx = threadIdx.x;

    // main loop - iterate forward in BLK_SIZE chunks
    for(rocblas_int i = 0; i < n; i += BLK_SIZE)
    {
        // cache x into shared memory
        if(tx + i < n)
            xshared[tx] = x[(tx + i) * incx];

        __syncthreads();

        // iterate through the current block and solve elements
        for(rocblas_int j = 0; j < BLK_SIZE; j++)
        {
            // If the current block covers more than what's left,
            // we break early.
            if(j + i >= n)
                break;

            // solve element that can be solved
            if(tx == j && !is_unit_diag)
            {
                rocblas_int colA = j + i;
                rocblas_int rowA = j + i;
                size_t indexA    = rocblas_banded_matrix_index<TRANS, TRANS>(n, lda, k, rowA, colA);
                xshared[tx]      = xshared[tx] / (CONJ ? conj(A[indexA]) : A[indexA]);
            }

            __syncthreads();

            // for rest of block, subtract previous solved part
            if(tx > j)
            {
                rocblas_int colA = j + i;
                rocblas_int rowA = tx + i;
                size_t indexA    = rocblas_banded_matrix_index<TRANS, TRANS>(n, lda, k, rowA, colA);

                // Ensure row is in range, and subtract
                if(rowA < n && colA >= rowA - k)
                    xshared[tx] -= (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[j];
            }
        }

        __syncthreads();

        // apply solved diagonal block to the rest of the array
        // 1. Iterate down rows
        for(rocblas_int j = BLK_SIZE + i; j < n; j += BLK_SIZE)
        {
            if(tx + j >= n)
                break;

            // 2. Sum result (across columns) to be subtracted from original value
            T val = 0;
            for(rocblas_int p = 0; p < BLK_SIZE; p++)
            {
                rocblas_int colA = i + p;
                rocblas_int rowA = tx + j;
                size_t indexA    = rocblas_banded_matrix_index<TRANS, TRANS>(n, lda, k, rowA, colA);

                if(is_unit_diag && colA == rowA)
                    val += xshared[p];
                else if(colA < n && colA >= rowA - k)
                    val += (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[p];
            }

            x[(tx + j) * int64_t(incx)] -= val;
        }

        // store solved part back to global memory
        if(tx + i < n)
            x[(tx + i) * int64_t(incx)] = xshared[tx];

        __syncthreads();
    }
}

// Uses backward substitution to solve Ax = b. Used for a non-transposed upper-triangular matrix
// or a transposed lower-triangular matrix.
template <bool CONJ, bool TRANS, rocblas_int BLK_SIZE, typename T>
ROCBLAS_KERNEL_ILF void rocblas_tbsv_backward_substitution_calc(
    bool is_unit_diag, int n, int k, const T* A, int64_t lda, T* x, int64_t incx)
{
    __shared__ T xshared[BLK_SIZE];
    int          tx = threadIdx.x;

    // main loop - Start at end of array and iterate backwards in BLK_SIZE chunks
    for(rocblas_int i = n - BLK_SIZE; i > -BLK_SIZE; i -= BLK_SIZE)
    {
        // cache x into shared memory
        if(tx + i >= 0)
            xshared[tx] = x[(tx + i) * incx];

        __syncthreads();

        // Iterate backwards through the current block to solve elements.
        for(rocblas_int j = BLK_SIZE - 1; j >= 0; j--)
        {
            // If the current block covers more than what's left,
            // we break; early
            if(j + i < 0)
                break;

            // Solve the new element that can be solved
            if(tx == j && !is_unit_diag)
            {
                rocblas_int colA = j + i;
                rocblas_int rowA = j + i;
                size_t indexA = rocblas_banded_matrix_index<!TRANS, TRANS>(n, lda, k, rowA, colA);
                xshared[tx]   = xshared[tx] / (CONJ ? conj(A[indexA]) : A[indexA]);
            }

            __syncthreads();

            // for rest of block, subtract previous solved part
            if(tx < j)
            {
                rocblas_int colA = j + i;
                rocblas_int rowA = tx + i;
                size_t indexA = rocblas_banded_matrix_index<!TRANS, TRANS>(n, lda, k, rowA, colA);

                // Ensure row is in range, and subtract
                if(rowA >= 0 && colA <= rowA + k)
                    xshared[tx] -= (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[j];
            }
        }

        __syncthreads();

        // apply solved diagonal block to the rest of the array
        // 1. Iterate up rows, starting at the block above the current block
        for(rocblas_int j = i - BLK_SIZE; j > -BLK_SIZE; j -= BLK_SIZE)
        {
            if(tx + j < 0)
                break;

            // 2. Sum result (across columns) to be subtracted from the original value
            T val = 0;
            for(rocblas_int p = 0; p < BLK_SIZE; p++)
            {
                rocblas_int colA = i + p;
                rocblas_int rowA = tx + j;
                size_t indexA = rocblas_banded_matrix_index<!TRANS, TRANS>(n, lda, k, rowA, colA);

                if(is_unit_diag && colA == rowA)
                    val += xshared[p];
                else if(colA <= rowA + k)
                    val += (CONJ ? conj(A[indexA]) : A[indexA]) * xshared[p];
            }

            x[(tx + j) * incx] -= val;
        }

        // store solved part back to global memory
        if(tx + i >= 0)
            x[(tx + i) * incx] = xshared[tx];

        __syncthreads();
    }
}

/**
     *  Calls forwards/backwards substitution kernels with appropriate arguments.
     *  Note the attribute here - apparently this is needed for group sizes > 256.
     *  Currently BLK_SIZE is set to so this is required or else we get
     *  incorrect behaviour.
     *
     *  To optimize we can probably use multiple kernels (a substitution kernel and a
     *  multiplication kernel) so we can use multiple blocks instead of a single one.
     */
template <bool CONJ, rocblas_int BLK_SIZE, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(BLK_SIZE)
rocblas_tbsv_kernel(rocblas_operation transA,
                    bool              is_upper,
                    bool              is_unit_diag,
                    rocblas_int       n,
                    rocblas_int       k,
                    TConstPtr         Aa,
                    rocblas_stride    shift_A,
                    int64_t           lda,
                    rocblas_stride    stride_A,
                    TPtr              xa,
                    rocblas_stride    shift_x,
                    int64_t           incx,
                    rocblas_stride    stride_x)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        const auto* A = load_ptr_batch(Aa, blockIdx.x, shift_A, stride_A);
        auto*       x = load_ptr_batch(xa, blockIdx.x, shift_x, stride_x);

        if(transA == rocblas_operation_none)
        {
            if(is_upper)
                rocblas_tbsv_backward_substitution_calc<false, false, BLK_SIZE>(
                    is_unit_diag, n, k, A, lda, x, incx);
            else
                rocblas_tbsv_forward_substitution_calc<false, false, BLK_SIZE>(
                    is_unit_diag, n, k, A, lda, x, incx);
        }
        else if(is_upper)
            rocblas_tbsv_forward_substitution_calc<CONJ, true, BLK_SIZE>(
                is_unit_diag, n, k, A, lda, x, incx);
        else
            rocblas_tbsv_backward_substitution_calc<CONJ, true, BLK_SIZE>(
                is_unit_diag, n, k, A, lda, x, incx);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_tbsv_launcher(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              TConstPtr         A,
                                              rocblas_stride    offset_A,
                                              int64_t           lda,
                                              rocblas_stride    stride_A,
                                              TPtr              x,
                                              rocblas_stride    offset_x,
                                              int64_t           incx,
                                              rocblas_stride    stride_x,
                                              rocblas_int       batch_count)
{
    if(batch_count == 0 || n == 0)
        return rocblas_status_success;

    // Temporarily switch to host pointer mode, restoring on return
    // cppcheck-suppress unreadVariable
    auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

    ptrdiff_t shift_x = incx < 0 ? offset_x - incx * (n - 1) : offset_x;
    ptrdiff_t shift_A = offset_A;

    static constexpr rocblas_int NB = ROCBLAS_TBSV_NB;

    //Currently NB=512 and it should be NB > 256 as this is required or else we get incorrect behaviour.
    static_assert(NB > 256);

    dim3 grid(batch_count);
    dim3 threads(NB);

    if(transA == rocblas_operation_conjugate_transpose)
    {
        ROCBLAS_LAUNCH_KERNEL((rocblas_tbsv_kernel<true, NB>),
                              grid,
                              threads,
                              0,
                              handle->get_stream(),
                              transA,
                              uplo == rocblas_fill_upper,
                              diag == rocblas_diagonal_unit,
                              n,
                              k,
                              A,
                              shift_A,
                              lda,
                              stride_A,
                              x,
                              shift_x,
                              incx,
                              stride_x);
    }
    else
    {
        ROCBLAS_LAUNCH_KERNEL((rocblas_tbsv_kernel<false, NB>),
                              grid,
                              threads,
                              0,
                              handle->get_stream(),
                              transA,
                              uplo == rocblas_fill_upper,
                              diag == rocblas_diagonal_unit,
                              n,
                              k,
                              A,
                              shift_A,
                              lda,
                              stride_A,
                              x,
                              shift_x,
                              incx,
                              stride_x);
    }

    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_tb_matrix_template for checking Matrix `A` which is a Triangular Band Matrix
template <typename T, typename U>
rocblas_status rocblas_tbsv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           int64_t        n,
                                           T              A,
                                           rocblas_stride offset_A,
                                           int64_t        lda,
                                           rocblas_stride stride_A,
                                           U              x,
                                           rocblas_stride offset_x,
                                           int64_t        incx,
                                           rocblas_stride stride_x,
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
                                                          incx,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *tbsv*.cpp

// clang-format off

#ifdef INSTANTIATE_TBSV_TEMPLATE
#error INSTANTIATE_TBSV_TEMPLATE already defined
#endif

#define INSTANTIATE_TBSV_TEMPLATE(TConstPtr_, TPtr_)             \
template rocblas_status rocblas_internal_tbsv_launcher<TConstPtr_, TPtr_> \
                                    (rocblas_handle    handle,           \
                                     rocblas_fill      uplo,             \
                                     rocblas_operation transA,           \
                                     rocblas_diagonal  diag,             \
                                     rocblas_int       n,                \
                                     rocblas_int       k,                \
                                     TConstPtr_        A,                \
                                     rocblas_stride       offset_A,         \
                                     int64_t       lda,              \
                                     rocblas_stride    stride_A,         \
                                     TPtr_             x,                \
                                     rocblas_stride       offset_x,         \
                                     int64_t       incx,             \
                                     rocblas_stride    stride_x,         \
                                     rocblas_int       batch_count);

INSTANTIATE_TBSV_TEMPLATE(float const*, float*)
INSTANTIATE_TBSV_TEMPLATE(double const*, double*)
INSTANTIATE_TBSV_TEMPLATE(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_TBSV_TEMPLATE(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_TBSV_TEMPLATE(float const* const*, float* const*)
INSTANTIATE_TBSV_TEMPLATE(double const* const*, double* const*)
INSTANTIATE_TBSV_TEMPLATE(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_TBSV_TEMPLATE(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TBSV_TEMPLATE

#ifdef INSTANTIATE_TBSV_NUMERICS
#error INSTANTIATE_TBSV_NUMERICS already defined
#endif

#define INSTANTIATE_TBSV_NUMERICS(T_, U_)                                 \
template rocblas_status rocblas_tbsv_check_numerics<T_, U_>               \
                                          (const char*    function_name,  \
                                           rocblas_handle handle,         \
                                           int64_t    n,              \
                                           T_             A,              \
                                           rocblas_stride    offset_A,       \
                                           int64_t    lda,            \
                                           rocblas_stride stride_A,       \
                                           U_             x,              \
                                           rocblas_stride    offset_x,       \
                                           int64_t    incx,          \
                                           rocblas_stride stride_x,       \
                                           int64_t    batch_count,    \
                                           const int      check_numerics, \
                                           bool           is_input);

INSTANTIATE_TBSV_NUMERICS(float const*, float*)
INSTANTIATE_TBSV_NUMERICS(double const*, double*)
INSTANTIATE_TBSV_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_TBSV_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_TBSV_NUMERICS(float const* const*, float* const*)
INSTANTIATE_TBSV_NUMERICS(double const* const*, double* const*)
INSTANTIATE_TBSV_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_TBSV_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TBSV_NUMERICS

// clang-format on
