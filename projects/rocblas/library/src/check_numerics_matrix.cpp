/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "int64_helpers.hpp"
#include "utility.hpp"

/**
  *
  * rocblas_check_numerics_ge_matrix_kernel(m, n, Aa, offset_a, lda, stride_a, abnormal)
  *
  * Info about rocblas_check_numerics_ge_matrix_kernel function:
  *
  *    It is the kernel function which checks a matrix for numerical abnormalities such as NaN/zero/Inf/denormal values and updates the rocblas_check_numerics_t structure.
  *    ge in rocblas_check_numerics_ge_matrix_kernel refers to general.
  *
  * Parameters   : m            : number of rows of matrix 'A'
  *                n            : number of columns of matrix 'A'
  *                Aa           : Pointer to the matrix which is under consideration for numerical abnormalities
  *                offset_a     : Offset of matrix 'Aa'
  *                lda          : specifies the leading dimension of matrix 'Aa'
  *                stride_a     : Specifies the pointer increment between one matrix 'A_i' and the next one (Aa_i+1) (where (Aa_i) is the i-th instance of the batch)
  *                abnormal     : Device pointer to the rocblas_check_numerics_t structure
  *
  * Return Value : Nothing --
  *
**/

template <int DIM_X, int DIM_Y, typename T>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_check_numerics_ge_matrix_kernel(rocblas_int               m,
                                        rocblas_int               n,
                                        T                         Aa,
                                        rocblas_stride            offset_a,
                                        int64_t                   lda,
                                        rocblas_stride            stride_a,
                                        rocblas_check_numerics_t* abnormal)
{
    rocblas_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    rocblas_int ty = blockIdx.y * blockDim.y + threadIdx.y;

    //Check every element of the A matrix for a NaN/zero/Inf/denormal value
    if(tx < m && ty < n)
    {
        auto* A = load_ptr_batch(Aa, blockIdx.z, offset_a, stride_a);

        int64_t tid   = tx + lda * ty;
        auto    value = A[tid];
        if(!abnormal->has_zero && rocblas_iszero(value))
            abnormal->has_zero = true;
        if(!abnormal->has_NaN && rocblas_isnan(value))
            abnormal->has_NaN = true;
        if(!abnormal->has_Inf && rocblas_isinf(value))
            abnormal->has_Inf = true;
        if(!abnormal->has_denorm && rocblas_isdenorm(value))
            abnormal->has_denorm = true;
    }
}

/**
  *
  * rocblas_check_numerics_sym_herm_tri_matrix_kernel(is_upper, n, Aa, offset_a, lda, stride_a, abnormal)
  *
  * Info about rocblas_check_numerics_sym_herm_tri_matrix_kernel function:
  *
  *    It is the kernel function which checks symmetric, hermitian and triangular matrices for numerical abnormalities such as NaN/zero/Inf/denormal values
  *    and updates the rocblas_check_numerics_t structure.
  *    sym_herm_tri in rocblas_check_numerics_sym_herm_tri_matrix_kernel refers to symmetric, hermitian and triangular matrices.
  *
  * Parameters   : is_upper     : Boolean which is true when the rocblas_fill is rocblas_fill_upper and false when it is rocblas_fill_lower
  *                n            : number of columns of matrix 'A'
  *                Aa           : Pointer to the matrix which is under consideration for numerical abnormalities
  *                offset_a     : Offset of matrix 'Aa'
  *                lda          : specifies the leading dimension of matrix 'Aa'
  *                stride_a     : Specifies the pointer increment between one matrix 'A_i' and the next one (Aa_i+1) (where (Aa_i) is the i-th instance of the batch)
  *                abnormal     : Device pointer to the rocblas_check_numerics_t structure
  *
  * Return Value : Nothing --
  *
**/

template <int DIM_X, int DIM_Y, typename T>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_check_numerics_sym_herm_tri_matrix_kernel(bool                      is_upper,
                                                  rocblas_int               n,
                                                  T                         Aa,
                                                  rocblas_stride            offset_a,
                                                  int64_t                   lda,
                                                  rocblas_stride            stride_a,
                                                  rocblas_check_numerics_t* abnormal)
{
    rocblas_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    rocblas_int ty = blockIdx.y * blockDim.y + threadIdx.y;

    //Check every element of the A matrix for a NaN/zero/Inf/denormal value
    if(is_upper ? ty < n && tx <= ty : tx < n && ty <= tx)
    {
        auto* A = load_ptr_batch(Aa, blockIdx.z, offset_a, stride_a);

        int64_t tid   = tx + lda * ty;
        auto    value = A[tid];
        if(!abnormal->has_zero && rocblas_iszero(value))
            abnormal->has_zero = true;
        if(!abnormal->has_NaN && rocblas_isnan(value))
            abnormal->has_NaN = true;
        if(!abnormal->has_Inf && rocblas_isinf(value))
            abnormal->has_Inf = true;
        if(!abnormal->has_denorm && rocblas_isdenorm(value))
            abnormal->has_denorm = true;
    }
}

/**
  *
  * rocblas_internal_check_numerics_matrix_template(function_name, handle, n, x, offset_x, inc_x, stride_x, batch_count, check_numerics, is_input)
  *
  * Info about rocblas_internal_check_numerics_matrix_template function:
  *
  *    It is the host function which accepts a matrix and calls the 'rocblas_check_numerics_ge_matrix_kernel' kernel function
  *    to check for numerical abnormalities such as NaN/zero/Inf/denormal in that matrix.
  *    It also helps in debugging based on the different types of flags in rocblas_check_numerics_mode that users set to debug potential NaN/zero/Inf/denormal value.
  *
  * Parameters   : function_name         : Name of the rocBLAS math function
  *                handle                : Handle to the rocblas library context queue
  *                m                     : number of rows of matrix 'A'
  *                n                     : number of columns of matrix 'A'
  *                A                     : Pointer to the matrix which is under check for numerical abnormalities
  *                offset_a              : Offset of matrix 'A'
  *                lda                   : specifies the leading dimension of matrix 'A'
  *                stride_a              : Specifies the pointer increment between one matrix 'A_i' and the next one (A_i+1) (where (A_i) is the i-th instance of the batch)
  *                check_numerics        : User defined flag for debugging
  *                is_input              : To check if the matrix under consideration is an Input or an Output matrix
  *
  * Return Value : rocblas_status
  *        rocblas_status_success        : Return status if the matrix does not have a NaN/Inf/denormal value
  *   rocblas_status_check_numerics_fail : Return status if the matrix contains a NaN/Inf/denormal value and 'check_numerics' enum is set to 'rocblas_check_numerics_mode_fail'
  *
**/

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_check_numerics_matrix_template(const char*               function_name,
                                                    rocblas_handle            handle,
                                                    rocblas_operation         trans_a,
                                                    rocblas_fill              uplo,
                                                    rocblas_check_matrix_type matrix_type,
                                                    int64_t                   m_64,
                                                    int64_t                   n_64,
                                                    T                         A,
                                                    rocblas_stride            offset_a,
                                                    int64_t                   lda,
                                                    rocblas_stride            stride_a,
                                                    int64_t                   batch_count_64,
                                                    const int                 check_numerics,
                                                    bool                      is_input)
{
    //Graph capture do not support any use of sync APIs.
    //Quick return: check numerics not supported
    if(handle->is_stream_in_capture_mode())
    {
        return rocblas_status_success;
    }

    //Quick return if possible. Not Argument error
    if(!m_64 || !n_64 || !batch_count_64 || !A)
        return rocblas_status_success;

    //Creating structure host object
    rocblas_check_numerics_t h_abnormal;

    //Allocating memory for device structure
    auto d_abnormal = handle->device_malloc(sizeof(rocblas_check_numerics_t));

    if(!d_abnormal)
    {
        rocblas_cerr << "rocBLAS internal error: No workspace memory available to allocate the "
                        "struct d_abnormal in "
                        "rocblas_check_numerics"
                     << std::endl;
        return rocblas_status_memory_error;
    }

    hipStream_t rocblas_stream = handle->get_stream();

    //Transferring the rocblas_check_numerics_t structure from host to the device
    RETURN_IF_HIP_ERROR(hipMemcpyAsync((rocblas_check_numerics_t*)d_abnormal,
                                       &h_abnormal,
                                       sizeof(rocblas_check_numerics_t),
                                       hipMemcpyHostToDevice,
                                       rocblas_stream));

    //Checking trans_a to transpose a matrix 'A'
    int64_t rows_64 = trans_a == rocblas_operation_none ? m_64 : n_64;
    int64_t cols_64 = trans_a == rocblas_operation_none ? n_64 : m_64;

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto    a_ptr       = adjust_ptr_batch(A, b_base, stride_a);
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        for(int64_t n_base = 0; n_base < cols_64; n_base += c_i64_grid_YZ_chunk)
        {
            int32_t n = int32_t(std::min(cols_64 - n_base, c_i64_grid_YZ_chunk));

            rocblas_stride col_offset = offset_a + n_base * lda;

            for(int64_t m_base = 0; m_base < rows_64; m_base += c_i64_grid_X_chunk)
            {
                int32_t m = int32_t(std::min(rows_64 - m_base, c_i64_grid_X_chunk));

                rocblas_stride shift_a = col_offset + m_base;

                static constexpr int DIM_X    = 16;
                static constexpr int DIM_Y    = 16;
                rocblas_int          blocks_X = (m - 1) / DIM_X + 1;
                rocblas_int          blocks_Y = (n - 1) / DIM_Y + 1;

                dim3 blocks(blocks_X, blocks_Y, batch_count);
                dim3 threads(DIM_X, DIM_Y);

                if(matrix_type == rocblas_client_general_matrix)
                {
                    ROCBLAS_LAUNCH_KERNEL((rocblas_check_numerics_ge_matrix_kernel<DIM_X, DIM_Y>),
                                          blocks,
                                          threads,
                                          0,
                                          rocblas_stream,
                                          m,
                                          n,
                                          a_ptr,
                                          shift_a,
                                          lda,
                                          stride_a,
                                          (rocblas_check_numerics_t*)d_abnormal);
                }
                else if(matrix_type == rocblas_client_symmetric_matrix
                        || matrix_type == rocblas_client_hermitian_matrix
                        || matrix_type == rocblas_client_triangular_matrix)
                {
                    ROCBLAS_LAUNCH_KERNEL(
                        (rocblas_check_numerics_sym_herm_tri_matrix_kernel<DIM_X, DIM_Y>),
                        blocks,
                        threads,
                        0,
                        rocblas_stream,
                        uplo == rocblas_fill_upper,
                        n,
                        a_ptr,
                        shift_a,
                        lda,
                        stride_a,
                        (rocblas_check_numerics_t*)d_abnormal);
                }
            }
        }
    }

    //Transferring the rocblas_check_numerics_t structure from device to the host
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&h_abnormal,
                                       (rocblas_check_numerics_t*)d_abnormal,
                                       sizeof(rocblas_check_numerics_t),
                                       hipMemcpyDeviceToHost,
                                       rocblas_stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(rocblas_stream));

    return rocblas_check_numerics_abnormal_struct(
        function_name, check_numerics, is_input, &h_abnormal);
}

// INSTANTIATIONS TO SUPPORT output: T*, T* const*, and input: const T*, const T* const*

#ifdef INST
#error INST IS ALREADY DEFINED
#endif
#define INST(typet_)                                                                              \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                      \
        rocblas_internal_check_numerics_matrix_template(const char*               function_name,  \
                                                        rocblas_handle            handle,         \
                                                        rocblas_operation         trans_A,        \
                                                        rocblas_fill              uplo,           \
                                                        rocblas_check_matrix_type matrix_type,    \
                                                        int64_t                   m,              \
                                                        int64_t                   n,              \
                                                        typet_                    A,              \
                                                        rocblas_stride            offset_a,       \
                                                        int64_t                   lda,            \
                                                        rocblas_stride            stride_a,       \
                                                        int64_t                   batch_count,    \
                                                        const int                 check_numerics, \
                                                        bool                      is_input)
// INST(int*);
// INST(int* const*);
// INST(int const*);
// INST(int const* const*);

INST(float*);
INST(float* const*);
INST(float const*);
INST(float const* const*);

INST(double*);
INST(double const*);
INST(double* const*);
INST(double const* const*);

INST(rocblas_float_complex*);
INST(rocblas_float_complex* const*);
INST(rocblas_float_complex const*);
INST(rocblas_float_complex const* const*);

INST(rocblas_double_complex*);
INST(rocblas_double_complex* const*);
INST(rocblas_double_complex const*);
INST(rocblas_double_complex const* const*);

INST(rocblas_half*);
INST(rocblas_half* const*);
INST(rocblas_half const*);
INST(rocblas_half const* const*);

INST(rocblas_bfloat16*);
INST(rocblas_bfloat16* const*);
INST(rocblas_bfloat16 const*);
INST(rocblas_bfloat16 const* const*);

#undef INST
