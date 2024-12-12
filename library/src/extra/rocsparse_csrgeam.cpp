/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "internal/extra/rocsparse_csrgeam.h"
#include "common.h"
#include "control.h"
#include "rocsparse_csrgeam.hpp"
#include "utility.h"

namespace rocsparse
{
    // Compute matrix addition, where each row is processed by a wavefront.
    // Splitting row into several chunks such that we can use shared memory to store whether
    // a column index is populated or not.
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename T>
    ROCSPARSE_DEVICE_ILF void
        csrgeam_fill_multipass_device(rocsparse_int m,
                                      rocsparse_int n,
                                      T             alpha,
                                      const rocsparse_int* __restrict__ csr_row_ptr_A,
                                      const rocsparse_int* __restrict__ csr_col_ind_A,
                                      const T* __restrict__ csr_val_A,
                                      T beta,
                                      const rocsparse_int* __restrict__ csr_row_ptr_B,
                                      const rocsparse_int* __restrict__ csr_col_ind_B,
                                      const T* __restrict__ csr_val_B,
                                      const rocsparse_int* __restrict__ csr_row_ptr_C,
                                      rocsparse_int* __restrict__ csr_col_ind_C,
                                      T* __restrict__ csr_val_C,
                                      rocsparse_index_base idx_base_A,
                                      rocsparse_index_base idx_base_B,
                                      rocsparse_index_base idx_base_C)
    {
        // Lane id
        rocsparse_int lid = hipThreadIdx_x & (WFSIZE - 1);

        // Wavefront id
        rocsparse_int wid = hipThreadIdx_x / WFSIZE;

        // Each wavefront processes a row
        rocsparse_int row = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= m)
        {
            return;
        }

        // Row entry marker and value accumulator
        __shared__ bool stable[BLOCKSIZE];
        __shared__ T    sdata[BLOCKSIZE];

        bool* table = &stable[wid * WFSIZE];
        T*    data  = &sdata[wid * WFSIZE];

        // Get row entry and exit point of A
        rocsparse_int row_begin_A = csr_row_ptr_A[row] - idx_base_A;
        rocsparse_int row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

        // Get row entry and exit point of B
        rocsparse_int row_begin_B = csr_row_ptr_B[row] - idx_base_B;
        rocsparse_int row_end_B   = csr_row_ptr_B[row + 1] - idx_base_B;

        // Get row entry point of C
        rocsparse_int row_begin_C = csr_row_ptr_C[row] - idx_base_C;

        // Load the first column of the current row from A and B to set the starting
        // point for the first chunk
        rocsparse_int col_A
            = (row_begin_A < row_end_A) ? csr_col_ind_A[row_begin_A] - idx_base_A : n;
        rocsparse_int col_B
            = (row_begin_B < row_end_B) ? csr_col_ind_B[row_begin_B] - idx_base_B : n;

        // Begin of the current row chunk
        rocsparse_int chunk_begin = rocsparse::min(col_A, col_B);

        // Initialize the index for column access into A and B
        row_begin_A += lid;
        row_begin_B += lid;

        // Loop over the chunks until the end of both rows (A and B) has been reached (which
        // is the number of total columns n)
        while(true)
        {
            // Initialize row nnz table and value accumulator
            table[lid] = false;
            data[lid]  = static_cast<T>(0);

            __threadfence_block();

            // Initialize the beginning of the next chunk
            rocsparse_int min_col = n;

            // Loop over all columns of A, starting with the first entry that did not fit
            // into the previous chunk
            for(; row_begin_A < row_end_A; row_begin_A += WFSIZE)
            {
                // Get the column of A
                rocsparse_int col = csr_col_ind_A[row_begin_A] - idx_base_A;

                // Get the column of A shifted by the chunk_begin
                rocsparse_int shf_A = col - chunk_begin;

                // Check if this column of A is within the chunk
                if(shf_A < WFSIZE)
                {
                    // Mark nnz
                    table[shf_A] = true;

                    // Initialize with value of A
                    data[shf_A] = alpha * csr_val_A[row_begin_A];
                }
                else
                {
                    // Store the first column index of A that exceeds the current chunk
                    min_col = rocsparse::min(min_col, col);
                    break;
                }
            }

            __threadfence_block();

            // Loop over all columns of B, starting with the first entry that did not fit
            // into the previous chunk
            for(; row_begin_B < row_end_B; row_begin_B += WFSIZE)
            {
                // Get the column of B
                rocsparse_int col = csr_col_ind_B[row_begin_B] - idx_base_B;

                // Get the column of B shifted by the chunk_begin
                rocsparse_int shf_B = col - chunk_begin;

                // Check if this column of B is within the chunk
                if(shf_B < WFSIZE)
                {
                    // Mark nnz
                    table[shf_B] = true;

                    // Add values of B
                    data[shf_B] = rocsparse::fma(beta, csr_val_B[row_begin_B], data[shf_B]);
                }
                else
                {
                    // Store the first column index of B that exceeds the current chunk
                    min_col = rocsparse::min(min_col, col);
                    break;
                }
            }

            __threadfence_block();

            // Each lane checks whether there is an non-zero entry to fill or not
            bool has_nnz = table[lid];

            // Obtain the bitmask that marks the position of each non-zero entry
            uint64_t mask = __ballot(has_nnz);

            // If the lane has an nnz assign, it must be filled into C
            if(has_nnz)
            {
                rocsparse_int offset;

                // Compute the lane's fill position in C
                if(WFSIZE == 32)
                {
                    offset = __popc(mask & (0xffffffff >> (WFSIZE - 1 - lid)));
                }
                else
                {
                    offset = __popcll(mask & (0xffffffffffffffff >> (WFSIZE - 1 - lid)));
                }

                // Fill C
                csr_col_ind_C[row_begin_C + offset - 1] = lid + chunk_begin + idx_base_C;
                csr_val_C[row_begin_C + offset - 1]     = data[lid];
            }

            // Shift the row entry to C by the number of total nnz of the current row
            row_begin_C += __popcll(mask);

            // Gather wavefront-wide minimum for the next chunks starting column index
            // Using shfl_xor here so that each thread in the wavefront obtains the final
            // result
            for(uint32_t i = WFSIZE >> 1; i > 0; i >>= 1)
            {
                min_col = rocsparse::min(min_col, __shfl_xor(min_col, i));
            }

            // Each thread sets the new chunk beginning
            chunk_begin = min_col;

            // Once the chunk beginning has reached the total number of columns n,
            // we are done
            if(chunk_begin >= n)
            {
                break;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgeam_fill_multipass_kernel(rocsparse_int m,
                                       rocsparse_int n,
                                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                       const rocsparse_int* __restrict__ csr_row_ptr_A,
                                       const rocsparse_int* __restrict__ csr_col_ind_A,
                                       const T* __restrict__ csr_val_A,
                                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                       const rocsparse_int* __restrict__ csr_row_ptr_B,
                                       const rocsparse_int* __restrict__ csr_col_ind_B,
                                       const T* __restrict__ csr_val_B,
                                       const rocsparse_int* __restrict__ csr_row_ptr_C,
                                       rocsparse_int* __restrict__ csr_col_ind_C,
                                       T* __restrict__ csr_val_C,
                                       rocsparse_index_base idx_base_A,
                                       rocsparse_index_base idx_base_B,
                                       rocsparse_index_base idx_base_C,
                                       bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        rocsparse::csrgeam_fill_multipass_device<BLOCKSIZE, WFSIZE>(m,
                                                                    n,
                                                                    alpha,
                                                                    csr_row_ptr_A,
                                                                    csr_col_ind_A,
                                                                    csr_val_A,
                                                                    beta,
                                                                    csr_row_ptr_B,
                                                                    csr_col_ind_B,
                                                                    csr_val_B,
                                                                    csr_row_ptr_C,
                                                                    csr_col_ind_C,
                                                                    csr_val_C,
                                                                    idx_base_A,
                                                                    idx_base_B,
                                                                    idx_base_C);
    }

    template <typename T>
    static rocsparse_status csrgeam_dispatch(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const T*                  alpha_device_host,
                                             const rocsparse_mat_descr descr_A,
                                             rocsparse_int             nnz_A,
                                             const T*                  csr_val_A,
                                             const rocsparse_int*      csr_row_ptr_A,
                                             const rocsparse_int*      csr_col_ind_A,
                                             const T*                  beta_device_host,
                                             const rocsparse_mat_descr descr_B,
                                             rocsparse_int             nnz_B,
                                             const T*                  csr_val_B,
                                             const rocsparse_int*      csr_row_ptr_B,
                                             const rocsparse_int*      csr_col_ind_B,
                                             const rocsparse_mat_descr descr_C,
                                             T*                        csr_val_C,
                                             const rocsparse_int*      csr_row_ptr_C,
                                             rocsparse_int*            csr_col_ind_C)
    {
        // Stream
        hipStream_t stream = handle->stream;

        // Pointer mode device
#define CSRGEAM_DIM 256
        if(handle->wavefront_size == 32)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgeam_fill_multipass_kernel<CSRGEAM_DIM, 32>),
                dim3((m - 1) / (CSRGEAM_DIM / 32) + 1),
                dim3(CSRGEAM_DIM),
                0,
                stream,
                m,
                n,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                csr_row_ptr_A,
                csr_col_ind_A,
                csr_val_A,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                csr_row_ptr_B,
                csr_col_ind_B,
                csr_val_B,
                csr_row_ptr_C,
                csr_col_ind_C,
                csr_val_C,
                descr_A->base,
                descr_B->base,
                descr_C->base,
                handle->pointer_mode == rocsparse_pointer_mode_host);
        }
        else
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgeam_fill_multipass_kernel<CSRGEAM_DIM, 64>),
                dim3((m - 1) / (CSRGEAM_DIM / 64) + 1),
                dim3(CSRGEAM_DIM),
                0,
                stream,
                m,
                n,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                csr_row_ptr_A,
                csr_col_ind_A,
                csr_val_A,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                csr_row_ptr_B,
                csr_col_ind_B,
                csr_val_B,
                csr_row_ptr_C,
                csr_col_ind_C,
                csr_val_C,
                descr_A->base,
                descr_B->base,
                descr_C->base,
                handle->pointer_mode == rocsparse_pointer_mode_host);
        }

#undef CSRGEAM_DIM

        return rocsparse_status_success;
    }
}

template <typename T>
rocsparse_status rocsparse::csrgeam_core(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             n,
                                         const T*                  alpha,
                                         const rocsparse_mat_descr descr_A,
                                         rocsparse_int             nnz_A,
                                         const T*                  csr_val_A,
                                         const rocsparse_int*      csr_row_ptr_A,
                                         const rocsparse_int*      csr_col_ind_A,
                                         const T*                  beta,
                                         const rocsparse_mat_descr descr_B,
                                         rocsparse_int             nnz_B,
                                         const T*                  csr_val_B,
                                         const rocsparse_int*      csr_row_ptr_B,
                                         const rocsparse_int*      csr_col_ind_B,
                                         const rocsparse_mat_descr descr_C,
                                         T*                        csr_val_C,
                                         const rocsparse_int*      csr_row_ptr_C,
                                         rocsparse_int*            csr_col_ind_C)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_dispatch(handle,
                                                          m,
                                                          n,
                                                          alpha,
                                                          descr_A,
                                                          nnz_A,
                                                          csr_val_A,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          beta,
                                                          descr_B,
                                                          nnz_B,
                                                          csr_val_B,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          descr_C,
                                                          csr_val_C,
                                                          csr_row_ptr_C,
                                                          csr_col_ind_C));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrgeam_quickreturn(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             n,
                                                const void*               alpha,
                                                const rocsparse_mat_descr descr_A,
                                                rocsparse_int             nnz_A,
                                                const void*               csr_val_A,
                                                const rocsparse_int*      csr_row_ptr_A,
                                                const rocsparse_int*      csr_col_ind_A,
                                                const void*               beta,
                                                const rocsparse_mat_descr descr_B,
                                                rocsparse_int             nnz_B,
                                                const void*               csr_val_B,
                                                const rocsparse_int*      csr_row_ptr_B,
                                                const rocsparse_int*      csr_col_ind_B,
                                                const rocsparse_mat_descr descr_C,
                                                void*                     csr_val_C,
                                                const rocsparse_int*      csr_row_ptr_C,
                                                rocsparse_int*            csr_col_ind_C)
{

    if(m == 0 || n == 0 || (nnz_A == 0 && nnz_B == 0))
    {
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

namespace rocsparse
{
    static rocsparse_status csrgeam_checkarg(rocsparse_handle          handle, //0
                                             rocsparse_int             m, //1
                                             rocsparse_int             n, //2
                                             const void*               alpha, //3
                                             const rocsparse_mat_descr descr_A, //4
                                             rocsparse_int             nnz_A, //5
                                             const void*               csr_val_A, //6
                                             const rocsparse_int*      csr_row_ptr_A, //7
                                             const rocsparse_int*      csr_col_ind_A, //8
                                             const void*               beta, //9
                                             const rocsparse_mat_descr descr_B, //10
                                             rocsparse_int             nnz_B, //11
                                             const void*               csr_val_B, //12
                                             const rocsparse_int*      csr_row_ptr_B, //13
                                             const rocsparse_int*      csr_col_ind_B, //14
                                             const rocsparse_mat_descr descr_C, //15
                                             void*                     csr_val_C, //16
                                             const rocsparse_int*      csr_row_ptr_C, //17
                                             rocsparse_int*            csr_col_ind_C) //18
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(3, alpha);
        ROCSPARSE_CHECKARG_POINTER(9, beta);
        ROCSPARSE_CHECKARG_POINTER(4, descr_A);
        ROCSPARSE_CHECKARG_POINTER(10, descr_B);
        ROCSPARSE_CHECKARG_POINTER(15, descr_C);

        ROCSPARSE_CHECKARG(4,
                           descr_A,
                           (descr_A->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(10,
                           descr_B,
                           (descr_B->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(15,
                           descr_C,
                           (descr_C->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(4,
                           descr_A,
                           (descr_A->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(10,
                           descr_B,
                           (descr_B->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(15,
                           descr_C,
                           (descr_C->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, n);
        ROCSPARSE_CHECKARG_SIZE(5, nnz_A);
        ROCSPARSE_CHECKARG_SIZE(11, nnz_B);

        const rocsparse_status status = rocsparse::csrgeam_quickreturn(handle,
                                                                       m,
                                                                       n,
                                                                       alpha,
                                                                       descr_A,
                                                                       nnz_A,
                                                                       csr_val_A,
                                                                       csr_row_ptr_A,
                                                                       csr_col_ind_A,
                                                                       beta,
                                                                       descr_B,
                                                                       nnz_B,
                                                                       csr_val_B,
                                                                       csr_row_ptr_B,
                                                                       csr_col_ind_B,
                                                                       descr_C,
                                                                       csr_val_C,
                                                                       csr_row_ptr_C,
                                                                       csr_col_ind_C);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_ARRAY(7, m, csr_row_ptr_A);
        ROCSPARSE_CHECKARG_ARRAY(13, m, csr_row_ptr_B);
        ROCSPARSE_CHECKARG_ARRAY(17, m, csr_row_ptr_C);

        ROCSPARSE_CHECKARG_ARRAY(6, nnz_A, csr_val_A);
        ROCSPARSE_CHECKARG_ARRAY(8, nnz_A, csr_col_ind_A);

        ROCSPARSE_CHECKARG_ARRAY(12, nnz_B, csr_val_B);
        ROCSPARSE_CHECKARG_ARRAY(14, nnz_B, csr_col_ind_B);

        if(csr_col_ind_C == nullptr || csr_val_C == nullptr)
        {
            rocsparse_int start = 0;
            rocsparse_int end   = 0;
            if(csr_row_ptr_C != nullptr)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                                   &csr_row_ptr_C[m],
                                                   sizeof(rocsparse_int),
                                                   hipMemcpyDeviceToHost,
                                                   handle->stream));
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                                   &csr_row_ptr_C[0],
                                                   sizeof(rocsparse_int),
                                                   hipMemcpyDeviceToHost,
                                                   handle->stream));
                RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
            }
            const rocsparse_int nnz_C = (end - start);
            ROCSPARSE_CHECKARG_ARRAY(16, nnz_C, csr_val_C);
            ROCSPARSE_CHECKARG_ARRAY(18, nnz_C, csr_col_ind_C);
        }

        return rocsparse_status_continue;
    }

    template <typename... P>
    static rocsparse_status csrgeam_impl(P&&... p)
    {
        rocsparse::log_trace("rocsparse_Xcsrgeam", p...);

        const rocsparse_status status = rocsparse::csrgeam_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_core(p...));
        return rocsparse_status_success;
    }
}

#define C_IMPL(NAME, TYPE)                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,        \
                                     rocsparse_int             m,             \
                                     rocsparse_int             n,             \
                                     const TYPE*               alpha,         \
                                     const rocsparse_mat_descr descr_A,       \
                                     rocsparse_int             nnz_A,         \
                                     const TYPE*               csr_val_A,     \
                                     const rocsparse_int*      csr_row_ptr_A, \
                                     const rocsparse_int*      csr_col_ind_A, \
                                     const TYPE*               beta,          \
                                     const rocsparse_mat_descr descr_B,       \
                                     rocsparse_int             nnz_B,         \
                                     const TYPE*               csr_val_B,     \
                                     const rocsparse_int*      csr_row_ptr_B, \
                                     const rocsparse_int*      csr_col_ind_B, \
                                     const rocsparse_mat_descr descr_C,       \
                                     TYPE*                     csr_val_C,     \
                                     const rocsparse_int*      csr_row_ptr_C, \
                                     rocsparse_int*            csr_col_ind_C) \
    try                                                                       \
    {                                                                         \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_impl(handle,             \
                                                          m,                  \
                                                          n,                  \
                                                          alpha,              \
                                                          descr_A,            \
                                                          nnz_A,              \
                                                          csr_val_A,          \
                                                          csr_row_ptr_A,      \
                                                          csr_col_ind_A,      \
                                                          beta,               \
                                                          descr_B,            \
                                                          nnz_B,              \
                                                          csr_val_B,          \
                                                          csr_row_ptr_B,      \
                                                          csr_col_ind_B,      \
                                                          descr_C,            \
                                                          csr_val_C,          \
                                                          csr_row_ptr_C,      \
                                                          csr_col_ind_C));    \
        return rocsparse_status_success;                                      \
    }                                                                         \
    catch(...)                                                                \
    {                                                                         \
        RETURN_ROCSPARSE_EXCEPTION();                                         \
    }

C_IMPL(rocsparse_scsrgeam, float);
C_IMPL(rocsparse_dcsrgeam, double);
C_IMPL(rocsparse_ccsrgeam, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrgeam, rocsparse_double_complex);

#undef C_IMPL
