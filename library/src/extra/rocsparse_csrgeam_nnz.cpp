/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"
#include "control.h"
#include "internal/extra/rocsparse_csrgeam.h"
#include "rocsparse_common.h"
#include "rocsparse_csrgeam.hpp"
#include "rocsparse_primitives.h"
#include "utility.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgeam_index_base(rocsparse_int* nnz)
    {
        --(*nnz);
    }

    // Compute non-zero entries per row, where each row is processed by a wavefront.
    // Splitting row into several chunks such that we can use shared memory to store whether
    // a column index is populated or not.
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgeam_nnz_multipass_device(rocsparse_int m,
                                      rocsparse_int n,
                                      const rocsparse_int* __restrict__ csr_row_ptr_A,
                                      const rocsparse_int* __restrict__ csr_col_ind_A,
                                      const rocsparse_int* __restrict__ csr_row_ptr_B,
                                      const rocsparse_int* __restrict__ csr_col_ind_B,
                                      rocsparse_int* __restrict__ row_nnz,
                                      rocsparse_index_base idx_base_A,
                                      rocsparse_index_base idx_base_B)
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

        // Row nnz marker
        __shared__ bool stable[BLOCKSIZE];
        bool*           table = &stable[wid * WFSIZE];

        // Get row entry and exit point of A
        rocsparse_int row_begin_A = csr_row_ptr_A[row] - idx_base_A;
        rocsparse_int row_end_A   = csr_row_ptr_A[row + 1] - idx_base_A;

        // Get row entry and exit point of B
        rocsparse_int row_begin_B = csr_row_ptr_B[row] - idx_base_B;
        rocsparse_int row_end_B   = csr_row_ptr_B[row + 1] - idx_base_B;

        // Load the first column of the current row from A and B to set the starting
        // point for the first chunk
        rocsparse_int col_A
            = (row_begin_A < row_end_A) ? csr_col_ind_A[row_begin_A] - idx_base_A : n;
        rocsparse_int col_B
            = (row_begin_B < row_end_B) ? csr_col_ind_B[row_begin_B] - idx_base_B : n;

        // Begin of the current row chunk
        rocsparse_int chunk_begin = rocsparse::min(col_A, col_B);

        // Initialize the row nnz for the full (wavefront-wide) row
        rocsparse_int nnz = 0;

        // Initialize the index for column access into A and B
        row_begin_A += lid;
        row_begin_B += lid;

        // Loop over the chunks until the end of both rows (A and B) has been reached (which
        // is the number of total columns n)
        while(true)
        {
            // Initialize row nnz table
            table[lid] = false;

            __threadfence_block();

            // Initialize the beginning of the next chunk
            rocsparse_int min_col = n;

            // Loop over all columns of A, starting with the first entry that did not fit
            // into the previous chunk
            for(; row_begin_A < row_end_A; row_begin_A += WFSIZE)
            {
                // Get the column of A
                rocsparse_int col_A = csr_col_ind_A[row_begin_A] - idx_base_A;

                // Get the column of A shifted by the chunk_begin
                rocsparse_int shf_A = col_A - chunk_begin;

                // Check if this column of A is within the chunk
                if(shf_A < WFSIZE)
                {
                    // Mark this column in shared memory
                    table[shf_A] = true;
                }
                else
                {
                    // Store the first column index of A that exceeds the current chunk
                    min_col = rocsparse::min(min_col, col_A);
                    break;
                }
            }

            // Loop over all columns of B, starting with the first entry that did not fit
            // into the previous chunk
            for(; row_begin_B < row_end_B; row_begin_B += WFSIZE)
            {
                // Get the column of B
                rocsparse_int col_B = csr_col_ind_B[row_begin_B] - idx_base_B;

                // Get the column of B shifted by the chunk_begin
                rocsparse_int shf_B = col_B - chunk_begin;

                // Check if this column of B is within the chunk
                if(shf_B < WFSIZE)
                {
                    // Mark this column in shared memory
                    table[shf_B] = true;
                }
                else
                {
                    // Store the first column index of B that exceeds the current chunk
                    min_col = rocsparse::min(min_col, col_B);
                    break;
                }
            }

            __threadfence_block();

            // Compute the chunk's number of non-zeros of the row and add it to the global
            // row nnz counter
            nnz += __popcll(__ballot(table[lid]));

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

        // Last thread in each wavefront writes the accumulated total row nnz to global
        // memory
        if(lid == WFSIZE - 1)
        {
            row_nnz[row] = nnz;
        }
    }
}

rocsparse_status rocsparse::csrgeam_nnz_core(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const rocsparse_mat_descr descr_A,
                                             rocsparse_int             nnz_A,
                                             const rocsparse_int*      csr_row_ptr_A,
                                             const rocsparse_int*      csr_col_ind_A,
                                             const rocsparse_mat_descr descr_B,
                                             rocsparse_int             nnz_B,
                                             const rocsparse_int*      csr_row_ptr_B,
                                             const rocsparse_int*      csr_col_ind_B,
                                             const rocsparse_mat_descr descr_C,
                                             rocsparse_int*            csr_row_ptr_C,
                                             rocsparse_int*            nnz_C)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Stream
    hipStream_t stream = handle->stream;

#define CSRGEAM_DIM 256
    if(handle->wavefront_size == 32)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgeam_nnz_multipass_device<CSRGEAM_DIM, 32>),
            dim3((m - 1) / (CSRGEAM_DIM / 32) + 1),
            dim3(CSRGEAM_DIM),
            0,
            stream,
            m,
            n,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_C,
            descr_A->base,
            descr_B->base);
    }
    else
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgeam_nnz_multipass_device<CSRGEAM_DIM, 64>),
            dim3((m - 1) / (CSRGEAM_DIM / 64) + 1),
            dim3(CSRGEAM_DIM),
            0,
            stream,
            m,
            n,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_C,
            descr_A->base,
            descr_B->base);
    }
#undef CSRGEAM_DIM

    // Exclusive sum to obtain row pointers of C
    size_t rocprim_size;
    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::primitives::exclusive_scan_buffer_size<rocsparse_int, rocsparse_int>(
            handle, static_cast<rocsparse_int>(descr_C->base), m + 1, &rocprim_size)));

    bool  rocprim_alloc;
    void* rocprim_buffer;

    if(handle->buffer_size >= rocprim_size)
    {
        rocprim_buffer = handle->buffer;
        rocprim_alloc  = false;
    }
    else
    {
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&rocprim_buffer, rocprim_size, handle->stream));
        rocprim_alloc = true;
    }

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::primitives::exclusive_scan(handle,
                                              csr_row_ptr_C,
                                              csr_row_ptr_C,
                                              static_cast<rocsparse_int>(descr_C->base),
                                              m + 1,
                                              rocprim_size,
                                              rocprim_buffer));

    if(rocprim_alloc == true)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(rocprim_buffer, handle->stream));
    }

    // Extract the number of non-zero elements of C
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        // Blocking mode
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(nnz_C,
                                           csr_row_ptr_C + m,
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        // Adjust index base of nnz_C
        *nnz_C -= descr_C->base;
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            nnz_C, csr_row_ptr_C + m, sizeof(rocsparse_int), hipMemcpyDeviceToDevice, stream));

        // Adjust index base of nnz_C
        if(descr_C->base == rocsparse_index_base_one)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgeam_index_base<1>), dim3(1), dim3(1), 0, stream, nnz_C);
        }
    }

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrgeam_nnz_quickreturn(rocsparse_handle          handle,
                                                    rocsparse_int             m,
                                                    rocsparse_int             n,
                                                    const rocsparse_mat_descr descr_A,
                                                    rocsparse_int             nnz_A,
                                                    const rocsparse_int*      csr_row_ptr_A,
                                                    const rocsparse_int*      csr_col_ind_A,
                                                    const rocsparse_mat_descr descr_B,
                                                    rocsparse_int             nnz_B,
                                                    const rocsparse_int*      csr_row_ptr_B,
                                                    const rocsparse_int*      csr_col_ind_B,
                                                    const rocsparse_mat_descr descr_C,
                                                    rocsparse_int*            csr_row_ptr_C,
                                                    rocsparse_int*            nnz_C)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Quick return if possible
    if(m == 0 || n == 0 || (nnz_A == 0 && nnz_B == 0))
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_host)
        {
            *nnz_C = 0;
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(rocsparse_int), handle->stream));
        }

        if(nnz_A == 0 && nnz_B == 0)
        {
            if(csr_row_ptr_C != nullptr)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset(
                    handle, m + 1, static_cast<rocsparse_int>(descr_C->base), csr_row_ptr_C));
            }
        }
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

namespace rocsparse
{
    static rocsparse_status csrgeam_nnz_checkarg(rocsparse_handle          handle, //0
                                                 rocsparse_int             m, //1
                                                 rocsparse_int             n, //2
                                                 const rocsparse_mat_descr descr_A, //3
                                                 rocsparse_int             nnz_A, //4
                                                 const rocsparse_int*      csr_row_ptr_A, //5
                                                 const rocsparse_int*      csr_col_ind_A, //6
                                                 const rocsparse_mat_descr descr_B, //7
                                                 rocsparse_int             nnz_B, //8
                                                 const rocsparse_int*      csr_row_ptr_B, //9
                                                 const rocsparse_int*      csr_col_ind_B, //10
                                                 const rocsparse_mat_descr descr_C, //11
                                                 rocsparse_int*            csr_row_ptr_C, //12
                                                 rocsparse_int*            nnz_C) //13
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(3, descr_A);
        ROCSPARSE_CHECKARG_POINTER(7, descr_B);
        ROCSPARSE_CHECKARG_POINTER(11, descr_C);

        ROCSPARSE_CHECKARG(3,
                           descr_A,
                           (descr_A->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(7,
                           descr_B,
                           (descr_B->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(11,
                           descr_C,
                           (descr_C->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(3,
                           descr_A,
                           (descr_A->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(7,
                           descr_B,
                           (descr_B->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(11,
                           descr_C,
                           (descr_C->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, n);
        ROCSPARSE_CHECKARG_SIZE(4, nnz_A);
        ROCSPARSE_CHECKARG_SIZE(8, nnz_B);

        ROCSPARSE_CHECKARG_POINTER(13, nnz_C);

        const rocsparse_status status = rocsparse::csrgeam_nnz_quickreturn(handle,
                                                                           m,
                                                                           n,
                                                                           descr_A,
                                                                           nnz_A,
                                                                           csr_row_ptr_A,
                                                                           csr_col_ind_A,
                                                                           descr_B,
                                                                           nnz_B,
                                                                           csr_row_ptr_B,
                                                                           csr_col_ind_B,
                                                                           descr_C,
                                                                           csr_row_ptr_C,
                                                                           nnz_C);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_ARRAY(12, m, csr_row_ptr_C);
        ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr_A);
        ROCSPARSE_CHECKARG_ARRAY(9, m, csr_row_ptr_B);
        ROCSPARSE_CHECKARG_ARRAY(6, nnz_A, csr_col_ind_A);
        ROCSPARSE_CHECKARG_ARRAY(10, nnz_B, csr_col_ind_B);

        return rocsparse_status_continue;
    }

    template <typename... P>
    static rocsparse_status csrgeam_nnz_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace("rocsparse_csrgeam_nnz", p...);

        const rocsparse_status status = rocsparse::csrgeam_nnz_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_nnz_core(p...));
        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status rocsparse_csrgeam_nnz(rocsparse_handle          handle,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  const rocsparse_mat_descr descr_A,
                                                  rocsparse_int             nnz_A,
                                                  const rocsparse_int*      csr_row_ptr_A,
                                                  const rocsparse_int*      csr_col_ind_A,
                                                  const rocsparse_mat_descr descr_B,
                                                  rocsparse_int             nnz_B,
                                                  const rocsparse_int*      csr_row_ptr_B,
                                                  const rocsparse_int*      csr_col_ind_B,
                                                  const rocsparse_mat_descr descr_C,
                                                  rocsparse_int*            csr_row_ptr_C,
                                                  rocsparse_int*            nnz_C)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_nnz_impl(handle,
                                                          m,
                                                          n,
                                                          descr_A,
                                                          nnz_A,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          descr_B,
                                                          nnz_B,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          descr_C,
                                                          csr_row_ptr_C,
                                                          nnz_C));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
