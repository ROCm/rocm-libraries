/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include <iostream>
#include <rocsparse/rocsparse.h>

#define HIP_CHECK(stat)                                                                       \
    {                                                                                         \
        if(stat != hipSuccess)                                                                \
        {                                                                                     \
            std::cerr << "Error: hip error " << stat << " in line " << __LINE__ << std::endl; \
            return -1;                                                                        \
        }                                                                                     \
    }

#define ROCSPARSE_CHECK(stat)                                                         \
    {                                                                                 \
        if(stat != rocsparse_status_success)                                          \
        {                                                                             \
            std::cerr << "Error: rocsparse error " << stat << " in line " << __LINE__ \
                      << std::endl;                                                   \
            return -1;                                                                \
        }                                                                             \
    }

//! [doc example]
int main()
{
    // Initialize rocSPARSE
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    //     1 2 0 3 0
    // A = 0 4 5 0 0
    //     6 0 0 7 8

    rocsparse_int m_A   = 3;
    rocsparse_int n_A   = 5;
    rocsparse_int nnz_A = 8;

    // Define host arrays
    rocsparse_int h_csr_row_ptr_A[] = {0, 3, 5, 8};
    rocsparse_int h_csr_col_ind_A[] = {0, 1, 3, 1, 2, 0, 3, 4};

    // Allocate and initialize device memory for matrix A
    rocsparse_int* d_csr_row_ptr_A;
    rocsparse_int* d_csr_col_ind_A;

    HIP_CHECK(hipMalloc((void**)&d_csr_row_ptr_A, sizeof(rocsparse_int) * (m_A + 1)));
    HIP_CHECK(hipMalloc((void**)&d_csr_col_ind_A, sizeof(rocsparse_int) * nnz_A));

    HIP_CHECK(hipMemcpy(d_csr_row_ptr_A,
                        h_csr_row_ptr_A,
                        sizeof(rocsparse_int) * (m_A + 1),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_csr_col_ind_A, h_csr_col_ind_A, sizeof(rocsparse_int) * nnz_A, hipMemcpyHostToDevice));

    // Allocate memory for transposed CSR matrix
    rocsparse_int m_T   = n_A;
    rocsparse_int n_T   = m_A;
    rocsparse_int nnz_T = nnz_A;

    rocsparse_int* d_csr_row_ptr_T;
    rocsparse_int* d_csr_col_ind_T;

    HIP_CHECK(hipMalloc((void**)&d_csr_row_ptr_T, sizeof(rocsparse_int) * (m_T + 1)));
    HIP_CHECK(hipMalloc((void**)&d_csr_col_ind_T, sizeof(rocsparse_int) * nnz_T));

    // Obtain the temporary buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_csr2csc_buffer_size(handle,
                                                  m_A,
                                                  n_A,
                                                  nnz_A,
                                                  d_csr_row_ptr_A,
                                                  d_csr_col_ind_A,
                                                  rocsparse_action_symbolic,
                                                  &buffer_size));

    // Allocate temporary buffer
    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Perform the CSR to CSC conversion
    ROCSPARSE_CHECK(rocsparse_scsr2csc(handle,
                                       m_A,
                                       n_A,
                                       nnz_A,
                                       nullptr,
                                       d_csr_row_ptr_A,
                                       d_csr_col_ind_A,
                                       nullptr,
                                       d_csr_col_ind_T,
                                       d_csr_row_ptr_T,
                                       rocsparse_action_symbolic,
                                       rocsparse_index_base_zero,
                                       temp_buffer));

    // Clean up
    HIP_CHECK(hipFree(d_csr_row_ptr_A));
    HIP_CHECK(hipFree(d_csr_col_ind_A));
    HIP_CHECK(hipFree(d_csr_row_ptr_T));
    HIP_CHECK(hipFree(d_csr_col_ind_T));
    HIP_CHECK(hipFree(temp_buffer));

    // Destroy rocSPARSE handle
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
//! [doc example]
