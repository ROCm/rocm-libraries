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

    //     1 4 0 0 0 0
    // A = 0 2 3 0 0 0
    //     5 0 0 7 8 0
    //     0 0 9 0 6 0

    rocsparse_int m         = 4;
    rocsparse_int n         = 6;
    rocsparse_int block_dim = 2;
    rocsparse_int nnz       = 9;
    rocsparse_int mb        = (m + block_dim - 1) / block_dim;
    rocsparse_int nb        = (n + block_dim - 1) / block_dim;

    // Define host arrays
    rocsparse_int h_csr_row_ptr[] = {0, 2, 4, 7, 9};
    rocsparse_int h_csr_col_ind[] = {0, 1, 1, 2, 0, 3, 4, 2, 4};
    float         h_csr_val[]     = {1, 4, 2, 3, 5, 7, 8, 9, 6};

    // Allocate and initialize device memory
    rocsparse_int* d_csr_row_ptr;
    rocsparse_int* d_csr_col_ind;
    float*         d_csr_val;

    HIP_CHECK(hipMalloc((void**)&d_csr_row_ptr, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&d_csr_col_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&d_csr_val, sizeof(float) * nnz));

    HIP_CHECK(hipMemcpy(
        d_csr_row_ptr, h_csr_row_ptr, sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_csr_col_ind, h_csr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_val, h_csr_val, sizeof(float) * nnz, hipMemcpyHostToDevice));

    // Create matrix descriptors
    rocsparse_mat_descr csr_descr;
    rocsparse_mat_descr bsr_descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&csr_descr));
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&bsr_descr));

    // Allocate BSR row pointer array
    rocsparse_int* d_bsr_row_ptr;
    HIP_CHECK(hipMalloc((void**)&d_bsr_row_ptr, sizeof(rocsparse_int) * (mb + 1)));

    // Compute the number of non-zero block entries
    rocsparse_int  nnzb;
    rocsparse_int* nnzTotalDevHostPtr = &nnzb;
    ROCSPARSE_CHECK(rocsparse_csr2bsr_nnz(handle,
                                          rocsparse_direction_row,
                                          m,
                                          n,
                                          csr_descr,
                                          d_csr_row_ptr,
                                          d_csr_col_ind,
                                          block_dim,
                                          bsr_descr,
                                          d_bsr_row_ptr,
                                          nnzTotalDevHostPtr));

    // Allocate BSR column indices and values arrays
    rocsparse_int* d_bsr_col_ind;
    float*         d_bsr_val;
    HIP_CHECK(hipMalloc((void**)&d_bsr_col_ind, sizeof(rocsparse_int) * nnzb));
    HIP_CHECK(hipMalloc((void**)&d_bsr_val, sizeof(float) * (block_dim * block_dim) * nnzb));

    // Convert CSR to BSR
    ROCSPARSE_CHECK(rocsparse_scsr2bsr(handle,
                                       rocsparse_direction_row,
                                       m,
                                       n,
                                       csr_descr,
                                       d_csr_val,
                                       d_csr_row_ptr,
                                       d_csr_col_ind,
                                       block_dim,
                                       bsr_descr,
                                       d_bsr_val,
                                       d_bsr_row_ptr,
                                       d_bsr_col_ind));

    // Clean up
    HIP_CHECK(hipFree(d_csr_row_ptr));
    HIP_CHECK(hipFree(d_csr_col_ind));
    HIP_CHECK(hipFree(d_csr_val));
    HIP_CHECK(hipFree(d_bsr_row_ptr));
    HIP_CHECK(hipFree(d_bsr_col_ind));
    HIP_CHECK(hipFree(d_bsr_val));

    // Destroy matrix descriptors and rocSPARSE handle
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(csr_descr));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(bsr_descr));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
//! [doc example]
