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

    float tol = 0.0f;

    rocsparse_int m     = 3;
    rocsparse_int n     = 5;
    rocsparse_int nnz_A = 8;

    // Define host arrays
    rocsparse_int h_csr_row_ptr_A[] = {0, 3, 5, 8};
    rocsparse_int h_csr_col_ind_A[] = {0, 1, 3, 1, 2, 0, 3, 4};
    float         h_csr_val_A[]     = {1, 0, 3, 4, 0, 6, 7, 0};

    // Allocate and initialize device memory for matrix A
    rocsparse_int* d_csr_row_ptr_A;
    rocsparse_int* d_csr_col_ind_A;
    float*         d_csr_val_A;

    HIP_CHECK(hipMalloc((void**)&d_csr_row_ptr_A, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&d_csr_col_ind_A, sizeof(rocsparse_int) * nnz_A));
    HIP_CHECK(hipMalloc((void**)&d_csr_val_A, sizeof(float) * nnz_A));

    HIP_CHECK(hipMemcpy(
        d_csr_row_ptr_A, h_csr_row_ptr_A, sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_csr_col_ind_A, h_csr_col_ind_A, sizeof(rocsparse_int) * nnz_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_val_A, h_csr_val_A, sizeof(float) * nnz_A, hipMemcpyHostToDevice));

    // Allocate memory for the row pointer array of the compressed CSR matrix
    rocsparse_int* d_csr_row_ptr_C;
    HIP_CHECK(hipMalloc((void**)&d_csr_row_ptr_C, sizeof(rocsparse_int) * (m + 1)));

    // Allocate memory for the nnz_per_row array
    rocsparse_int* d_nnz_per_row;
    HIP_CHECK(hipMalloc((void**)&d_nnz_per_row, sizeof(rocsparse_int) * m));

    rocsparse_mat_descr descr = nullptr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Call nnz_compress() which fills in nnz_per_row array and finds the number
    // of entries that will be in the compressed CSR matrix
    rocsparse_int nnz_C;
    ROCSPARSE_CHECK(rocsparse_snnz_compress(
        handle, m, descr, d_csr_val_A, d_csr_row_ptr_A, d_nnz_per_row, &nnz_C, tol));

    // Allocate column indices and values array for the compressed CSR matrix
    rocsparse_int* d_csr_col_ind_C;
    float*         d_csr_val_C;
    HIP_CHECK(hipMalloc((void**)&d_csr_col_ind_C, sizeof(rocsparse_int) * nnz_C));
    HIP_CHECK(hipMalloc((void**)&d_csr_val_C, sizeof(float) * nnz_C));

    // Finish compression by calling csr2csr_compress()
    ROCSPARSE_CHECK(rocsparse_scsr2csr_compress(handle,
                                                m,
                                                n,
                                                descr,
                                                d_csr_val_A,
                                                d_csr_row_ptr_A,
                                                d_csr_col_ind_A,
                                                nnz_A,
                                                d_nnz_per_row,
                                                d_csr_val_C,
                                                d_csr_row_ptr_C,
                                                d_csr_col_ind_C,
                                                tol));

    // Clean up
    HIP_CHECK(hipFree(d_csr_row_ptr_A));
    HIP_CHECK(hipFree(d_csr_col_ind_A));
    HIP_CHECK(hipFree(d_csr_val_A));
    HIP_CHECK(hipFree(d_csr_row_ptr_C));
    HIP_CHECK(hipFree(d_csr_col_ind_C));
    HIP_CHECK(hipFree(d_csr_val_C));
    HIP_CHECK(hipFree(d_nnz_per_row));

    // Destroy rocSPARSE handle
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));

    return 0;
}
//! [doc example]
