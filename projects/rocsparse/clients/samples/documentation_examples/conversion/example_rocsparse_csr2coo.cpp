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

    rocsparse_int m   = 3;
    rocsparse_int n   = 5;
    rocsparse_int nnz = 8;

    // Define host arrays
    rocsparse_int h_csr_row_ptr[] = {0, 3, 5, 8};
    rocsparse_int h_csr_col_ind[] = {0, 1, 3, 1, 2, 0, 3, 4};
    float         h_csr_val[]     = {1, 2, 3, 4, 5, 6, 7, 8};

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

    // Allocate COO matrix arrays
    rocsparse_int* d_coo_row_ind;
    rocsparse_int* d_coo_col_ind;
    float*         d_coo_val;

    HIP_CHECK(hipMalloc((void**)&d_coo_row_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&d_coo_col_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&d_coo_val, sizeof(float) * nnz));

    // Convert the CSR row offsets into COO row indices
    ROCSPARSE_CHECK(
        rocsparse_csr2coo(handle, d_csr_row_ptr, nnz, m, d_coo_row_ind, rocsparse_index_base_zero));

    // Copy the column and value arrays
    HIP_CHECK(hipMemcpy(
        d_coo_col_ind, d_csr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(d_coo_val, d_csr_val, sizeof(float) * nnz, hipMemcpyDeviceToDevice));

    // Clean up
    HIP_CHECK(hipFree(d_csr_row_ptr));
    HIP_CHECK(hipFree(d_csr_col_ind));
    HIP_CHECK(hipFree(d_csr_val));
    HIP_CHECK(hipFree(d_coo_row_ind));
    HIP_CHECK(hipFree(d_coo_col_ind));
    HIP_CHECK(hipFree(d_coo_val));

    // Destroy rocSPARSE handle
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
//! [doc example]
