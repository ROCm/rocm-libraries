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

    //     1 2 3
    // A = 4 5 6
    //     7 8 9
    rocsparse_int m   = 3;
    rocsparse_int n   = 3;
    rocsparse_int nnz = 9;

    // Define host arrays
    rocsparse_int h_csr_row_ptr[] = {0, 3, 6, 9};
    rocsparse_int h_csr_col_ind[] = {2, 0, 1, 0, 1, 2, 0, 2, 1};
    float         h_csr_val[]     = {3, 1, 2, 4, 5, 6, 7, 9, 8};

    // Allocate and initialize device memory for CSR matrix
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

    // Create matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Create permutation vector perm as the identity map
    rocsparse_int* perm;
    HIP_CHECK(hipMalloc((void**)&perm, sizeof(rocsparse_int) * nnz));
    ROCSPARSE_CHECK(rocsparse_create_identity_permutation(handle, nnz, perm));

    // Allocate temporary buffer
    size_t buffer_size;
    void*  temp_buffer;
    ROCSPARSE_CHECK(rocsparse_csrsort_buffer_size(
        handle, m, n, nnz, d_csr_row_ptr, d_csr_col_ind, &buffer_size));
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Sort the CSR matrix
    ROCSPARSE_CHECK(rocsparse_csrsort(
        handle, m, n, nnz, descr, d_csr_row_ptr, d_csr_col_ind, perm, temp_buffer));

    // Gather sorted csr_val array
    float* d_csr_val_sorted;
    HIP_CHECK(hipMalloc((void**)&d_csr_val_sorted, sizeof(float) * nnz));
    ROCSPARSE_CHECK(
        rocsparse_sgthr(handle, nnz, d_csr_val, d_csr_val_sorted, perm, rocsparse_index_base_zero));

    // Clean up
    HIP_CHECK(hipFree(temp_buffer));
    HIP_CHECK(hipFree(perm));
    HIP_CHECK(hipFree(d_csr_val));
    HIP_CHECK(hipFree(d_csr_val_sorted));
    HIP_CHECK(hipFree(d_csr_row_ptr));
    HIP_CHECK(hipFree(d_csr_col_ind));

    // Destroy matrix descriptor and rocSPARSE handle
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
//! [doc example]
