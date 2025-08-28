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
    // Initialize scalar multipliers
    float alpha = 2.0f;
    float beta  = 1.0f;

    // Define matrix dimensions
    rocsparse_int m = 3; // Number of rows
    rocsparse_int n = 3; // Number of columns
    rocsparse_int k = 3; // Inner dimension for multiplication

    // Create rocsparse handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create matrix descriptors
    rocsparse_mat_descr descr_A;
    rocsparse_mat_descr descr_B;
    rocsparse_mat_descr descr_C;
    rocsparse_mat_descr descr_D;

    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_A));
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_B));
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_C));
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_D));

    // Create matrix info structure
    rocsparse_mat_info info_C;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info_C));

    // Set pointer mode
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Define host arrays for CSR matrices A, B, and D
    rocsparse_int nnz_A = 4; // Number of non-zero entries in A
    rocsparse_int nnz_B = 4; // Number of non-zero entries in B
    rocsparse_int nnz_D = 4; // Number of non-zero entries in D

    rocsparse_int h_csr_row_ptr_A[] = {0, 2, 3, 4};
    rocsparse_int h_csr_col_ind_A[] = {0, 1, 2, 2};
    float         h_csr_val_A[]     = {1.0f, 2.0f, 3.0f, 4.0f};

    rocsparse_int h_csr_row_ptr_B[] = {0, 1, 3, 4};
    rocsparse_int h_csr_col_ind_B[] = {0, 1, 2, 2};
    float         h_csr_val_B[]     = {5.0f, 6.0f, 7.0f, 8.0f};

    rocsparse_int h_csr_row_ptr_D[] = {0, 1, 3, 4};
    rocsparse_int h_csr_col_ind_D[] = {0, 1, 2, 2};
    float         h_csr_val_D[]     = {9.0f, 10.0f, 11.0f, 12.0f};

    // Allocate device memory for CSR matrices A, B, and D
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

    rocsparse_int* d_csr_row_ptr_B;
    rocsparse_int* d_csr_col_ind_B;
    float*         d_csr_val_B;

    HIP_CHECK(hipMalloc((void**)&d_csr_row_ptr_B, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&d_csr_col_ind_B, sizeof(rocsparse_int) * nnz_B));
    HIP_CHECK(hipMalloc((void**)&d_csr_val_B, sizeof(float) * nnz_B));

    HIP_CHECK(hipMemcpy(
        d_csr_row_ptr_B, h_csr_row_ptr_B, sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_csr_col_ind_B, h_csr_col_ind_B, sizeof(rocsparse_int) * nnz_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_val_B, h_csr_val_B, sizeof(float) * nnz_B, hipMemcpyHostToDevice));

    rocsparse_int* d_csr_row_ptr_D;
    rocsparse_int* d_csr_col_ind_D;
    float*         d_csr_val_D;

    HIP_CHECK(hipMalloc((void**)&d_csr_row_ptr_D, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&d_csr_col_ind_D, sizeof(rocsparse_int) * nnz_D));
    HIP_CHECK(hipMalloc((void**)&d_csr_val_D, sizeof(float) * nnz_D));

    HIP_CHECK(hipMemcpy(
        d_csr_row_ptr_D, h_csr_row_ptr_D, sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_csr_col_ind_D, h_csr_col_ind_D, sizeof(rocsparse_int) * nnz_D, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csr_val_D, h_csr_val_D, sizeof(float) * nnz_D, hipMemcpyHostToDevice));

    // Query rocsparse for the required buffer size
    size_t buffer_size;

    ROCSPARSE_CHECK(rocsparse_scsrgemm_buffer_size(handle,
                                                   rocsparse_operation_none,
                                                   rocsparse_operation_none,
                                                   m,
                                                   n,
                                                   k,
                                                   &alpha,
                                                   descr_A,
                                                   nnz_A,
                                                   d_csr_row_ptr_A,
                                                   d_csr_col_ind_A,
                                                   descr_B,
                                                   nnz_B,
                                                   d_csr_row_ptr_B,
                                                   d_csr_col_ind_B,
                                                   &beta,
                                                   descr_D,
                                                   nnz_D,
                                                   d_csr_row_ptr_D,
                                                   d_csr_col_ind_D,
                                                   info_C,
                                                   &buffer_size));

    // Allocate buffer
    void* buffer;
    HIP_CHECK(hipMalloc(&buffer, buffer_size));

    // Obtain number of total non-zero entries in C and row pointers of C
    rocsparse_int  nnz_C;
    rocsparse_int* d_csr_row_ptr_C;
    HIP_CHECK(hipMalloc((void**)&d_csr_row_ptr_C, sizeof(rocsparse_int) * (m + 1)));

    ROCSPARSE_CHECK(rocsparse_csrgemm_nnz(handle,
                                          rocsparse_operation_none,
                                          rocsparse_operation_none,
                                          m,
                                          n,
                                          k,
                                          descr_A,
                                          nnz_A,
                                          d_csr_row_ptr_A,
                                          d_csr_col_ind_A,
                                          descr_B,
                                          nnz_B,
                                          d_csr_row_ptr_B,
                                          d_csr_col_ind_B,
                                          descr_D,
                                          nnz_D,
                                          d_csr_row_ptr_D,
                                          d_csr_col_ind_D,
                                          descr_C,
                                          d_csr_row_ptr_C,
                                          &nnz_C,
                                          info_C,
                                          buffer));

    // Compute column indices and values of C
    rocsparse_int* d_csr_col_ind_C;
    float*         d_csr_val_C;
    HIP_CHECK(hipMalloc((void**)&d_csr_col_ind_C, sizeof(rocsparse_int) * nnz_C));
    HIP_CHECK(hipMalloc((void**)&d_csr_val_C, sizeof(float) * nnz_C));

    ROCSPARSE_CHECK(rocsparse_scsrgemm(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       m,
                                       n,
                                       k,
                                       &alpha,
                                       descr_A,
                                       nnz_A,
                                       d_csr_val_A,
                                       d_csr_row_ptr_A,
                                       d_csr_col_ind_A,
                                       descr_B,
                                       nnz_B,
                                       d_csr_val_B,
                                       d_csr_row_ptr_B,
                                       d_csr_col_ind_B,
                                       &beta,
                                       descr_D,
                                       nnz_D,
                                       d_csr_val_D,
                                       d_csr_row_ptr_D,
                                       d_csr_col_ind_D,
                                       descr_C,
                                       d_csr_val_C,
                                       d_csr_row_ptr_C,
                                       d_csr_col_ind_C,
                                       info_C,
                                       buffer));

    HIP_CHECK(hipFree(d_csr_row_ptr_A));
    HIP_CHECK(hipFree(d_csr_col_ind_A));
    HIP_CHECK(hipFree(d_csr_val_A));

    HIP_CHECK(hipFree(d_csr_row_ptr_B));
    HIP_CHECK(hipFree(d_csr_col_ind_B));
    HIP_CHECK(hipFree(d_csr_val_B));

    HIP_CHECK(hipFree(d_csr_row_ptr_C));
    HIP_CHECK(hipFree(d_csr_col_ind_C));
    HIP_CHECK(hipFree(d_csr_val_C));

    HIP_CHECK(hipFree(d_csr_row_ptr_D));
    HIP_CHECK(hipFree(d_csr_col_ind_D));
    HIP_CHECK(hipFree(d_csr_val_D));

    HIP_CHECK(hipFree(buffer));

    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_A));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_B));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_C));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_D));

    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info_C));

    // Destroy rocSPARSE handle
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
//! [doc example]
