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
    float alpha = 1.0f;
    float beta  = 1.0f;

    // Define matrix dimensions and block size
    rocsparse_int mb        = 2; // Number of block rows
    rocsparse_int nb        = 2; // Number of block columns
    rocsparse_int block_dim = 2; // Block dimension

    // Create rocsparse handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create matrix descriptors
    rocsparse_mat_descr descr_A;
    rocsparse_mat_descr descr_B;
    rocsparse_mat_descr descr_C;

    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_A));
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_B));
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_C));

    // Set pointer mode
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Define host arrays for BSR matrices A and B
    rocsparse_int nnzb_A = 3; // Number of non-zero blocks in A
    rocsparse_int nnzb_B = 3; // Number of non-zero blocks in B

    rocsparse_int h_bsr_row_ptr_A[] = {0, 1, 3};
    rocsparse_int h_bsr_col_ind_A[] = {0, 0, 1};
    float         h_bsr_val_A[]     = {1, 0, 4, 2, 0, 3, 5, 0, 0, 0, 0, 9};

    rocsparse_int h_bsr_row_ptr_B[] = {0, 1, 3};
    rocsparse_int h_bsr_col_ind_B[] = {0, 0, 1};
    float         h_bsr_val_B[]     = {0, 0, 0, 0, 0, 0, 7, 0, 8, 6, 0, 0};

    // Allocate device memory for BSR matrices A and B
    rocsparse_int* d_bsr_row_ptr_A;
    rocsparse_int* d_bsr_col_ind_A;
    float*         d_bsr_val_A;

    HIP_CHECK(hipMalloc((void**)&d_bsr_row_ptr_A, sizeof(rocsparse_int) * (mb + 1)));
    HIP_CHECK(hipMalloc((void**)&d_bsr_col_ind_A, sizeof(rocsparse_int) * nnzb_A));
    HIP_CHECK(hipMalloc((void**)&d_bsr_val_A, sizeof(float) * nnzb_A * block_dim * block_dim));

    HIP_CHECK(hipMemcpy(
        d_bsr_row_ptr_A, h_bsr_row_ptr_A, sizeof(rocsparse_int) * (mb + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_bsr_col_ind_A, h_bsr_col_ind_A, sizeof(rocsparse_int) * nnzb_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_val_A,
                        h_bsr_val_A,
                        sizeof(float) * nnzb_A * block_dim * block_dim,
                        hipMemcpyHostToDevice));

    rocsparse_int* d_bsr_row_ptr_B;
    rocsparse_int* d_bsr_col_ind_B;
    float*         d_bsr_val_B;

    HIP_CHECK(hipMalloc((void**)&d_bsr_row_ptr_B, sizeof(rocsparse_int) * (mb + 1)));
    HIP_CHECK(hipMalloc((void**)&d_bsr_col_ind_B, sizeof(rocsparse_int) * nnzb_B));
    HIP_CHECK(hipMalloc((void**)&d_bsr_val_B, sizeof(float) * nnzb_B * block_dim * block_dim));

    HIP_CHECK(hipMemcpy(
        d_bsr_row_ptr_B, h_bsr_row_ptr_B, sizeof(rocsparse_int) * (mb + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_bsr_col_ind_B, h_bsr_col_ind_B, sizeof(rocsparse_int) * nnzb_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_val_B,
                        h_bsr_val_B,
                        sizeof(float) * nnzb_B * block_dim * block_dim,
                        hipMemcpyHostToDevice));

    // Allocate memory for the row pointer array of the compressed CSR matrix
    rocsparse_int* d_bsr_row_ptr_C;
    HIP_CHECK(hipMalloc((void**)&d_bsr_row_ptr_C, sizeof(rocsparse_int) * (mb + 1)));

    // Obtain number of total non-zero block entries in C and block row pointers of C
    rocsparse_int nnzb_C;
    ROCSPARSE_CHECK(rocsparse_bsrgeam_nnzb(handle,
                                           rocsparse_direction_column,
                                           mb,
                                           nb,
                                           block_dim,
                                           descr_A,
                                           nnzb_A,
                                           d_bsr_row_ptr_A,
                                           d_bsr_col_ind_A,
                                           descr_B,
                                           nnzb_B,
                                           d_bsr_row_ptr_B,
                                           d_bsr_col_ind_B,
                                           descr_C,
                                           d_bsr_row_ptr_C,
                                           &nnzb_C));

    // Compute block column indices and block values of C
    rocsparse_int* d_bsr_col_ind_C;
    float*         d_bsr_val_C;
    HIP_CHECK(hipMalloc((void**)&d_bsr_col_ind_C, sizeof(rocsparse_int) * nnzb_C));
    HIP_CHECK(hipMalloc((void**)&d_bsr_val_C, sizeof(float) * nnzb_C * block_dim * block_dim));

    ROCSPARSE_CHECK(rocsparse_sbsrgeam(handle,
                                       rocsparse_direction_column,
                                       mb,
                                       nb,
                                       block_dim,
                                       &alpha,
                                       descr_A,
                                       nnzb_A,
                                       d_bsr_val_A,
                                       d_bsr_row_ptr_A,
                                       d_bsr_col_ind_A,
                                       &beta,
                                       descr_B,
                                       nnzb_B,
                                       d_bsr_val_B,
                                       d_bsr_row_ptr_B,
                                       d_bsr_col_ind_B,
                                       descr_C,
                                       d_bsr_val_C,
                                       d_bsr_row_ptr_C,
                                       d_bsr_col_ind_C));

    // Clean up
    HIP_CHECK(hipFree(d_bsr_row_ptr_A));
    HIP_CHECK(hipFree(d_bsr_col_ind_A));
    HIP_CHECK(hipFree(d_bsr_val_A));
    HIP_CHECK(hipFree(d_bsr_row_ptr_B));
    HIP_CHECK(hipFree(d_bsr_col_ind_B));
    HIP_CHECK(hipFree(d_bsr_val_B));

    return 0;
}
//! [doc example]
