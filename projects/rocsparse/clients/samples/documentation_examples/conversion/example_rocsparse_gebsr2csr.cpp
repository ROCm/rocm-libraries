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
    //     1 4 0 0 0 0
    // A = 0 2 3 0 0 0
    //     5 0 0 7 8 0
    //     0 0 9 0 6 0

    rocsparse_int mb            = 2;
    rocsparse_int nb            = 2;
    rocsparse_int row_block_dim = 2;
    rocsparse_int col_block_dim = 3;
    rocsparse_int m             = mb * row_block_dim;
    rocsparse_int n             = nb * col_block_dim;

    // Define host arrays
    rocsparse_int h_bsr_row_ptr[] = {0, 1, 3};
    rocsparse_int h_bsr_col_ind[] = {0, 0, 1};
    float         h_bsr_val[]     = {1, 0, 4, 2, 0, 3, 5, 0, 0, 0, 0, 9, 7, 0, 8, 6, 0, 0};

    rocsparse_int nnzb = h_bsr_row_ptr[mb] - h_bsr_row_ptr[0];

    // Allocate device memory for BSR matrix
    rocsparse_int* d_bsr_row_ptr;
    rocsparse_int* d_bsr_col_ind;
    float*         d_bsr_val;

    HIP_CHECK(hipMalloc((void**)&d_bsr_row_ptr, sizeof(rocsparse_int) * (mb + 1)));
    HIP_CHECK(hipMalloc((void**)&d_bsr_col_ind, sizeof(rocsparse_int) * nnzb));
    HIP_CHECK(hipMalloc((void**)&d_bsr_val, sizeof(float) * nnzb * row_block_dim * col_block_dim));

    HIP_CHECK(hipMemcpy(
        d_bsr_row_ptr, h_bsr_row_ptr, sizeof(rocsparse_int) * (mb + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_bsr_col_ind, h_bsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bsr_val,
                        h_bsr_val,
                        sizeof(float) * nnzb * row_block_dim * col_block_dim,
                        hipMemcpyHostToDevice));

    // Create CSR arrays on device
    rocsparse_int* d_csr_row_ptr;
    rocsparse_int* d_csr_col_ind;
    float*         d_csr_val;
    HIP_CHECK(hipMalloc((void**)&d_csr_row_ptr, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&d_csr_col_ind,
                        sizeof(rocsparse_int) * nnzb * row_block_dim * col_block_dim));
    HIP_CHECK(hipMalloc((void**)&d_csr_val, sizeof(float) * nnzb * row_block_dim * col_block_dim));

    // Create rocsparse handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    rocsparse_mat_descr bsr_descr = nullptr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&bsr_descr));

    rocsparse_mat_descr csr_descr = nullptr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&csr_descr));

    ROCSPARSE_CHECK(rocsparse_set_mat_index_base(bsr_descr, rocsparse_index_base_zero));
    ROCSPARSE_CHECK(rocsparse_set_mat_index_base(csr_descr, rocsparse_index_base_zero));

    // Format conversion
    ROCSPARSE_CHECK(rocsparse_sgebsr2csr(handle,
                                         rocsparse_direction_column,
                                         mb,
                                         nb,
                                         bsr_descr,
                                         d_bsr_val,
                                         d_bsr_row_ptr,
                                         d_bsr_col_ind,
                                         row_block_dim,
                                         col_block_dim,
                                         csr_descr,
                                         d_csr_val,
                                         d_csr_row_ptr,
                                         d_csr_col_ind));

    // Clean up
    HIP_CHECK(hipFree(d_bsr_row_ptr));
    HIP_CHECK(hipFree(d_bsr_col_ind));
    HIP_CHECK(hipFree(d_bsr_val));
    HIP_CHECK(hipFree(d_csr_row_ptr));
    HIP_CHECK(hipFree(d_csr_col_ind));
    HIP_CHECK(hipFree(d_csr_val));

    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(bsr_descr));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(csr_descr));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
//! [doc example]
