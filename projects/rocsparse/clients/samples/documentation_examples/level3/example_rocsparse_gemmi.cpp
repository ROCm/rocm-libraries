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
#include <vector>

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
    rocsparse_int m   = 2;
    rocsparse_int n   = 5;
    rocsparse_int k   = 3;
    rocsparse_int nnz = 8;
    rocsparse_int lda = m;
    rocsparse_int ldc = m;

    // alpha and beta
    float alpha = 1.0f;
    float beta  = 0.0f;

    // Matrix A (m x k)
    // (  9.0  10.0  11.0 )
    // ( 12.0  13.0  14.0 )

    // Matrix B (k x n)
    // ( 1.0  2.0  0.0  3.0  0.0 )
    // ( 0.0  4.0  5.0  0.0  0.0 )
    // ( 6.0  0.0  0.0  7.0  8.0 )

    // Matrix C (m x n)
    // ( 15.0  16.0  17.0  18.0  19.0 )
    // ( 20.0  21.0  22.0  23.0  24.0 )

    std::vector<float> hA             = {9.0, 12.0, 10.0, 13.0, 11.0, 14.0};
    std::vector<int>   hcsc_col_ptr_B = {0, 2, 4, 5, 7, 8};
    std::vector<int>   hcsc_row_ind_B = {0, 2, 0, 1, 1, 0, 2, 2};
    std::vector<float> hcsc_val_B     = {1.0, 6.0, 2.0, 4.0, 5.0, 3.0, 7.0, 8.0};
    std::vector<float> hC = {15.0, 20.0, 16.0, 21.0, 17.0, 22.0, 18.0, 23.0, 19.0, 24.0};

    float* dA;
    int*   dcsc_col_ptr_B;
    int*   dcsc_row_ind_B;
    float* dcsc_val_B;
    float* dC;
    HIP_CHECK(hipMalloc(&dA, sizeof(float) * lda * k));
    HIP_CHECK(hipMalloc(&dcsc_col_ptr_B, sizeof(int) * (n + 1)));
    HIP_CHECK(hipMalloc(&dcsc_row_ind_B, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc(&dcsc_val_B, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc(&dC, sizeof(float) * ldc * n));

    HIP_CHECK(hipMemcpy(dA, hA.data(), sizeof(float) * lda * k, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcsc_col_ptr_B, hcsc_col_ptr_B.data(), sizeof(int) * (n + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsc_row_ind_B, hcsc_row_ind_B.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsc_val_B, hcsc_val_B.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hC.data(), sizeof(float) * ldc * n, hipMemcpyHostToDevice));

    // Create rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create matrix descriptor
    rocsparse_mat_descr descr_B;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_B));

    // Perform the matrix multiplication
    ROCSPARSE_CHECK(rocsparse_sgemmi(handle,
                                     rocsparse_operation_none,
                                     rocsparse_operation_transpose,
                                     m,
                                     n,
                                     k,
                                     nnz,
                                     &alpha,
                                     dA,
                                     lda,
                                     descr_B,
                                     dcsc_val_B,
                                     dcsc_col_ptr_B,
                                     dcsc_row_ind_B,
                                     &beta,
                                     dC,
                                     ldc));

    HIP_CHECK(hipMemcpy(hC.data(), dC, sizeof(float) * ldc * n, hipMemcpyDeviceToHost));

    std::cout << "hC" << std::endl;
    for(size_t i = 0; i < hC.size(); i++)
    {
        std::cout << hC[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clean up
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_B));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear up on device
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dcsc_col_ptr_B));
    HIP_CHECK(hipFree(dcsc_row_ind_B));
    HIP_CHECK(hipFree(dcsc_val_B));
    HIP_CHECK(hipFree(dC));

    return 0;
}
//! [doc example]
