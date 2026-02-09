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
    //     1 2 0 3 0 0
    // A = 0 4 5 0 0 0
    //     0 0 0 7 8 0
    //     0 0 1 2 4 1

    rocsparse_int       block_dim = 2;
    rocsparse_int       mb        = 2;
    rocsparse_int       kb        = 3;
    rocsparse_int       nnzb      = 4;
    rocsparse_direction dir       = rocsparse_direction_row;

    // alpha and beta
    float alpha = 1.0f;
    float beta  = 0.0f;

    std::vector<rocsparse_int> hbsr_row_ptr = {0, 2, 4};
    std::vector<rocsparse_int> hbsr_col_ind = {0, 1, 1, 2};
    std::vector<float>         hbsr_val     = {1, 2, 0, 4, 0, 3, 5, 0, 0, 7, 1, 2, 8, 0, 4, 1};

    // Set dimension n of B
    rocsparse_int n = 64;
    rocsparse_int m = mb * block_dim;
    rocsparse_int k = kb * block_dim;

    // Allocate and generate column-oriented dense matrix B
    std::vector<float> hB(k * n);
    for(rocsparse_int i = 0; i < k * n; ++i)
    {
        hB[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    std::vector<float> hC(m * n);

    int*   dbsr_row_ptr;
    int*   dbsr_col_ind;
    float* dbsr_val;
    float* dB;
    float* dC;
    HIP_CHECK(hipMalloc(&dbsr_row_ptr, sizeof(int) * (mb + 1)));
    HIP_CHECK(hipMalloc(&dbsr_col_ind, sizeof(int) * nnzb));
    HIP_CHECK(hipMalloc(&dbsr_val, sizeof(float) * nnzb * block_dim * block_dim));
    HIP_CHECK(hipMalloc(&dB, sizeof(float) * k * n));
    HIP_CHECK(hipMalloc(&dC, sizeof(float) * m * n));

    HIP_CHECK(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr.data(), sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind.data(), sizeof(int) * nnzb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsr_val,
                        hbsr_val.data(),
                        sizeof(float) * nnzb * block_dim * block_dim,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), sizeof(float) * k * n, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hC.data(), sizeof(float) * m * n, hipMemcpyHostToDevice));

    // Create rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Perform the matrix multiplication
    ROCSPARSE_CHECK(rocsparse_sbsrmm(handle,
                                     dir,
                                     rocsparse_operation_none,
                                     rocsparse_operation_none,
                                     mb,
                                     n,
                                     kb,
                                     nnzb,
                                     &alpha,
                                     descr,
                                     dbsr_val,
                                     dbsr_row_ptr,
                                     dbsr_col_ind,
                                     block_dim,
                                     dB,
                                     k,
                                     &beta,
                                     dC,
                                     m));

    HIP_CHECK(hipMemcpy(hC.data(), dC, sizeof(float) * m * n, hipMemcpyDeviceToHost));

    std::cout << "hC" << std::endl;
    for(size_t i = 0; i < hC.size(); i++)
    {
        std::cout << hC[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clean up
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear up on device
    HIP_CHECK(hipFree(dbsr_row_ptr));
    HIP_CHECK(hipFree(dbsr_col_ind));
    HIP_CHECK(hipFree(dbsr_val));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));

    return 0;
}
//! [doc example]
