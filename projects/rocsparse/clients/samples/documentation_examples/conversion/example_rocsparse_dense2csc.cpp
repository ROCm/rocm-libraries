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
#include <vector>

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

    // Define a dense matrix
    int                 m      = 3;
    int                 n      = 3;
    rocsparse_direction dir    = rocsparse_direction_column;
    std::vector<float>  hdense = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Allocate device memory for the dense matrix
    float* ddense;
    HIP_CHECK(hipMalloc((void**)&ddense, sizeof(float) * m * n));
    HIP_CHECK(hipMemcpy(ddense, hdense.data(), sizeof(float) * m * n, hipMemcpyHostToDevice));

    // Create matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    rocsparse_int* dnnz_per_column;
    HIP_CHECK(hipMalloc(&dnnz_per_column, sizeof(rocsparse_int) * n));

    rocsparse_int nnz_A;
    ROCSPARSE_CHECK(rocsparse_snnz(handle, dir, m, n, descr, ddense, m, dnnz_per_column, &nnz_A));

    // Allocate device memory for CSC format
    rocsparse_int* dcsc_col_ptr;
    rocsparse_int* dcsc_row_ind;
    float*         dcsc_val;

    HIP_CHECK(hipMalloc((void**)&dcsc_col_ptr, sizeof(rocsparse_int) * (n + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsc_row_ind, sizeof(rocsparse_int) * nnz_A));
    HIP_CHECK(hipMalloc((void**)&dcsc_val, sizeof(float) * nnz_A));

    // Convert dense matrix to CSC format
    ROCSPARSE_CHECK(rocsparse_sdense2csc(
        handle, m, n, descr, ddense, m, dnnz_per_column, dcsc_val, dcsc_col_ptr, dcsc_row_ind));

    // Copy result back to host
    std::vector<int>   hcsc_col_ptr(n + 1);
    std::vector<int>   hcsc_row_ind(nnz_A);
    std::vector<float> hcsc_val(nnz_A);

    HIP_CHECK(
        hipMemcpy(hcsc_col_ptr.data(), dcsc_col_ptr, sizeof(int) * (n + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(hcsc_row_ind.data(), dcsc_row_ind, sizeof(int) * nnz_A, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsc_val.data(), dcsc_val, sizeof(float) * nnz_A, hipMemcpyDeviceToHost));

    // Print the CSC matrix
    std::cout << "CSC format:" << std::endl;
    for(int i = 0; i < n; ++i)
    {
        for(int j = hcsc_col_ptr[i]; j < hcsc_col_ptr[i + 1]; ++j)
        {
            std::cout << "Col: " << i << ", Row: " << hcsc_row_ind[j] << ", Val: " << hcsc_val[j]
                      << std::endl;
        }
    }

    // Clean up
    HIP_CHECK(hipFree(ddense));
    HIP_CHECK(hipFree(dnnz_per_column));
    HIP_CHECK(hipFree(dcsc_col_ptr));
    HIP_CHECK(hipFree(dcsc_row_ind));
    HIP_CHECK(hipFree(dcsc_val));

    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
//! [doc example]
