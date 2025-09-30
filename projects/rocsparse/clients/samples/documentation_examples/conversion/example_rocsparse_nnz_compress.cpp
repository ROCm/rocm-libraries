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
    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Matrix descriptor
    rocsparse_mat_descr descr_A;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_A));

    //     1 2 0 3 0
    // A = 0 4 5 0 0
    //     6 0 0 7 8
    float tol = 4.2f;

    rocsparse_int m     = 3;
    rocsparse_int n     = 5;
    rocsparse_int nnz_A = 8;

    rocsparse_int hcsr_row_ptr_A[4] = {0, 3, 5, 8};
    float         hcsr_val_A[8]     = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    rocsparse_int* dcsr_row_ptr_A = nullptr;
    float*         dcsr_val_A     = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr_A, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_val_A, sizeof(float) * nnz_A));

    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A, sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val_A, hcsr_val_A, sizeof(float) * nnz_A, hipMemcpyHostToDevice));

    // Allocate memory for the nnz_per_row array
    rocsparse_int* dnnz_per_row;
    HIP_CHECK(hipMalloc((void**)&dnnz_per_row, sizeof(rocsparse_int) * m));

    // Call snnz_compress() which fills in nnz_per_row array and finds the number
    // of entries that will be in the compressed CSR matrix
    rocsparse_int nnz_C;
    ROCSPARSE_CHECK(rocsparse_snnz_compress(
        handle, m, descr_A, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, &nnz_C, tol));

    // Copy result back to host
    rocsparse_int hnnz_per_row[3];
    HIP_CHECK(
        hipMemcpy(hnnz_per_row, dnnz_per_row, sizeof(rocsparse_int) * m, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(dcsr_row_ptr_A));
    HIP_CHECK(hipFree(dcsr_val_A));
    HIP_CHECK(hipFree(dnnz_per_row));

    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_A));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
//! [doc example]
