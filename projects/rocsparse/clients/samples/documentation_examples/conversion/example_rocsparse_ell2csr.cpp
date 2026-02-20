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
    //     1 2 0 3 0
    // A = 0 4 5 0 0
    //     6 0 0 7 8
    rocsparse_int m         = 3;
    rocsparse_int n         = 5;
    rocsparse_int nnz       = 8;
    rocsparse_int ell_width = 3;

    std::vector<rocsparse_int> hell_col_ind = {0, 1, 0, 1, 2, 3, 3, -1, 4};
    std::vector<float>         hell_val     = {1, 4, 6, 2, 5, 7, 3, 0, 8};

    rocsparse_int* dell_col_ind = nullptr;
    float*         dell_val     = nullptr;
    HIP_CHECK(hipMalloc((void**)&dell_col_ind, sizeof(rocsparse_int) * m * ell_width));
    HIP_CHECK(hipMalloc((void**)&dell_val, sizeof(float) * m * ell_width));

    HIP_CHECK(hipMemcpy(dell_col_ind,
                        hell_col_ind.data(),
                        sizeof(rocsparse_int) * m * ell_width,
                        hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dell_val, hell_val.data(), sizeof(float) * m * ell_width, hipMemcpyHostToDevice));

    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create ELL matrix descriptor
    rocsparse_mat_descr ell_descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&ell_descr));

    // Create CSR matrix descriptor
    rocsparse_mat_descr csr_descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&csr_descr));

    // Allocate csr_row_ptr array for row offsets
    rocsparse_int* dcsr_row_ptr;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr, sizeof(rocsparse_int) * (m + 1)));

    // Obtain the number of CSR non-zero entries
    // and fill csr_row_ptr array with row offsets
    rocsparse_int csr_nnz;
    ROCSPARSE_CHECK(rocsparse_ell2csr_nnz(
        handle, m, n, ell_descr, ell_width, dell_col_ind, csr_descr, dcsr_row_ptr, &csr_nnz));

    // Allocate CSR column and value arrays
    rocsparse_int* dcsr_col_ind = nullptr;
    float*         dcsr_val     = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind, sizeof(rocsparse_int) * csr_nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val, sizeof(float) * csr_nnz));

    // Format conversion
    ROCSPARSE_CHECK(rocsparse_sell2csr(handle,
                                       m,
                                       n,
                                       ell_descr,
                                       ell_width,
                                       dell_val,
                                       dell_col_ind,
                                       csr_descr,
                                       dcsr_val,
                                       dcsr_row_ptr,
                                       dcsr_col_ind));

    HIP_CHECK(hipFree(dell_col_ind));
    HIP_CHECK(hipFree(dell_val));

    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));

    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(ell_descr));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(csr_descr));

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
//! [doc example]
