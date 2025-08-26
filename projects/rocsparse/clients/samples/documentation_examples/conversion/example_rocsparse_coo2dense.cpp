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
    // 1 2 3 0
    // 0 0 4 5
    // 0 6 0 0
    // 7 0 0 8
    rocsparse_int m   = 4;
    rocsparse_int n   = 4;
    rocsparse_int nnz = 8;
    rocsparse_int ld  = m;

    std::vector<rocsparse_int> hcoo_row_ind = {0, 0, 0, 1, 1, 2, 3, 3};
    std::vector<rocsparse_int> hcoo_col_ind = {0, 1, 2, 2, 3, 1, 0, 3};
    std::vector<float>         hcoo_val     = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    rocsparse_int* dcoo_row_ind = nullptr;
    rocsparse_int* dcoo_col_ind = nullptr;
    float*         dcoo_val     = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcoo_row_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcoo_col_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcoo_val, sizeof(float) * nnz));

    HIP_CHECK(hipMemcpy(
        dcoo_row_ind, hcoo_row_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcoo_col_ind, hcoo_col_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcoo_val, hcoo_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));

    float* ddense = nullptr;
    HIP_CHECK(hipMalloc((void**)&ddense, sizeof(float) * ld * n));

    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    ROCSPARSE_CHECK(rocsparse_scoo2dense(
        handle, m, n, nnz, descr, dcoo_val, dcoo_row_ind, dcoo_col_ind, ddense, ld));

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));

    HIP_CHECK(hipFree(dcoo_row_ind));
    HIP_CHECK(hipFree(dcoo_col_ind));
    HIP_CHECK(hipFree(dcoo_val));

    HIP_CHECK(hipFree(ddense));

    return 0;
}
//! [doc example]
