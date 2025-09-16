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
    //     1 4 2 1 0 0
    // A = 0 2 3 5 0 0
    //     5 2 2 7 8 6
    //     9 3 9 1 6 1
    rocsparse_int mb        = 2;
    rocsparse_int nb        = 3;
    rocsparse_int block_dim = 2;
    rocsparse_int m         = mb * block_dim;
    rocsparse_int n         = nb * block_dim;
    rocsparse_int nnzb      = 5;
    rocsparse_int nnz       = nnzb * block_dim * block_dim;

    std::vector<rocsparse_int> hbsr_row_ptr = {0, 2, 5};
    std::vector<rocsparse_int> hbsr_col_ind = {0, 1, 0, 1, 2};
    std::vector<float> hbsr_val = {1.0f, 0.0f, 4.0f, 2.0f, 2.0f, 3.0f, 1.0f, 5.0f, 5.0f, 9.0f,
                                   2.0f, 3.0f, 2.0f, 9.0f, 7.0f, 1.0f, 8.0f, 6.0f, 6.0f, 1.0f};

    rocsparse_int* dbsr_row_ptr = nullptr;
    rocsparse_int* dbsr_col_ind = nullptr;
    float*         dbsr_val     = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1)));
    HIP_CHECK(hipMalloc((void**)&dbsr_col_ind, sizeof(rocsparse_int) * nnzb));
    HIP_CHECK(hipMalloc((void**)&dbsr_val, sizeof(float) * nnzb * block_dim * block_dim));

    HIP_CHECK(hipMemcpy(dbsr_row_ptr,
                        hbsr_row_ptr.data(),
                        sizeof(rocsparse_int) * (mb + 1),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dbsr_col_ind, hbsr_col_ind.data(), sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsr_val,
                        hbsr_val.data(),
                        sizeof(float) * nnzb * block_dim * block_dim,
                        hipMemcpyHostToDevice));

    // Create CSR arrays on device
    rocsparse_int* dcsr_row_ptr = nullptr;
    rocsparse_int* dcsr_col_ind = nullptr;
    float*         dcsr_val     = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val, sizeof(float) * nnz));

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
    ROCSPARSE_CHECK(rocsparse_sbsr2csr(handle,
                                       rocsparse_direction_column,
                                       mb,
                                       nb,
                                       bsr_descr,
                                       dbsr_val,
                                       dbsr_row_ptr,
                                       dbsr_col_ind,
                                       block_dim,
                                       csr_descr,
                                       dcsr_val,
                                       dcsr_row_ptr,
                                       dcsr_col_ind));

    // Copy to host
    std::vector<rocsparse_int> hcsr_row_ptr(m + 1);
    std::vector<rocsparse_int> hcsr_col_ind(nnz);
    std::vector<float>         hcsr_val(nnz);
    HIP_CHECK(hipMemcpy(
        hcsr_row_ptr.data(), dcsr_row_ptr, sizeof(rocsparse_int) * (m + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        hcsr_col_ind.data(), dcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsr_val.data(), dcsr_val, sizeof(float) * nnz, hipMemcpyDeviceToHost));

    std::cout << "CSR" << std::endl;
    for(rocsparse_int i = 0; i < m; i++)
    {
        rocsparse_int start = hcsr_row_ptr[i];
        rocsparse_int end   = hcsr_row_ptr[i + 1];

        std::vector<float> temp(n, 0.0f);
        for(rocsparse_int j = start; j < end; j++)
        {
            temp[hcsr_col_ind[j]] = hcsr_val[j];
        }

        for(rocsparse_int j = 0; j < n; j++)
        {
            std::cout << temp[j] << " ";
        }
        std::cout << "" << std::endl;
    }
    std::cout << "" << std::endl;

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(csr_descr));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(bsr_descr));

    HIP_CHECK(hipFree(dbsr_row_ptr));
    HIP_CHECK(hipFree(dbsr_col_ind));
    HIP_CHECK(hipFree(dbsr_val));

    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));

    return 0;
}
//! [doc example]
