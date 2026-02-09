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
    //     1 4 0 0 0 0
    // A = 0 2 3 0 0 0
    //     5 0 0 7 8 0
    //     0 0 9 0 6 0

    rocsparse_int m             = 4;
    rocsparse_int n             = 6;
    rocsparse_int row_block_dim = 2;
    rocsparse_int col_block_dim = 3;
    rocsparse_int nnz           = 9;
    rocsparse_int mb            = (m + row_block_dim - 1) / row_block_dim;
    rocsparse_int nb            = (n + col_block_dim - 1) / col_block_dim;

    rocsparse_direction dir = rocsparse_direction_row;

    std::vector<rocsparse_int> hcsr_row_ptr = {0, 2, 4, 7, 9};
    std::vector<rocsparse_int> hcsr_col_ind = {0, 1, 1, 2, 0, 3, 4, 2, 4};
    std::vector<float>         hcsr_val     = {1, 4, 2, 3, 5, 7, 8, 9, 6};

    rocsparse_int* dcsr_row_ptr = nullptr;
    rocsparse_int* dcsr_col_ind = nullptr;
    float*         dcsr_val     = nullptr;

    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val, sizeof(float) * nnz));

    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));

    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    rocsparse_mat_descr csr_descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&csr_descr));

    rocsparse_mat_descr bsr_descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&bsr_descr));

    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_scsr2gebsr_buffer_size(handle,
                                                     dir,
                                                     m,
                                                     n,
                                                     csr_descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     row_block_dim,
                                                     col_block_dim,
                                                     &buffer_size));

    void* buffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&buffer, buffer_size));

    rocsparse_int* dbsr_row_ptr = nullptr;
    HIP_CHECK(hipMalloc(&dbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1)));

    rocsparse_int nnzb;
    ROCSPARSE_CHECK(rocsparse_csr2gebsr_nnz(handle,
                                            dir,
                                            m,
                                            n,
                                            csr_descr,
                                            dcsr_row_ptr,
                                            dcsr_col_ind,
                                            bsr_descr,
                                            dbsr_row_ptr,
                                            row_block_dim,
                                            col_block_dim,
                                            &nnzb,
                                            buffer));

    rocsparse_int* dbsr_col_ind = nullptr;
    float*         dbsr_val     = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbsr_col_ind, sizeof(rocsparse_int) * nnzb));
    HIP_CHECK(hipMalloc((void**)&dbsr_val, sizeof(float) * nnzb * row_block_dim * col_block_dim));

    ROCSPARSE_CHECK(rocsparse_scsr2gebsr(handle,
                                         dir,
                                         m,
                                         n,
                                         csr_descr,
                                         dcsr_val,
                                         dcsr_row_ptr,
                                         dcsr_col_ind,
                                         bsr_descr,
                                         dbsr_val,
                                         dbsr_row_ptr,
                                         dbsr_col_ind,
                                         row_block_dim,
                                         col_block_dim,
                                         buffer));

    HIP_CHECK(hipFree(buffer));

    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));

    HIP_CHECK(hipFree(dbsr_row_ptr));
    HIP_CHECK(hipFree(dbsr_col_ind));
    HIP_CHECK(hipFree(dbsr_val));

    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(csr_descr));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(bsr_descr));

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
//! [doc example]
