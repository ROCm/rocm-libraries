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
    //     1 2 3 4 0 0
    // A = 3 4 0 0 0 0
    //     6 5 3 4 0 0
    //     1 2 0 0 0 0
    rocsparse_int m   = 4;
    rocsparse_int n   = 6;
    rocsparse_int nnz = 12;

    std::vector<rocsparse_int> hcsr_row_ptr = {0, 4, 6, 10, 12};
    std::vector<rocsparse_int> hcsr_col_ind = {0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 0, 1};
    std::vector<float>         hcsr_val     = {1, 2, 3, 4, 3, 4, 6, 5, 3, 4, 1, 2};

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

    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    rocsparse_hyb_mat hyb;
    ROCSPARSE_CHECK(rocsparse_create_hyb_mat(&hyb));

    rocsparse_int           user_ell_width = 3;
    rocsparse_hyb_partition partition_type = rocsparse_hyb_partition_user;
    ROCSPARSE_CHECK(rocsparse_scsr2hyb(handle,
                                       m,
                                       n,
                                       descr,
                                       dcsr_val,
                                       dcsr_row_ptr,
                                       dcsr_col_ind,
                                       hyb,
                                       user_ell_width,
                                       partition_type));

    rocsparse_int* dcsr_row_ptr2 = nullptr;
    rocsparse_int* dcsr_col_ind2 = nullptr;
    float*         dcsr_val2     = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr2, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind2, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val2, sizeof(float) * nnz));

    // Obtain the temporary buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_hyb2csr_buffer_size(handle, descr, hyb, dcsr_row_ptr2, &buffer_size));

    // Allocate temporary buffer
    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    ROCSPARSE_CHECK(rocsparse_shyb2csr(
        handle, descr, hyb, dcsr_val2, dcsr_row_ptr2, dcsr_col_ind2, temp_buffer));

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_hyb_mat(hyb));

    HIP_CHECK(hipFree(temp_buffer));

    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));

    HIP_CHECK(hipFree(dcsr_row_ptr2));
    HIP_CHECK(hipFree(dcsr_col_ind2));
    HIP_CHECK(hipFree(dcsr_val2));

    return 0;
}
//! [doc example]
