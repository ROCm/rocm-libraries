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
    //     1 2 0 0
    // A = 3 0 0 4
    //     5 6 0 4
    //     7 4 2 5
    rocsparse_int m         = 4;
    rocsparse_int n         = 4;
    rocsparse_int nnz_A     = 11;
    float         threshold = 5.0f;

    std::vector<rocsparse_int> hcsr_row_ptr_A = {0, 2, 4, 7, 11};
    std::vector<rocsparse_int> hcsr_col_ind_A = {0, 1, 0, 3, 0, 1, 3, 0, 1, 2, 3};
    std::vector<float>         hcsr_val_A     = {1, 2, 3, 4, 5, 6, 4, 7, 4, 2, 5};

    rocsparse_int* dcsr_row_ptr_A = nullptr;
    rocsparse_int* dcsr_col_ind_A = nullptr;
    float*         dcsr_val_A     = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr_A, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind_A, sizeof(rocsparse_int) * nnz_A));
    HIP_CHECK(hipMalloc((void**)&dcsr_val_A, sizeof(float) * nnz_A));

    HIP_CHECK(hipMemcpy(dcsr_row_ptr_A,
                        hcsr_row_ptr_A.data(),
                        sizeof(rocsparse_int) * (m + 1),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_col_ind_A,
                        hcsr_col_ind_A.data(),
                        sizeof(rocsparse_int) * nnz_A,
                        hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_val_A, hcsr_val_A.data(), sizeof(float) * nnz_A, hipMemcpyHostToDevice));

    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    rocsparse_mat_descr descr_A;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_A));

    rocsparse_mat_descr descr_C;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_C));

    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    rocsparse_int* dcsr_row_ptr_C = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr_C, sizeof(rocsparse_int) * (m + 1)));

    // Obtain the temporary buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_sprune_csr2csr_buffer_size(handle,
                                                         m,
                                                         n,
                                                         nnz_A,
                                                         descr_A,
                                                         dcsr_val_A,
                                                         dcsr_row_ptr_A,
                                                         dcsr_col_ind_A,
                                                         &threshold,
                                                         descr_C,
                                                         nullptr,
                                                         dcsr_row_ptr_C,
                                                         nullptr,
                                                         &buffer_size));

    // Allocate temporary buffer
    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    rocsparse_int nnz_C;
    ROCSPARSE_CHECK(rocsparse_sprune_csr2csr_nnz(handle,
                                                 m,
                                                 n,
                                                 nnz_A,
                                                 descr_A,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 &threshold,
                                                 descr_C,
                                                 dcsr_row_ptr_C,
                                                 &nnz_C,
                                                 temp_buffer));

    rocsparse_int* dcsr_col_ind_C = nullptr;
    float*         dcsr_val_C     = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind_C, sizeof(rocsparse_int) * nnz_C));
    HIP_CHECK(hipMalloc((void**)&dcsr_val_C, sizeof(float) * nnz_C));

    ROCSPARSE_CHECK(rocsparse_sprune_csr2csr(handle,
                                             m,
                                             n,
                                             nnz_A,
                                             descr_A,
                                             dcsr_val_A,
                                             dcsr_row_ptr_A,
                                             dcsr_col_ind_A,
                                             &threshold,
                                             descr_C,
                                             dcsr_val_C,
                                             dcsr_row_ptr_C,
                                             dcsr_col_ind_C,
                                             temp_buffer));

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_A));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_C));

    HIP_CHECK(hipFree(temp_buffer));

    HIP_CHECK(hipFree(dcsr_row_ptr_A));
    HIP_CHECK(hipFree(dcsr_col_ind_A));
    HIP_CHECK(hipFree(dcsr_val_A));

    HIP_CHECK(hipFree(dcsr_row_ptr_C));
    HIP_CHECK(hipFree(dcsr_col_ind_C));
    HIP_CHECK(hipFree(dcsr_val_C));

    return 0;
}
//! [doc example]
