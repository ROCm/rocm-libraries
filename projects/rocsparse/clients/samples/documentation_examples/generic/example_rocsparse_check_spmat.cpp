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
    // 1 2 0 0
    // 3 0 4 0  // <-------contains a "3" in the lower part of matrix
    // 0 0 1 1
    // 0 0 0 2
    std::vector<int>   hcsr_row_ptr = {0, 2, 4, 6, 7};
    std::vector<int>   hcsr_col_ind = {0, 1, 0, 2, 2, 3, 3};
    std::vector<float> hcsr_val     = {1, 2, 3, 4, 1, 1, 2};

    int M   = 4;
    int N   = 4;
    int nnz = 7;

    int*   dcsr_row_ptr;
    int*   dcsr_col_ind;
    float* dcsr_val;
    HIP_CHECK(hipMalloc(&dcsr_row_ptr, sizeof(int) * (M + 1)));
    HIP_CHECK(hipMalloc(&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc(&dcsr_val, sizeof(float) * nnz));

    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (M + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));

    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    rocsparse_spmat_descr matA;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matA,
                                               M,
                                               N,
                                               nnz,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               dcsr_val,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r));

    const rocsparse_fill_mode   fill_mode   = rocsparse_fill_mode_upper;
    const rocsparse_matrix_type matrix_type = rocsparse_matrix_type_triangular;

    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        matA, rocsparse_spmat_fill_mode, &fill_mode, sizeof(fill_mode)));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        matA, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)));

    rocsparse_data_status data_status;

    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_check_spmat(handle,
                                          matA,
                                          &data_status,
                                          rocsparse_check_spmat_stage_buffer_size,
                                          &buffer_size,
                                          nullptr));

    void* dbuffer;
    HIP_CHECK(hipMalloc(&dbuffer, buffer_size));

    ROCSPARSE_CHECK(rocsparse_check_spmat(
        handle, matA, &data_status, rocsparse_check_spmat_stage_compute, &buffer_size, dbuffer));

    std::cout << "data_status: " << data_status << std::endl;

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matA));

    HIP_CHECK(hipFree(dbuffer));
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));

    return 0;
}
//! [doc example]
