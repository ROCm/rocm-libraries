/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
    //     1 2 3 4 0 5
    // A = 0 2 3 0 0 4
    //     5 0 6 7 8 0
    //     1 0 9 0 6 7
    int m = 4;
    int n = 6;

    std::vector<int> hcsr_row_ptr = {0, 5, 8, 12, 16};
    // std::vector<int>    hcsr_col_ind = {0, 1, 2, 3, 5,
    //                                     1, 2, 5,
    //                                     0, 2, 3, 4,
    //                                     0, 2, 4, 5};
    // std::vector<float>  hcsr_val     = {1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7, 8, 1, 9, 6, 7};
    std::vector<int>   hcsr_col_ind = {5, 3, 2, 1, 0, 1, 5, 2, 4, 2, 3, 0, 0, 5, 4, 2};
    std::vector<float> hcsr_val     = {5, 4, 3, 2, 1, 2, 4, 3, 8, 6, 7, 5, 1, 7, 6, 9};

    int nnz = hcsr_row_ptr[m] - hcsr_row_ptr[0];

    // Offload data to device
    int*   dcsr_row_ptr;
    int*   dcsr_col_ind;
    float* dcsr_val;
    HIP_CHECK(hipMalloc(&dcsr_row_ptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc(&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc(&dcsr_val, sizeof(float) * nnz));

    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));

    rocsparse_handle      handle;
    rocsparse_error       p_error[1] = {};
    rocsparse_spmat_descr matA;

    rocsparse_indextype  row_idx_type = rocsparse_indextype_i32;
    rocsparse_indextype  col_idx_type = rocsparse_indextype_i32;
    rocsparse_datatype   data_type    = rocsparse_datatype_f32_r;
    rocsparse_index_base idx_base     = rocsparse_index_base_zero;

    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create sparse matrix A
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matA,
                                               m,
                                               n,
                                               nnz,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               dcsr_val,
                                               row_idx_type,
                                               col_idx_type,
                                               idx_base,
                                               data_type));

    rocsparse_spsort_descr spsort_descr;
    ROCSPARSE_CHECK(rocsparse_spsort_descr_create(handle, &spsort_descr, p_error));

    const rocsparse_spsort_alg spsort_alg = rocsparse_spsort_alg_default;
    ROCSPARSE_CHECK(rocsparse_spsort_set_input(handle,
                                               spsort_descr,
                                               rocsparse_spsort_input_alg,
                                               &spsort_alg,
                                               sizeof(spsort_alg),
                                               p_error));

    // Sort the column indices within each row
    const rocsparse_direction spsort_direction = rocsparse_direction_row;
    ROCSPARSE_CHECK(rocsparse_spsort_set_input(handle,
                                               spsort_descr,
                                               rocsparse_spsort_input_direction,
                                               &spsort_direction,
                                               sizeof(spsort_direction),
                                               p_error));

    // Passing matA as both the input and the output sorts it in place
    // Call spsort to get buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_spsort_buffer_size(
        handle, spsort_descr, matA, matA, rocsparse_spsort_stage_analysis, &buffer_size, p_error));

    void* buffer;
    HIP_CHECK(hipMalloc(&buffer, buffer_size));

    // Call spsort to perform analysis
    ROCSPARSE_CHECK(rocsparse_spsort(handle,
                                     spsort_descr,
                                     matA,
                                     matA,
                                     rocsparse_spsort_stage_analysis,
                                     buffer_size,
                                     buffer,
                                     p_error));

    HIP_CHECK(hipFree(buffer));

    ROCSPARSE_CHECK(rocsparse_spsort_buffer_size(
        handle, spsort_descr, matA, matA, rocsparse_spsort_stage_compute, &buffer_size, p_error));

    HIP_CHECK(hipMalloc(&buffer, buffer_size));

    // Call spsort to perform computation
    ROCSPARSE_CHECK(rocsparse_spsort(handle,
                                     spsort_descr,
                                     matA,
                                     matA,
                                     rocsparse_spsort_stage_compute,
                                     buffer_size,
                                     buffer,
                                     p_error));

    HIP_CHECK(hipFree(buffer));

    ROCSPARSE_CHECK(rocsparse_destroy_error(p_error[0]));
    ROCSPARSE_CHECK(rocsparse_spsort_descr_destroy(handle, spsort_descr, nullptr));

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matA));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));

    return 0;
}
//! [doc example]
