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
    //     1 4 0 0 0 0
    // A = 0 2 3 0 0 0
    //     5 0 0 7 8 0
    //     0 0 9 0 6 0
    int m             = 4;
    int k             = 6;
    int n             = 3;
    int batch_count_A = 1;
    int batch_count_B = 100;
    int batch_count_C = 100;

    std::vector<int>   hcsr_row_ptr = {0, 2, 4, 7, 9};
    std::vector<int>   hcsr_col_ind = {0, 1, 1, 2, 0, 3, 4, 2, 4};
    std::vector<float> hcsr_val     = {1, 4, 2, 3, 5, 7, 8, 9, 6};

    std::vector<float> hB(batch_count_B * k * n, 1.0f);
    std::vector<float> hC(batch_count_C * m * n, 1.0f);

    int nnz = hcsr_row_ptr[m] - hcsr_row_ptr[0];

    int offsets_batch_stride_A        = 0;
    int columns_values_batch_stride_A = 0;
    int batch_stride_B                = k * n;
    int batch_stride_C                = m * n;

    float alpha = 1.0f;
    float beta  = 0.0f;

    // Create CSR arrays on device
    int*   dcsr_row_ptr;
    int*   dcsr_col_ind;
    float* dcsr_val;
    float* dB;
    float* dC;
    HIP_CHECK(hipMalloc(&dcsr_row_ptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc(&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc(&dcsr_val, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc(&dB, sizeof(float) * batch_count_B * k * n));
    HIP_CHECK(hipMalloc(&dC, sizeof(float) * batch_count_C * m * n));

    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dB, hB.data(), sizeof(float) * batch_count_B * k * n, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dC, hC.data(), sizeof(float) * batch_count_C * m * n, hipMemcpyHostToDevice));

    // Create rocsparse handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Types
    rocsparse_indextype itype = rocsparse_indextype_i32;
    rocsparse_indextype jtype = rocsparse_indextype_i32;
    rocsparse_datatype  ttype = rocsparse_datatype_f32_r;

    // Create descriptors
    rocsparse_spmat_descr mat_A;
    rocsparse_dnmat_descr mat_B;
    rocsparse_dnmat_descr mat_C;

    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&mat_A,
                                               m,
                                               k,
                                               nnz,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               dcsr_val,
                                               itype,
                                               jtype,
                                               rocsparse_index_base_zero,
                                               ttype));
    ROCSPARSE_CHECK(
        rocsparse_create_dnmat_descr(&mat_B, k, n, k, dB, ttype, rocsparse_order_column));
    ROCSPARSE_CHECK(
        rocsparse_create_dnmat_descr(&mat_C, m, n, m, dC, ttype, rocsparse_order_column));

    ROCSPARSE_CHECK(rocsparse_csr_set_strided_batch(
        mat_A, batch_count_A, offsets_batch_stride_A, columns_values_batch_stride_A));
    ROCSPARSE_CHECK(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B));
    ROCSPARSE_CHECK(rocsparse_dnmat_set_strided_batch(mat_C, batch_count_C, batch_stride_C));

    // Query SpMM buffer
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_spmm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   &alpha,
                                   mat_A,
                                   mat_B,
                                   &beta,
                                   mat_C,
                                   ttype,
                                   rocsparse_spmm_alg_default,
                                   rocsparse_spmm_stage_buffer_size,
                                   &buffer_size,
                                   nullptr));

    // Allocate buffer
    void* buffer;
    HIP_CHECK(hipMalloc(&buffer, buffer_size));

    ROCSPARSE_CHECK(rocsparse_spmm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   &alpha,
                                   mat_A,
                                   mat_B,
                                   &beta,
                                   mat_C,
                                   ttype,
                                   rocsparse_spmm_alg_default,
                                   rocsparse_spmm_stage_preprocess,
                                   &buffer_size,
                                   buffer));

    // Pointer mode host
    ROCSPARSE_CHECK(rocsparse_spmm(handle,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   &alpha,
                                   mat_A,
                                   mat_B,
                                   &beta,
                                   mat_C,
                                   ttype,
                                   rocsparse_spmm_alg_default,
                                   rocsparse_spmm_stage_compute,
                                   &buffer_size,
                                   buffer));

    // Clear up on device
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    HIP_CHECK(hipFree(buffer));

    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(mat_A));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(mat_B));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(mat_C));

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
//! [doc example]
