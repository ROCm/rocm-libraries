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

#include <cmath>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <rocsparse/rocsparse.h>
#include <stdio.h>
#include <stdlib.h>

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
    // Host problem definition
    int A_num_rows  = 4;
    int A_num_cols  = 4;
    int B_num_rows  = A_num_cols;
    int B_num_cols  = 3;
    int C_nnz       = 9;
    int lda         = A_num_cols;
    int ldb         = B_num_cols;
    int A_size      = lda * A_num_rows;
    int B_size      = ldb * B_num_rows;
    int num_batches = 2;

    float hA1[] = {1.0f,
                   2.0f,
                   3.0f,
                   4.0f,
                   5.0f,
                   6.0f,
                   7.0f,
                   8.0f,
                   9.0f,
                   10.0f,
                   11.0f,
                   12.0f,
                   13.0f,
                   14.0f,
                   15.0f,
                   16.0f};
    float hA2[] = {10.0f,
                   11.0f,
                   12.0f,
                   13.0f,
                   14.0f,
                   15.0f,
                   16.0f,
                   17.0f,
                   18.0f,
                   19.0f,
                   20.0f,
                   21.0f,
                   22.0f,
                   23.0f,
                   24.0f,
                   25.0f};
    float hB1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float hB2[] = {6.0f, 4.0f, 2.0f, 3.0f, 7.0f, 1.0f, 9.0f, 5.0f, 2.0f, 8.0f, 4.0f, 7.0f};
    int   hC_offsets[]  = {0, 3, 4, 7, 9};
    int   hC_columns1[] = {0, 1, 2, 1, 0, 1, 2, 0, 2};
    int   hC_columns2[] = {0, 1, 2, 0, 0, 1, 2, 1, 2};
    float hC_values1[]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float hC_values2[]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float hC_result1[]  = {70.0f, 80.0f, 90.0f, 184.0f, 246.0f, 288.0f, 330.0f, 334.0f, 450.0f};
    float hC_result2[]  = {305.0f, 229.0f, 146.0f, 409.0f, 513.0f, 389.0f, 242.0f, 469.0f, 290.0f};
    float alpha         = 1.0f;
    float beta          = 0.0f;

    // Device memory management
    int*   dC_offsets;
    int*   dC_columns;
    float* dC_values;
    float* dB;
    float* dA;
    HIP_CHECK(hipMalloc((void**)&dA, A_size * num_batches * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&dB, B_size * num_batches * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&dC_offsets, (A_num_rows + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dC_columns, C_nnz * num_batches * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&dC_values, C_nnz * num_batches * sizeof(float)));

    HIP_CHECK(hipMemcpy(dA, hA1, A_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dA + A_size, hA2, A_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB1, B_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB + B_size, hB2, B_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dC_offsets, hC_offsets, (A_num_rows + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC_columns, hC_columns1, C_nnz * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dC_columns + C_nnz, hC_columns2, C_nnz * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC_values, hC_values1, C_nnz * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dC_values + C_nnz, hC_values2, C_nnz * sizeof(float), hipMemcpyHostToDevice));

    // rocSPARSE APIs
    rocsparse_handle      handle = NULL;
    rocsparse_dnmat_descr matA, matB;
    rocsparse_spmat_descr matC;
    void*                 dBuffer    = NULL;
    size_t                bufferSize = 0;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create dense matrix A
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(
        &matA, A_num_rows, A_num_cols, lda, dA, rocsparse_datatype_f32_r, rocsparse_order_row));
    ROCSPARSE_CHECK(rocsparse_dnmat_set_strided_batch(matA, num_batches, A_size));

    // Create dense matrix B
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(
        &matB, A_num_cols, B_num_cols, ldb, dB, rocsparse_datatype_f32_r, rocsparse_order_row));
    ROCSPARSE_CHECK(rocsparse_dnmat_set_strided_batch(matB, num_batches, B_size));

    // Create sparse matrix C in CSR format
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matC,
                                               A_num_rows,
                                               B_num_cols,
                                               C_nnz,
                                               dC_offsets,
                                               dC_columns,
                                               dC_values,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r));
    ROCSPARSE_CHECK(rocsparse_csr_set_strided_batch(matC, num_batches, 0, C_nnz));

    // Allocate an external buffer if needed
    ROCSPARSE_CHECK(rocsparse_sddmm_buffer_size(handle,
                                                rocsparse_operation_none,
                                                rocsparse_operation_none,
                                                &alpha,
                                                matA,
                                                matB,
                                                &beta,
                                                matC,
                                                rocsparse_datatype_f32_r,
                                                rocsparse_sddmm_alg_default,
                                                &bufferSize));
    HIP_CHECK(hipMalloc(&dBuffer, bufferSize));

    // Execute preprocess (optional)
    ROCSPARSE_CHECK(rocsparse_sddmm_preprocess(handle,
                                               rocsparse_operation_none,
                                               rocsparse_operation_none,
                                               &alpha,
                                               matA,
                                               matB,
                                               &beta,
                                               matC,
                                               rocsparse_datatype_f32_r,
                                               rocsparse_sddmm_alg_default,
                                               dBuffer));

    // Execute SDDMM
    ROCSPARSE_CHECK(rocsparse_sddmm(handle,
                                    rocsparse_operation_none,
                                    rocsparse_operation_none,
                                    &alpha,
                                    matA,
                                    matB,
                                    &beta,
                                    matC,
                                    rocsparse_datatype_f32_r,
                                    rocsparse_sddmm_alg_default,
                                    dBuffer));

    // Destroy matrix/vector descriptors
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(matA));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(matB));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matC));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Device result check
    HIP_CHECK(hipMemcpy(hC_values1, dC_values, C_nnz * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(hC_values2, dC_values + C_nnz, C_nnz * sizeof(float), hipMemcpyDeviceToHost));
    int         correct    = 1;
    const float abs_tol    = 1.0e-4f;
    const float rel_tol    = 1.0e-4f;
    auto        within_tol = [&](float computed, float expected) {
        const float diff = std::fabs(computed - expected);
        return diff <= abs_tol + rel_tol * std::fabs(expected);
    };
    for(int i = 0; i < C_nnz; i++)
    {
        if(!within_tol(hC_values1[i], hC_result1[i]))
        {
            correct = 0;
            break;
        }
        if(!within_tol(hC_values2[i], hC_result2[i]))
        {
            correct = 0;
            break;
        }
    }
    if(correct)
        std::cout << "sddmm_csr_batched_example test PASSED" << std::endl;
    else
        std::cout << "sddmm_csr_batched_example test FAILED: wrong result" << std::endl;

    // Device memory deallocation
    HIP_CHECK(hipFree(dBuffer));
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC_offsets));
    HIP_CHECK(hipFree(dC_columns));
    HIP_CHECK(hipFree(dC_values));
    return 0;
}