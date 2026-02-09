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
    // A - m x k
    // B - k x n
    // C - m x n
    int m = 4;
    int n = 4;
    int k = 3;

    // A
    // 1 2 3
    // 0 1 0
    // 2 0 0
    // 0 0 3

    // B
    // 0 1 2 0
    // 0 0 0 1
    // 1 2 3 4

    std::vector<int>   hcsr_row_ptr_A = {0, 3, 4, 5};
    std::vector<int>   hcsr_col_ind_A = {0, 1, 2, 1, 0, 2};
    std::vector<float> hcsr_val_A     = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};

    std::vector<int>   hcsr_row_ptr_B = {0, 2, 3, 7};
    std::vector<int>   hcsr_col_ind_B = {1, 2, 3, 0, 1, 2, 3};
    std::vector<float> hcsr_val_B     = {1.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f};

    int nnz_A = hcsr_val_A.size();
    int nnz_B = hcsr_val_B.size();

    float alpha = 1.0f;
    float beta  = 0.0f;

    int*   dcsr_row_ptr_A;
    int*   dcsr_col_ind_A;
    float* dcsr_val_A;

    int*   dcsr_row_ptr_B;
    int*   dcsr_col_ind_B;
    float* dcsr_val_B;

    int* dcsr_row_ptr_C;

    HIP_CHECK(hipMalloc(&dcsr_row_ptr_A, (m + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&dcsr_col_ind_A, nnz_A * sizeof(int)));
    HIP_CHECK(hipMalloc(&dcsr_val_A, nnz_A * sizeof(float)));

    HIP_CHECK(hipMalloc(&dcsr_row_ptr_B, (k + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&dcsr_col_ind_B, nnz_B * sizeof(int)));
    HIP_CHECK(hipMalloc(&dcsr_val_B, nnz_B * sizeof(float)));

    HIP_CHECK(hipMalloc(&dcsr_row_ptr_C, (m + 1) * sizeof(int)));

    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcsr_col_ind_A, hcsr_col_ind_A.data(), nnz_A * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_val_A, hcsr_val_A.data(), nnz_A * sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr_B, hcsr_row_ptr_B.data(), (k + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcsr_col_ind_B, hcsr_col_ind_B.data(), nnz_B * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_val_B, hcsr_val_B.data(), nnz_B * sizeof(float), hipMemcpyHostToDevice));

    rocsparse_handle      handle;
    rocsparse_spmat_descr matA, matB, matC, matD;
    void*                 temp_buffer;
    size_t                buffer_size = 0;

    rocsparse_operation  trans_A    = rocsparse_operation_none;
    rocsparse_operation  trans_B    = rocsparse_operation_none;
    rocsparse_index_base index_base = rocsparse_index_base_zero;
    rocsparse_indextype  itype      = rocsparse_indextype_i32;
    rocsparse_indextype  jtype      = rocsparse_indextype_i32;
    rocsparse_datatype   ttype      = rocsparse_datatype_f32_r;

    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create sparse matrix A in CSR format
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matA,
                                               m,
                                               k,
                                               nnz_A,
                                               dcsr_row_ptr_A,
                                               dcsr_col_ind_A,
                                               dcsr_val_A,
                                               itype,
                                               jtype,
                                               index_base,
                                               ttype));

    // Create sparse matrix B in CSR format
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matB,
                                               k,
                                               n,
                                               nnz_B,
                                               dcsr_row_ptr_B,
                                               dcsr_col_ind_B,
                                               dcsr_val_B,
                                               itype,
                                               jtype,
                                               index_base,
                                               ttype));

    // Create sparse matrix C in CSR format
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &matC, m, n, 0, dcsr_row_ptr_C, nullptr, nullptr, itype, jtype, index_base, ttype));

    // Create sparse matrix D in CSR format
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &matD, 0, 0, 0, nullptr, nullptr, nullptr, itype, jtype, index_base, ttype));

    // Determine buffer size
    ROCSPARSE_CHECK(rocsparse_spgemm(handle,
                                     trans_A,
                                     trans_B,
                                     &alpha,
                                     matA,
                                     matB,
                                     &beta,
                                     matD,
                                     matC,
                                     ttype,
                                     rocsparse_spgemm_alg_default,
                                     rocsparse_spgemm_stage_buffer_size,
                                     &buffer_size,
                                     nullptr));

    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Determine number of non-zeros in C matrix
    ROCSPARSE_CHECK(rocsparse_spgemm(handle,
                                     trans_A,
                                     trans_B,
                                     &alpha,
                                     matA,
                                     matB,
                                     &beta,
                                     matD,
                                     matC,
                                     ttype,
                                     rocsparse_spgemm_alg_default,
                                     rocsparse_spgemm_stage_nnz,
                                     &buffer_size,
                                     temp_buffer));

    int64_t rows_C;
    int64_t cols_C;
    int64_t nnz_C;

    // Extract number of non-zeros in C matrix so we can allocate the column indices and values arrays
    ROCSPARSE_CHECK(rocsparse_spmat_get_size(matC, &rows_C, &cols_C, &nnz_C));

    std::cout << "rows_C: " << rows_C << " cols_C: " << cols_C << " nnz_C: " << nnz_C << std::endl;
    int*   dcsr_col_ind_C;
    float* dcsr_val_C;
    HIP_CHECK(hipMalloc(&dcsr_col_ind_C, sizeof(int) * nnz_C));
    HIP_CHECK(hipMalloc(&dcsr_val_C, sizeof(float) * nnz_C));

    // Set C matrix pointers
    ROCSPARSE_CHECK(rocsparse_csr_set_pointers(matC, dcsr_row_ptr_C, dcsr_col_ind_C, dcsr_val_C));

    // SpGEMM computation
    ROCSPARSE_CHECK(rocsparse_spgemm(handle,
                                     trans_A,
                                     trans_B,
                                     &alpha,
                                     matA,
                                     matB,
                                     &beta,
                                     matD,
                                     matC,
                                     ttype,
                                     rocsparse_spgemm_alg_default,
                                     rocsparse_spgemm_stage_compute,
                                     &buffer_size,
                                     temp_buffer));

    // Copy C matrix result back to host
    std::vector<int>   hcsr_row_ptr_C(m + 1);
    std::vector<int>   hcsr_col_ind_C(nnz_C);
    std::vector<float> hcsr_val_C(nnz_C);

    HIP_CHECK(hipMemcpy(
        hcsr_row_ptr_C.data(), dcsr_row_ptr_C, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        hcsr_col_ind_C.data(), dcsr_col_ind_C, sizeof(int) * nnz_C, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(hcsr_val_C.data(), dcsr_val_C, sizeof(float) * nnz_C, hipMemcpyDeviceToHost));

    // Destroy matrix descriptors
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matA));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matB));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matC));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matD));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Free device arrays
    HIP_CHECK(hipFree(temp_buffer));
    HIP_CHECK(hipFree(dcsr_row_ptr_A));
    HIP_CHECK(hipFree(dcsr_col_ind_A));
    HIP_CHECK(hipFree(dcsr_val_A));

    HIP_CHECK(hipFree(dcsr_row_ptr_B));
    HIP_CHECK(hipFree(dcsr_col_ind_B));
    HIP_CHECK(hipFree(dcsr_val_B));

    HIP_CHECK(hipFree(dcsr_row_ptr_C));
    HIP_CHECK(hipFree(dcsr_col_ind_C));
    HIP_CHECK(hipFree(dcsr_val_C));

    return 0;
}
//! [doc example]
