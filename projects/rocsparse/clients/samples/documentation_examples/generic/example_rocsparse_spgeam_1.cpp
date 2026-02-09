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
    // A - m x n
    // B - m x n
    // C - m x n
    int m = 4;
    int n = 6;

    // 1 2 0 0 3 7
    // 0 0 1 4 6 8
    // 0 2 0 4 0 0
    // 9 8 0 0 2 0
    std::vector<int> hcsr_row_ptr_A = {0, 4, 8, 10, 13}; // host A m x n matrix
    std::vector<int> hcsr_col_ind_A
        = {0, 1, 4, 5, 2, 3, 4, 5, 1, 3, 0, 1, 4}; // host A m x n matrix
    std::vector<float> hcsr_val_A = {1, 2, 3, 7, 1, 4, 6, 8, 2, 4, 9, 8, 2}; // host A m x n matrix

    // 0 2 1 0 0 5
    // 0 1 1 3 0 2
    // 0 0 0 0 0 0
    // 1 2 3 4 5 6
    std::vector<int> hcsr_row_ptr_B = {0, 3, 7, 7, 13}; // host B m x n matrix
    std::vector<int> hcsr_col_ind_B
        = {1, 2, 5, 1, 2, 3, 5, 0, 1, 2, 3, 4, 5}; // host B m x n matrix
    std::vector<float> hcsr_val_B = {2, 1, 5, 1, 1, 3, 2, 1, 2, 3, 4, 5, 6}; // host B m x n matrix

    int nnz_A = hcsr_val_A.size();
    int nnz_B = hcsr_val_B.size();

    float alpha = 1.0f;
    float beta  = 1.0f;

    int*   dcsr_row_ptr_A;
    int*   dcsr_col_ind_A;
    float* dcsr_val_A;

    int*   dcsr_row_ptr_B;
    int*   dcsr_col_ind_B;
    float* dcsr_val_B;

    HIP_CHECK(hipMalloc(&dcsr_row_ptr_A, (m + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&dcsr_col_ind_A, nnz_A * sizeof(int)));
    HIP_CHECK(hipMalloc(&dcsr_val_A, nnz_A * sizeof(float)));

    HIP_CHECK(hipMalloc(&dcsr_row_ptr_B, (m + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&dcsr_col_ind_B, nnz_B * sizeof(int)));
    HIP_CHECK(hipMalloc(&dcsr_val_B, nnz_B * sizeof(float)));

    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcsr_col_ind_A, hcsr_col_ind_A.data(), nnz_A * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_val_A, hcsr_val_A.data(), nnz_A * sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr_B, hcsr_row_ptr_B.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcsr_col_ind_B, hcsr_col_ind_B.data(), nnz_B * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_val_B, hcsr_val_B.data(), nnz_B * sizeof(float), hipMemcpyHostToDevice));

    rocsparse_handle      handle;
    rocsparse_error       p_error[1] = {};
    rocsparse_spmat_descr matA, matB, matC;
    rocsparse_index_base  index_base = rocsparse_index_base_zero;
    rocsparse_indextype   itype      = rocsparse_indextype_i32;
    rocsparse_indextype   jtype      = rocsparse_indextype_i32;
    rocsparse_datatype    ttype      = rocsparse_datatype_f32_r;

    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    hipStream_t stream;
    ROCSPARSE_CHECK(rocsparse_get_stream(handle, &stream));

    // Create sparse matrix A in CSR format
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matA,
                                               m,
                                               n,
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
                                               m,
                                               n,
                                               nnz_B,
                                               dcsr_row_ptr_B,
                                               dcsr_col_ind_B,
                                               dcsr_val_B,
                                               itype,
                                               jtype,
                                               index_base,
                                               ttype));

    // Create SpGEAM descriptor.
    rocsparse_spgeam_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_spgeam_descr(&descr));

    // Set the algorithm on the descriptor
    const rocsparse_spgeam_alg alg = rocsparse_spgeam_alg_default;
    ROCSPARSE_CHECK(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_alg, &alg, sizeof(alg), p_error));

    // Set the transpose operation for sparses matrix A and B on the descriptor
    const rocsparse_operation trans_A = rocsparse_operation_none;
    const rocsparse_operation trans_B = rocsparse_operation_none;
    ROCSPARSE_CHECK(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_operation_A, &trans_A, sizeof(trans_A), p_error));
    ROCSPARSE_CHECK(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_operation_B, &trans_B, sizeof(trans_B), p_error));

    // Set the scalar type on the descriptor
    const rocsparse_datatype scalar_datatype = rocsparse_datatype_f32_r;
    ROCSPARSE_CHECK(rocsparse_spgeam_set_input(handle,
                                               descr,
                                               rocsparse_spgeam_input_scalar_datatype,
                                               &scalar_datatype,
                                               sizeof(scalar_datatype),
                                               p_error));

    // Set the compute type on the descriptor
    const rocsparse_datatype compute_datatype = rocsparse_datatype_f32_r;
    ROCSPARSE_CHECK(rocsparse_spgeam_set_input(handle,
                                               descr,
                                               rocsparse_spgeam_input_compute_datatype,
                                               &compute_datatype,
                                               sizeof(compute_datatype),
                                               p_error));

    // Calculate NNZ phase
    size_t buffer_size_in_bytes;
    void*  buffer;
    ROCSPARSE_CHECK(rocsparse_spgeam_buffer_size(handle,
                                                 descr,
                                                 matA,
                                                 matB,
                                                 nullptr,
                                                 rocsparse_spgeam_stage_analysis,
                                                 &buffer_size_in_bytes,
                                                 p_error));

    HIP_CHECK(hipMalloc(&buffer, buffer_size_in_bytes));
    ROCSPARSE_CHECK(rocsparse_spgeam(handle,
                                     descr,
                                     matA,
                                     matB,
                                     nullptr,
                                     rocsparse_spgeam_stage_analysis,
                                     buffer_size_in_bytes,
                                     buffer,
                                     p_error));
    HIP_CHECK(hipFree(buffer));

    // Ensure analysis stage is complete before grabbing C non-zero count
    HIP_CHECK(hipStreamSynchronize(stream));

    int64_t nnz_C;
    ROCSPARSE_CHECK(rocsparse_spgeam_get_output(
        handle, descr, rocsparse_spgeam_output_nnz, &nnz_C, sizeof(int64_t), p_error));

    // Compute column indices and values of C
    int*   dcsr_row_ptr_C;
    int*   dcsr_col_ind_C;
    float* dcsr_val_C;
    HIP_CHECK(hipMalloc(&dcsr_row_ptr_C, (m + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&dcsr_col_ind_C, sizeof(int32_t) * nnz_C));
    HIP_CHECK(hipMalloc(&dcsr_val_C, sizeof(float) * nnz_C));

    // Create sparse matrix C in CSR format
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matC,
                                               m,
                                               n,
                                               nnz_C,
                                               dcsr_row_ptr_C,
                                               dcsr_col_ind_C,
                                               dcsr_val_C,
                                               itype,
                                               jtype,
                                               index_base,
                                               ttype));

    // Compute phase
    ROCSPARSE_CHECK(rocsparse_spgeam_buffer_size(handle,
                                                 descr,
                                                 matA,
                                                 matB,
                                                 matC,
                                                 rocsparse_spgeam_stage_compute,
                                                 &buffer_size_in_bytes,
                                                 p_error));

    // Set alpha and beta
    ROCSPARSE_CHECK(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_scalar_alpha, &alpha, sizeof(&alpha), p_error));
    ROCSPARSE_CHECK(rocsparse_spgeam_set_input(
        handle, descr, rocsparse_spgeam_input_scalar_beta, &beta, sizeof(&beta), p_error));

    HIP_CHECK(hipMalloc(&buffer, buffer_size_in_bytes));
    ROCSPARSE_CHECK(rocsparse_spgeam(handle,
                                     descr,
                                     matA,
                                     matB,
                                     matC,
                                     rocsparse_spgeam_stage_compute,
                                     buffer_size_in_bytes,
                                     buffer,
                                     p_error));
    HIP_CHECK(hipFree(buffer));

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

    std::cout << "C" << std::endl;
    for(int i = 0; i < m; i++)
    {
        int start = hcsr_row_ptr_C[i];
        int end   = hcsr_row_ptr_C[i + 1];

        std::vector<float> htemp(n, 0.0f);
        for(int j = start; j < end; j++)
        {
            htemp[hcsr_col_ind_C[j]] = hcsr_val_C[j];
        }

        for(int j = 0; j < n; j++)
        {
            std::cout << htemp[j] << " ";
        }
        std::cout << "" << std::endl;
    }
    std::cout << "" << std::endl;

    // Destroy matrix descriptors
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matA));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matB));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matC));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_error(p_error[0]));

    // Free device arrays
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
