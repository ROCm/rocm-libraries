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
    //     1 0 0 0
    // A = 4 2 0 0
    //     0 3 7 0
    //     0 0 0 1
    int m = 4;
    int n = 2;

    std::vector<int>   hcsr_row_ptr = {0, 1, 3, 5, 6};
    std::vector<int>   hcsr_col_ind = {0, 0, 1, 1, 2, 3};
    std::vector<float> hcsr_val     = {1, 4, 2, 3, 7, 1};
    std::vector<float> hB(m * n);
    std::vector<float> hC(m * n);

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            hB[m * i + j] = static_cast<float>(i + 1);
        }
    }

    // Scalar alpha
    float alpha = 1.0f;

    int nnz = hcsr_row_ptr[m] - hcsr_row_ptr[0];

    // Offload data to device
    int*   dcsr_row_ptr;
    int*   dcsr_col_ind;
    float* dcsr_val;
    float* dB;
    float* dC;
    HIP_CHECK(hipMalloc(&dcsr_row_ptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc(&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc(&dcsr_val, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc(&dB, sizeof(float) * m * n));
    HIP_CHECK(hipMalloc(&dC, sizeof(float) * m * n));

    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), sizeof(float) * m * n, hipMemcpyHostToDevice));

    rocsparse_handle      handle;
    rocsparse_spmat_descr matA;
    rocsparse_dnmat_descr matB;
    rocsparse_dnmat_descr matC;

    rocsparse_indextype  row_idx_type = rocsparse_indextype_i32;
    rocsparse_indextype  col_idx_type = rocsparse_indextype_i32;
    rocsparse_datatype   data_type    = rocsparse_datatype_f32_r;
    rocsparse_datatype   compute_type = rocsparse_datatype_f32_r;
    rocsparse_index_base idx_base     = rocsparse_index_base_zero;
    rocsparse_operation  trans_A      = rocsparse_operation_none;
    rocsparse_operation  trans_B      = rocsparse_operation_none;

    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create sparse matrix A
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matA,
                                               m,
                                               m,
                                               nnz,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               dcsr_val,
                                               row_idx_type,
                                               col_idx_type,
                                               idx_base,
                                               data_type));

    // Create dense matrix B
    ROCSPARSE_CHECK(
        rocsparse_create_dnmat_descr(&matB, m, n, m, dB, data_type, rocsparse_order_column));

    // Create dense matrix C
    ROCSPARSE_CHECK(
        rocsparse_create_dnmat_descr(&matC, m, n, m, dC, data_type, rocsparse_order_column));

    // Call spsv to get buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_spsm(handle,
                                   trans_A,
                                   trans_B,
                                   &alpha,
                                   matA,
                                   matB,
                                   matC,
                                   compute_type,
                                   rocsparse_spsm_alg_default,
                                   rocsparse_spsm_stage_buffer_size,
                                   &buffer_size,
                                   nullptr));

    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Call spsv to perform analysis
    ROCSPARSE_CHECK(rocsparse_spsm(handle,
                                   trans_A,
                                   trans_B,
                                   &alpha,
                                   matA,
                                   matB,
                                   matC,
                                   compute_type,
                                   rocsparse_spsm_alg_default,
                                   rocsparse_spsm_stage_preprocess,
                                   &buffer_size,
                                   temp_buffer));

    // Call spsv to perform computation
    ROCSPARSE_CHECK(rocsparse_spsm(handle,
                                   trans_A,
                                   trans_B,
                                   &alpha,
                                   matA,
                                   matB,
                                   matC,
                                   compute_type,
                                   rocsparse_spsm_alg_default,
                                   rocsparse_spsm_stage_compute,
                                   &buffer_size,
                                   temp_buffer));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hC.data(), dC, sizeof(float) * m * n, hipMemcpyDeviceToHost));

    std::cout << "hC" << std::endl;
    for(size_t i = 0; i < hC.size(); ++i)
    {
        std::cout << hC[i] << " ";
    }
    std::cout << std::endl;

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matA));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(matB));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(matC));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    HIP_CHECK(hipFree(temp_buffer));

    return 0;
}
//! [doc example]
