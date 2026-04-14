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
    //     2 1 0 0
    // A = 1 2 1 0
    //     0 1 2 1
    //     0 0 1 2

    int m = 4;
    int n = 4;

    double alpha = 1.0;

    std::vector<int>    hcsr_row_ptr = {0, 2, 5, 8, 10};
    std::vector<int>    hcsr_col_ind = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> hcsr_val     = {2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0};

    int nnz = hcsr_row_ptr[m] - hcsr_row_ptr[0];

    std::vector<double> hx(m, 1.0);

    int*    dcsr_row_ptr;
    int*    dcsr_col_ind;
    double* dcsr_val;
    double* dx;
    double* dy;
    double* dz;
    HIP_CHECK(hipMalloc(&dcsr_row_ptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc(&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc(&dcsr_val, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc(&dx, sizeof(double) * m));
    HIP_CHECK(hipMalloc(&dy, sizeof(double) * m));
    HIP_CHECK(hipMalloc(&dz, sizeof(double) * m));

    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(double) * m, hipMemcpyHostToDevice));

    // Create rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create matrix descriptor for M
    rocsparse_mat_descr descr_M;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_M));

    // Create matrix descriptor for L
    rocsparse_mat_descr descr_L;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_L));
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr_L, rocsparse_fill_mode_lower));
    ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr_L, rocsparse_diag_type_non_unit));

    // Create matrix descriptor for L'
    rocsparse_mat_descr descr_Lt;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_Lt));
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr_Lt, rocsparse_fill_mode_lower));
    ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr_Lt, rocsparse_diag_type_non_unit));

    // Create matrix info structure
    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    // Obtain required buffer size
    size_t buffer_size_M;
    size_t buffer_size_L;
    size_t buffer_size_Lt;
    ROCSPARSE_CHECK(rocsparse_dcsric0_buffer_size(
        handle, m, nnz, descr_M, dcsr_val, dcsr_row_ptr, dcsr_col_ind, info, &buffer_size_M));
    ROCSPARSE_CHECK(rocsparse_dcsrsv_buffer_size(handle,
                                                 rocsparse_operation_none,
                                                 m,
                                                 nnz,
                                                 descr_L,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 info,
                                                 &buffer_size_L));
    ROCSPARSE_CHECK(rocsparse_dcsrsv_buffer_size(handle,
                                                 rocsparse_operation_transpose,
                                                 m,
                                                 nnz,
                                                 descr_Lt,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 info,
                                                 &buffer_size_Lt));

    size_t buffer_size = max(buffer_size_M, max(buffer_size_L, buffer_size_Lt));

    // Allocate temporary buffer
    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Perform analysis steps, using rocsparse_analysis_policy_reuse to improve
    // computation performance
    ROCSPARSE_CHECK(rocsparse_dcsric0_analysis(handle,
                                               m,
                                               nnz,
                                               descr_M,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               info,
                                               rocsparse_analysis_policy_reuse,
                                               rocsparse_solve_policy_auto,
                                               temp_buffer));
    ROCSPARSE_CHECK(rocsparse_dcsrsv_analysis(handle,
                                              rocsparse_operation_none,
                                              m,
                                              nnz,
                                              descr_L,
                                              dcsr_val,
                                              dcsr_row_ptr,
                                              dcsr_col_ind,
                                              info,
                                              rocsparse_analysis_policy_reuse,
                                              rocsparse_solve_policy_auto,
                                              temp_buffer));
    ROCSPARSE_CHECK(rocsparse_dcsrsv_analysis(handle,
                                              rocsparse_operation_transpose,
                                              m,
                                              nnz,
                                              descr_Lt,
                                              dcsr_val,
                                              dcsr_row_ptr,
                                              dcsr_col_ind,
                                              info,
                                              rocsparse_analysis_policy_reuse,
                                              rocsparse_solve_policy_auto,
                                              temp_buffer));

    // Check for zero pivot
    rocsparse_int position;
    if(rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(handle, info, &position))
    {
        printf("A has structural zero at A(%d,%d)\n", position, position);
    }

    // Compute incomplete Cholesky factorization M = LL'
    ROCSPARSE_CHECK(rocsparse_dcsric0(handle,
                                      m,
                                      nnz,
                                      descr_M,
                                      dcsr_val,
                                      dcsr_row_ptr,
                                      dcsr_col_ind,
                                      info,
                                      rocsparse_solve_policy_auto,
                                      temp_buffer));

    // Check for zero pivot
    if(rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(handle, info, &position))
    {
        printf("L has structural and/or numerical zero at L(%d,%d)\n", position, position);
    }

    // Copy incomplete LL^T factorization to host (note only lower L is stored and is written inplace into the original matrix)
    HIP_CHECK(
        hipMemcpy(hcsr_row_ptr.data(), dcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(hcsr_col_ind.data(), dcsr_col_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsr_val.data(), dcsr_val, sizeof(double) * nnz, hipMemcpyDeviceToHost));

    std::cout << "LL^T" << std::endl;
    for(int i = 0; i < m; i++)
    {
        int start = hcsr_row_ptr[i];
        int end   = hcsr_row_ptr[i + 1];

        std::vector<double> temp(n, 0.0);
        for(int j = start; j < end; j++)
        {
            temp[hcsr_col_ind[j]] = hcsr_val[j];
        }

        for(int j = 0; j < n; j++)
        {
            std::cout << temp[j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Solve Lz = x
    ROCSPARSE_CHECK(rocsparse_dcsrsv_solve(handle,
                                           rocsparse_operation_none,
                                           m,
                                           nnz,
                                           &alpha,
                                           descr_L,
                                           dcsr_val,
                                           dcsr_row_ptr,
                                           dcsr_col_ind,
                                           info,
                                           dx,
                                           dz,
                                           rocsparse_solve_policy_auto,
                                           temp_buffer));

    // Solve L'y = z
    ROCSPARSE_CHECK(rocsparse_dcsrsv_solve(handle,
                                           rocsparse_operation_transpose,
                                           m,
                                           nnz,
                                           &alpha,
                                           descr_Lt,
                                           dcsr_val,
                                           dcsr_row_ptr,
                                           dcsr_col_ind,
                                           info,
                                           dz,
                                           dy,
                                           rocsparse_solve_policy_auto,
                                           temp_buffer));

    std::vector<double> hy(m);
    HIP_CHECK(hipMemcpy(hy.data(), dy, sizeof(double) * m, hipMemcpyDeviceToHost));

    std::cout << "hy" << std::endl;
    for(size_t i = 0; i < hy.size(); i++)
    {
        std::cout << hy[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clean up
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_M));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_L));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_Lt));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));
    HIP_CHECK(hipFree(dz));
    HIP_CHECK(hipFree(temp_buffer));

    return 0;
}
//! [doc example]
