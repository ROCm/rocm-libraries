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
    // A = ( 1.0  0.0  0.0  0.0 )
    //     ( 2.0  3.0  0.0  0.0 )
    //     ( 4.0  5.0  6.0  0.0 )
    //     ( 7.0  0.0  8.0  9.0 )

    // Number of rows and columns
    rocsparse_int m = 4;

    // Number of right-hand-sides
    rocsparse_int nrhs = 4;

    // Number of non-zero blocks
    rocsparse_int nnz = 9;

    // CSR row pointers
    std::vector<rocsparse_int> hcsr_row_ptr = {0, 1, 3, 6, 9};

    // CSR column indices
    std::vector<rocsparse_int> hcsr_col_ind = {0, 0, 1, 0, 1, 2, 0, 2, 3};

    // CSR values
    std::vector<double> hcsr_val = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    // Transposition of the matrix and rhs matrix
    rocsparse_operation transA = rocsparse_operation_none;
    rocsparse_operation transB = rocsparse_operation_none;

    // Analysis policy
    rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;

    // Solve policy
    rocsparse_solve_policy solve_policy = rocsparse_solve_policy_auto;

    rocsparse_int ldb = m;

    // Scalar alpha and beta
    double alpha = 3.7;

    std::vector<double> hB = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    // Offload data to device
    rocsparse_int* dcsr_row_ptr;
    rocsparse_int* dcsr_col_ind;
    double*        dcsr_val;
    double*        dB;

    HIP_CHECK(hipMalloc(&dcsr_row_ptr, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc(&dcsr_col_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc(&dcsr_val, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc(&dB, sizeof(double) * ldb * nrhs));

    HIP_CHECK(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), sizeof(double) * ldb * nrhs, hipMemcpyHostToDevice));

    // Create rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower));
    ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_unit));

    // Create matrix info structure
    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    // Obtain required buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_dcsrsm_buffer_size(handle,
                                                 transA,
                                                 transB,
                                                 m,
                                                 nrhs,
                                                 nnz,
                                                 &alpha,
                                                 descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 dB,
                                                 ldb,
                                                 info,
                                                 rocsparse_solve_policy_auto,
                                                 &buffer_size));

    // Allocate temporary buffer
    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Perform analysis step
    ROCSPARSE_CHECK(rocsparse_dcsrsm_analysis(handle,
                                              transA,
                                              transB,
                                              m,
                                              nrhs,
                                              nnz,
                                              &alpha,
                                              descr,
                                              dcsr_val,
                                              dcsr_row_ptr,
                                              dcsr_col_ind,
                                              dB,
                                              ldb,
                                              info,
                                              rocsparse_analysis_policy_reuse,
                                              rocsparse_solve_policy_auto,
                                              temp_buffer));

    // Solve LX = B
    ROCSPARSE_CHECK(rocsparse_dcsrsm_solve(handle,
                                           transA,
                                           transB,
                                           m,
                                           nrhs,
                                           nnz,
                                           &alpha,
                                           descr,
                                           dcsr_val,
                                           dcsr_row_ptr,
                                           dcsr_col_ind,
                                           dB,
                                           ldb,
                                           info,
                                           rocsparse_solve_policy_auto,
                                           temp_buffer));

    // No zero pivot should be found, with L having unit diagonal
    // Copy result back to host
    HIP_CHECK(hipMemcpy(hB.data(), dB, sizeof(double) * ldb * nrhs, hipMemcpyDeviceToHost));

    std::cout << "hB" << std::endl;
    for(size_t i = 0; i < hB.size(); i++)
    {
        std::cout << hB[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clean up
    HIP_CHECK(hipFree(temp_buffer));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    return 0;
}
//! [doc example]
