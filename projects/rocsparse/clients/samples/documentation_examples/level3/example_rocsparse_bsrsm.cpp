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
    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // A = ( 1.0  0.0  0.0  0.0 )
    //     ( 2.0  3.0  0.0  0.0 )
    //     ( 4.0  5.0  6.0  0.0 )
    //     ( 7.0  0.0  8.0  9.0 )
    //
    // with bsr_dim = 2
    //
    //      -------------------
    //   = | 1.0 0.0 | 0.0 0.0 |
    //     | 2.0 3.0 | 0.0 0.0 |
    //      -------------------
    //     | 4.0 5.0 | 6.0 0.0 |
    //     | 7.0 0.0 | 8.0 9.0 |
    //      -------------------

    // Number of rows and columns
    rocsparse_int m = 4;

    // Number of block rows and block columns
    rocsparse_int mb = 2;
    rocsparse_int nb = 2;

    // BSR block dimension
    rocsparse_int bsr_dim = 2;

    // Number of right-hand-sides
    rocsparse_int nrhs = 4;

    // Number of non-zero blocks
    rocsparse_int nnzb = 3;

    // BSR row pointers
    rocsparse_int hbsr_row_ptr[3] = {0, 1, 3};

    // BSR column indices
    rocsparse_int hbsr_col_ind[3] = {0, 0, 1};

    // BSR values
    double hbsr_val[12] = {1.0, 2.0, 0.0, 3.0, 4.0, 7.0, 5.0, 0.0, 6.0, 8.0, 0.0, 9.0};

    // Storage scheme of the BSR blocks
    rocsparse_direction dir = rocsparse_direction_column;

    // Transposition of the matrix and rhs matrix
    rocsparse_operation transA = rocsparse_operation_none;
    rocsparse_operation transX = rocsparse_operation_none;

    // Analysis policy
    rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;

    // Solve policy
    rocsparse_solve_policy solve_policy = rocsparse_solve_policy_auto;

    // Scalar alpha and beta
    double alpha = 3.7;

    // rhs and solution matrix
    rocsparse_int ldb = nb * bsr_dim;
    rocsparse_int ldx = mb * bsr_dim;

    double hB[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    double hX[16];

    // Offload data to device
    rocsparse_int* dbsr_row_ptr;
    rocsparse_int* dbsr_col_ind;
    double*        dbsr_val;
    double*        dB;
    double*        dX;

    HIP_CHECK(hipMalloc(&dbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1)));
    HIP_CHECK(hipMalloc(&dbsr_col_ind, sizeof(rocsparse_int) * nnzb));
    HIP_CHECK(hipMalloc(&dbsr_val, sizeof(double) * nnzb * bsr_dim * bsr_dim));
    HIP_CHECK(hipMalloc(&dB, sizeof(double) * nb * bsr_dim * nrhs));
    HIP_CHECK(hipMalloc(&dX, sizeof(double) * mb * bsr_dim * nrhs));

    HIP_CHECK(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dbsr_val, hbsr_val, sizeof(double) * nnzb * bsr_dim * bsr_dim, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, sizeof(double) * nb * bsr_dim * nrhs, hipMemcpyHostToDevice));

    // Matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Matrix fill mode
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower));

    // Matrix diagonal type
    ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit));

    // Matrix info structure
    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    // Obtain required buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_dbsrsm_buffer_size(handle,
                                                 dir,
                                                 transA,
                                                 transX,
                                                 mb,
                                                 nrhs,
                                                 nnzb,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 bsr_dim,
                                                 info,
                                                 &buffer_size));

    // Allocate temporary buffer
    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Perform analysis step
    ROCSPARSE_CHECK(rocsparse_dbsrsm_analysis(handle,
                                              dir,
                                              transA,
                                              transX,
                                              mb,
                                              nrhs,
                                              nnzb,
                                              descr,
                                              dbsr_val,
                                              dbsr_row_ptr,
                                              dbsr_col_ind,
                                              bsr_dim,
                                              info,
                                              analysis_policy,
                                              solve_policy,
                                              temp_buffer));

    // Call dbsrsm to perform lower triangular solve LX = B
    ROCSPARSE_CHECK(rocsparse_dbsrsm_solve(handle,
                                           dir,
                                           transA,
                                           transX,
                                           mb,
                                           nrhs,
                                           nnzb,
                                           &alpha,
                                           descr,
                                           dbsr_val,
                                           dbsr_row_ptr,
                                           dbsr_col_ind,
                                           bsr_dim,
                                           info,
                                           dB,
                                           ldb,
                                           dX,
                                           ldx,
                                           solve_policy,
                                           temp_buffer));

    // Check for zero pivots
    rocsparse_int    pivot;
    rocsparse_status status = rocsparse_bsrsm_zero_pivot(handle, info, &pivot);

    if(status == rocsparse_status_zero_pivot)
    {
        std::cout << "Found zero pivot in matrix row " << pivot << std::endl;
    }

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hX, dX, sizeof(double) * mb * bsr_dim * nrhs, hipMemcpyDeviceToHost));

    std::cout << "hX" << std::endl;
    for(rocsparse_int i = 0; i < mb * bsr_dim * nrhs; i++)
    {
        std::cout << hX[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dbsr_row_ptr));
    HIP_CHECK(hipFree(dbsr_col_ind));
    HIP_CHECK(hipFree(dbsr_val));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dX));
    HIP_CHECK(hipFree(temp_buffer));

    return 0;
}
//! [doc example]
