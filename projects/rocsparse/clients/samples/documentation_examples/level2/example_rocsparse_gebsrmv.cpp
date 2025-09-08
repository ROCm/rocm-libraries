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
    // alpha * ( 1.0  0.0  2.0 ) * ( 1.0 ) + beta * ( 4.0 ) = (  31.1 )
    //         ( 3.0  0.0  4.0 ) * ( 2.0 )          ( 5.0 ) = (  62.0 )
    //         ( 5.0  6.0  0.0 ) * ( 3.0 )          ( 6.0 ) = (  70.7 )
    //         ( 7.0  0.0  8.0 ) *                  ( 7.0 ) = ( 123.8 )

    // GEBSR block dimensions
    rocsparse_int row_block_dim = 2;
    rocsparse_int col_block_dim = 3;

    // Number of block rows and columns
    rocsparse_int mb = 2;
    rocsparse_int nb = 1;

    // Number of non-zero blocks
    rocsparse_int nnzb = 2;

    // BSR row pointers
    rocsparse_int hbsr_row_ptr[3] = {0, 1, 2};

    // BSR column indices
    rocsparse_int hbsr_col_ind[2] = {0, 0};

    // BSR values
    double hbsr_val[16] = {1.0, 3.0, 0.0, 0.0, 2.0, 4.0, 5.0, 7.0, 6.0, 0.0, 0.0, 8.0};

    // Block storage in column major
    rocsparse_direction dir = rocsparse_direction_column;

    // Transposition of the matrix
    rocsparse_operation trans = rocsparse_operation_none;

    // Scalar alpha and beta
    double alpha = 3.7;
    double beta  = 1.3;

    // x and y
    double hx[4] = {1.0, 2.0, 3.0, 0.0};
    double hy[4] = {4.0, 5.0, 6.0, 7.0};

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

    // Offload data to device
    rocsparse_int* dbsr_row_ptr;
    rocsparse_int* dbsr_col_ind;
    double*        dbsr_val;
    double*        dx;
    double*        dy;

    HIP_CHECK(hipMalloc(&dbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1)));
    HIP_CHECK(hipMalloc(&dbsr_col_ind, sizeof(rocsparse_int) * nnzb));
    HIP_CHECK(hipMalloc(&dbsr_val, sizeof(double) * nnzb * row_block_dim * col_block_dim));
    HIP_CHECK(hipMalloc(&dx, sizeof(double) * nb * col_block_dim));
    HIP_CHECK(hipMalloc(&dy, sizeof(double) * mb * row_block_dim));

    HIP_CHECK(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsr_val,
                        hbsr_val,
                        sizeof(double) * nnzb * row_block_dim * col_block_dim,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx, sizeof(double) * nb * col_block_dim, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy, sizeof(double) * mb * row_block_dim, hipMemcpyHostToDevice));

    // Call dbsrmv to perform y = alpha * A x + beta * y
    ROCSPARSE_CHECK(rocsparse_dgebsrmv(handle,
                                       dir,
                                       trans,
                                       mb,
                                       nb,
                                       nnzb,
                                       &alpha,
                                       descr,
                                       dbsr_val,
                                       dbsr_row_ptr,
                                       dbsr_col_ind,
                                       row_block_dim,
                                       col_block_dim,
                                       dx,
                                       &beta,
                                       dy));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hy, dy, sizeof(double) * mb * row_block_dim, hipMemcpyDeviceToHost));

    std::cout << "hy" << std::endl;
    for(size_t i = 0; i < mb * row_block_dim; i++)
    {
        std::cout << hy[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dbsr_row_ptr));
    HIP_CHECK(hipFree(dbsr_col_ind));
    HIP_CHECK(hipFree(dbsr_val));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));

    return 0;
}
//! [doc example]
