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
    // A = 1 2 0 0
    //     0 0 2 1
    //     0 0 1 2

    int mb        = 2;
    int nb        = 2;
    int nnzb      = 2;
    int block_dim = 2;

    double alpha = 1.0;

    std::vector<int>    hbsr_row_ptr = {0, 1, 2};
    std::vector<int>    hbsr_col_ind = {0, 1};
    std::vector<double> hbsr_val     = {2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0};

    std::vector<double> hx(mb * block_dim, 1.0);
    std::vector<double> hy(mb * block_dim);

    int*    dbsr_row_ptr;
    int*    dbsr_col_ind;
    double* dbsr_val;
    double* dx;
    double* dy;
    HIP_CHECK(hipMalloc(&dbsr_row_ptr, sizeof(int) * (mb + 1)));
    HIP_CHECK(hipMalloc(&dbsr_col_ind, sizeof(int) * nnzb));
    HIP_CHECK(hipMalloc(&dbsr_val, sizeof(double) * nnzb * block_dim * block_dim));
    HIP_CHECK(hipMalloc(&dx, sizeof(double) * mb * block_dim));
    HIP_CHECK(hipMalloc(&dy, sizeof(double) * mb * block_dim));

    HIP_CHECK(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr.data(), sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind.data(), sizeof(int) * nnzb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsr_val,
                        hbsr_val.data(),
                        sizeof(double) * nnzb * block_dim * block_dim,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(double) * mb * block_dim, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy.data(), sizeof(double) * mb * block_dim, hipMemcpyHostToDevice));

    // Create rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create matrix descriptor
    rocsparse_mat_descr descr;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower));
    ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit));

    // Create matrix info structure
    rocsparse_mat_info info;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info));

    // Obtain required buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_dbsrsv_buffer_size(handle,
                                                 rocsparse_direction_column,
                                                 rocsparse_operation_none,
                                                 mb,
                                                 nnzb,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 block_dim,
                                                 info,
                                                 &buffer_size));

    // Allocate temporary buffer
    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Perform analysis step
    ROCSPARSE_CHECK(rocsparse_dbsrsv_analysis(handle,
                                              rocsparse_direction_column,
                                              rocsparse_operation_none,
                                              mb,
                                              nnzb,
                                              descr,
                                              dbsr_val,
                                              dbsr_row_ptr,
                                              dbsr_col_ind,
                                              block_dim,
                                              info,
                                              rocsparse_analysis_policy_reuse,
                                              rocsparse_solve_policy_auto,
                                              temp_buffer));

    // Solve Ly = x
    ROCSPARSE_CHECK(rocsparse_dbsrsv_solve(handle,
                                           rocsparse_direction_column,
                                           rocsparse_operation_none,
                                           mb,
                                           nnzb,
                                           &alpha,
                                           descr,
                                           dbsr_val,
                                           dbsr_row_ptr,
                                           dbsr_col_ind,
                                           block_dim,
                                           info,
                                           dx,
                                           dy,
                                           rocsparse_solve_policy_auto,
                                           temp_buffer));

    HIP_CHECK(hipMemcpy(hy.data(), dy, sizeof(double) * mb * block_dim, hipMemcpyDeviceToHost));

    std::cout << "hy" << std::endl;
    for(size_t i = 0; i < hy.size(); i++)
    {
        std::cout << hy[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clean up
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    HIP_CHECK(hipFree(dbsr_row_ptr));
    HIP_CHECK(hipFree(dbsr_col_ind));
    HIP_CHECK(hipFree(dbsr_val));
    HIP_CHECK(hipFree(temp_buffer));

    return 0;
}
//! [doc example]
