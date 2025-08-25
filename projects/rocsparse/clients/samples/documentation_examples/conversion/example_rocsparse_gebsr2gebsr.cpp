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
#include <vector>

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
    //     1 2 0 0 5 6
    // A = 3 4 0 0 7 8
    //     6 5 3 4 0 0
    //     1 2 5 4 0 0

    rocsparse_int mb_A            = 2;
    rocsparse_int nb_A            = 2;
    rocsparse_int nnzb_A          = 4;
    rocsparse_int row_block_dim_A = 2;
    rocsparse_int col_block_dim_A = 2;

    rocsparse_int m = mb_A * row_block_dim_A;
    rocsparse_int n = nb_A * col_block_dim_A;

    rocsparse_direction dir = rocsparse_direction_row;

    std::vector<rocsparse_int> hbsr_row_ptr_A = {0, 2, 4};
    std::vector<rocsparse_int> hbsr_col_ind_A = {0, 2, 0, 1};
    std::vector<float>         hbsr_val_A     = {1, 2, 3, 4, 5, 6, 7, 8, 6, 5, 1, 2, 3, 4, 5, 4};

    rocsparse_int* dbsr_row_ptr_A = nullptr;
    rocsparse_int* dbsr_col_ind_A = nullptr;
    float*         dbsr_val_A     = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbsr_row_ptr_A, sizeof(rocsparse_int) * (mb_A + 1)));
    HIP_CHECK(hipMalloc((void**)&dbsr_col_ind_A, sizeof(rocsparse_int) * nnzb_A));
    HIP_CHECK(
        hipMalloc((void**)&dbsr_val_A, sizeof(float) * nnzb_A * row_block_dim_A * col_block_dim_A));

    HIP_CHECK(hipMemcpy(dbsr_row_ptr_A,
                        hbsr_row_ptr_A.data(),
                        sizeof(rocsparse_int) * (mb_A + 1),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsr_col_ind_A,
                        hbsr_col_ind_A.data(),
                        sizeof(rocsparse_int) * nnzb_A,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsr_val_A,
                        hbsr_val_A.data(),
                        sizeof(float) * nnzb_A * row_block_dim_A * col_block_dim_A,
                        hipMemcpyHostToDevice));

    rocsparse_int row_block_dim_C = 2;
    rocsparse_int col_block_dim_C = 3;
    rocsparse_int mb_C            = (m + row_block_dim_C - 1) / row_block_dim_C;
    rocsparse_int nb_C            = (m + row_block_dim_C - 1) / row_block_dim_C;

    rocsparse_int* dbsr_row_ptr_C = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbsr_row_ptr_C, sizeof(rocsparse_int) * (mb_C + 1)));

    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    rocsparse_mat_descr descr_A;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_A));

    rocsparse_mat_descr descr_C;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_C));

    // Obtain the temporary buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_sgebsr2gebsr_buffer_size(handle,
                                                       dir,
                                                       mb_A,
                                                       nb_A,
                                                       nnzb_A,
                                                       descr_A,
                                                       dbsr_val_A,
                                                       dbsr_row_ptr_A,
                                                       dbsr_col_ind_A,
                                                       row_block_dim_A,
                                                       col_block_dim_A,
                                                       row_block_dim_C,
                                                       col_block_dim_C,
                                                       &buffer_size));

    // Allocate temporary buffer
    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    rocsparse_int nnzb_C;
    ROCSPARSE_CHECK(rocsparse_gebsr2gebsr_nnz(handle,
                                              dir,
                                              mb_A,
                                              nb_A,
                                              nnzb_A,
                                              descr_A,
                                              dbsr_row_ptr_A,
                                              dbsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              descr_C,
                                              dbsr_row_ptr_C,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &nnzb_C,
                                              temp_buffer));

    rocsparse_int* dbsr_col_ind_C = nullptr;
    float*         dbsr_val_C     = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbsr_col_ind_C, sizeof(rocsparse_int) * nnzb_C));
    HIP_CHECK(
        hipMalloc((void**)&dbsr_val_C, sizeof(float) * nnzb_C * row_block_dim_C * col_block_dim_C));

    ROCSPARSE_CHECK(rocsparse_sgebsr2gebsr(handle,
                                           dir,
                                           mb_A,
                                           nb_A,
                                           nnzb_A,
                                           descr_A,
                                           dbsr_val_A,
                                           dbsr_row_ptr_A,
                                           dbsr_col_ind_A,
                                           row_block_dim_A,
                                           col_block_dim_A,
                                           descr_C,
                                           dbsr_val_C,
                                           dbsr_row_ptr_C,
                                           dbsr_col_ind_C,
                                           row_block_dim_C,
                                           col_block_dim_C,
                                           temp_buffer));

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_A));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_C));

    HIP_CHECK(hipFree(temp_buffer));

    HIP_CHECK(hipFree(dbsr_row_ptr_A));
    HIP_CHECK(hipFree(dbsr_col_ind_A));
    HIP_CHECK(hipFree(dbsr_val_A));

    HIP_CHECK(hipFree(dbsr_row_ptr_C));
    HIP_CHECK(hipFree(dbsr_col_ind_C));
    HIP_CHECK(hipFree(dbsr_val_C));

    return 0;
}
//! [doc example]
