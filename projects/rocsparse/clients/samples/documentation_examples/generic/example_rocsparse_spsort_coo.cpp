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
    //     1 2 3 4 0 5
    // A = 0 2 3 0 0 4
    //     5 0 6 7 8 0
    //     1 0 9 0 6 7
    int m   = 4;
    int n   = 6;
    int nnz = 16;

    std::vector<int>   hcoo_row_ind = {0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    std::vector<int>   hcoo_col_ind = {0, 1, 2, 3, 5, 1, 2, 5, 0, 2, 3, 4, 0, 2, 4, 5};
    std::vector<float> hcoo_val     = {1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7, 8, 1, 9, 6, 7};

    // Reference solution
    std::vector<int>   hcoo_row_ind_gold = {0, 2, 3, 0, 1, 0, 1, 2, 3, 0, 2, 2, 3, 0, 1, 3};
    std::vector<int>   hcoo_col_ind_gold = {0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5};
    std::vector<float> hcoo_val_gold     = {1, 5, 1, 2, 2, 3, 3, 6, 9, 4, 7, 8, 6, 5, 4, 7};

    // Offload data to device
    int*   dcoo_row_ind;
    int*   dcoo_col_ind;
    float* dcoo_val;
    HIP_CHECK(hipMalloc(&dcoo_row_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc(&dcoo_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc(&dcoo_val, sizeof(float) * nnz));

    HIP_CHECK(
        hipMemcpy(dcoo_row_ind, hcoo_row_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcoo_col_ind, hcoo_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcoo_val, hcoo_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));

    // Output arrays for the sorted matrix B
    int*   dcoo_row_ind_B;
    int*   dcoo_col_ind_B;
    float* dcoo_val_B;
    HIP_CHECK(hipMalloc(&dcoo_row_ind_B, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc(&dcoo_col_ind_B, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc(&dcoo_val_B, sizeof(float) * nnz));

    rocsparse_handle      handle;
    rocsparse_error       p_error[1] = {};
    rocsparse_spmat_descr matA;
    rocsparse_spmat_descr matB;

    rocsparse_indextype  idx_type  = rocsparse_indextype_i32;
    rocsparse_datatype   data_type = rocsparse_datatype_f32_r;
    rocsparse_index_base idx_base  = rocsparse_index_base_zero;

    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create sparse matrix A
    ROCSPARSE_CHECK(rocsparse_create_coo_descr(
        &matA, m, n, nnz, dcoo_row_ind, dcoo_col_ind, dcoo_val, idx_type, idx_base, data_type));

    // Create sparse matrix B, which receives the sorted matrix A. Passing matA as the
    // output instead would sort A in place.
    ROCSPARSE_CHECK(rocsparse_create_coo_descr(&matB,
                                               m,
                                               n,
                                               nnz,
                                               dcoo_row_ind_B,
                                               dcoo_col_ind_B,
                                               dcoo_val_B,
                                               idx_type,
                                               idx_base,
                                               data_type));

    rocsparse_spsort_descr spsort_descr;
    ROCSPARSE_CHECK(rocsparse_spsort_descr_create(handle, &spsort_descr, p_error));

    const rocsparse_spsort_alg spsort_alg = rocsparse_spsort_alg_default;
    ROCSPARSE_CHECK(rocsparse_spsort_set_input(handle,
                                               spsort_descr,
                                               rocsparse_spsort_input_alg,
                                               &spsort_alg,
                                               sizeof(spsort_alg),
                                               p_error));

    const rocsparse_direction spsort_sort_direction = rocsparse_direction_column;
    ROCSPARSE_CHECK(rocsparse_spsort_set_input(handle,
                                               spsort_descr,
                                               rocsparse_spsort_input_direction,
                                               &spsort_sort_direction,
                                               sizeof(spsort_sort_direction),
                                               p_error));

    // Call spsort to get buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_spsort_buffer_size(
        handle, spsort_descr, matA, matB, rocsparse_spsort_stage_analysis, &buffer_size, p_error));

    std::cout << "buffer_size: " << buffer_size << std::endl;

    void* buffer;
    HIP_CHECK(hipMalloc(&buffer, buffer_size));

    // Call spsort to perform analysis
    ROCSPARSE_CHECK(rocsparse_spsort(handle,
                                     spsort_descr,
                                     matA,
                                     matB,
                                     rocsparse_spsort_stage_analysis,
                                     buffer_size,
                                     buffer,
                                     p_error));

    HIP_CHECK(hipFree(buffer));

    ROCSPARSE_CHECK(rocsparse_spsort_buffer_size(
        handle, spsort_descr, matA, matB, rocsparse_spsort_stage_compute, &buffer_size, p_error));

    std::cout << "buffer_size: " << buffer_size << std::endl;

    HIP_CHECK(hipMalloc(&buffer, buffer_size));

    // Call spsort to perform computation
    ROCSPARSE_CHECK(rocsparse_spsort(handle,
                                     spsort_descr,
                                     matA,
                                     matB,
                                     rocsparse_spsort_stage_compute,
                                     buffer_size,
                                     buffer,
                                     p_error));

    HIP_CHECK(
        hipMemcpy(hcoo_row_ind.data(), dcoo_row_ind_B, sizeof(int) * nnz, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(hcoo_col_ind.data(), dcoo_col_ind_B, sizeof(int) * nnz, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcoo_val.data(), dcoo_val_B, sizeof(float) * nnz, hipMemcpyDeviceToHost));

    std::cout << "hcoo_row_ind" << std::endl;
    for(size_t i = 0; i < hcoo_row_ind.size(); i++)
    {
        std::cout << hcoo_row_ind[i] << " ";
    }
    std::cout << "" << std::endl;

    std::cout << "hcoo_col_ind" << std::endl;
    for(size_t i = 0; i < hcoo_col_ind.size(); i++)
    {
        std::cout << hcoo_col_ind[i] << " ";
    }
    std::cout << "" << std::endl;

    std::cout << "hcoo_val" << std::endl;
    for(size_t i = 0; i < hcoo_val.size(); i++)
    {
        std::cout << hcoo_val[i] << " ";
    }
    std::cout << "" << std::endl;

    bool pass = true;
    for(size_t i = 0; i < hcoo_row_ind.size(); i++)
    {
        if(hcoo_row_ind[i] != hcoo_row_ind_gold[i])
        {
            pass = false;
            break;
        }
    }
    for(size_t i = 0; i < hcoo_col_ind.size(); i++)
    {
        if(hcoo_col_ind[i] != hcoo_col_ind_gold[i])
        {
            pass = false;
            break;
        }
    }
    for(size_t i = 0; i < hcoo_val.size(); i++)
    {
        if(hcoo_val[i] != hcoo_val_gold[i])
        {
            pass = false;
            break;
        }
    }

    if(pass)
    {
        std::cout << "PASS" << std::endl;
    }
    else
    {
        std::cout << "FAIL" << std::endl;
    }

    HIP_CHECK(hipFree(buffer));

    ROCSPARSE_CHECK(rocsparse_destroy_error(p_error[0]));
    ROCSPARSE_CHECK(rocsparse_spsort_descr_destroy(handle, spsort_descr, nullptr));

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matA));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matB));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dcoo_row_ind));
    HIP_CHECK(hipFree(dcoo_col_ind));
    HIP_CHECK(hipFree(dcoo_val));
    HIP_CHECK(hipFree(dcoo_row_ind_B));
    HIP_CHECK(hipFree(dcoo_col_ind_B));
    HIP_CHECK(hipFree(dcoo_val_B));

    return 0;
}
//! [doc example]