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
    //     1 2 3
    // A = 4 5 6
    //     7 8 9
    rocsparse_int m   = 3;
    rocsparse_int n   = 3;
    rocsparse_int nnz = 9;

    std::vector<rocsparse_int> hcoo_row_ind = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<rocsparse_int> hcoo_col_ind = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    std::vector<float>         hcoo_val     = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    rocsparse_int* dcoo_row_ind;
    rocsparse_int* dcoo_col_ind;
    float*         dcoo_val;
    HIP_CHECK(hipMalloc(&dcoo_row_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc(&dcoo_col_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc(&dcoo_val, sizeof(float) * nnz));

    HIP_CHECK(hipMemcpy(
        dcoo_row_ind, hcoo_row_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dcoo_col_ind, hcoo_col_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcoo_val, hcoo_val.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));

    // Create rocsparse handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create permutation vector perm as the identity map
    rocsparse_int* perm;
    HIP_CHECK(hipMalloc((void**)&perm, sizeof(rocsparse_int) * nnz));
    ROCSPARSE_CHECK(rocsparse_create_identity_permutation(handle, nnz, perm));

    // Allocate temporary buffer
    size_t buffer_size;
    void*  temp_buffer;
    ROCSPARSE_CHECK(
        rocsparse_coosort_buffer_size(handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, &buffer_size));
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Sort the COO matrix
    ROCSPARSE_CHECK(
        rocsparse_coosort_by_row(handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, perm, temp_buffer));

    // Gather sorted coo_val array
    float* dcoo_val_sorted;
    HIP_CHECK(hipMalloc((void**)&dcoo_val_sorted, sizeof(float) * nnz));
    ROCSPARSE_CHECK(
        rocsparse_sgthr(handle, nnz, dcoo_val, dcoo_val_sorted, perm, rocsparse_index_base_zero));

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clean up
    HIP_CHECK(hipFree(temp_buffer));
    HIP_CHECK(hipFree(perm));
    HIP_CHECK(hipFree(dcoo_row_ind));
    HIP_CHECK(hipFree(dcoo_col_ind));
    HIP_CHECK(hipFree(dcoo_val));
    HIP_CHECK(hipFree(dcoo_val_sorted));

    return 0;
}
//! [doc example]
