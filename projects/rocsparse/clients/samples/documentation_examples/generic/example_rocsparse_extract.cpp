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
    // 1 2 3 0 0 0 4 5
    // 0 1 3 5 7 0 0 0
    // 0 0 0 1 0 3 0 9
    // 1 2 3 0 0 0 0 4
    // 0 0 0 0 0 0 0 0
    // 1 2 1 0 0 5 8 0
    // 0 1 2 3 0 0 0 4
    // 0 0 0 1 2 0 1 2
    int32_t M   = 8;
    int32_t N   = 8;
    int32_t nnz = 29;

    std::vector<int32_t> hsource_row_ptr = {0, 5, 9, 12, 16, 16, 21, 25, 29};
    std::vector<int32_t> hsource_col_ind
        = {0, 1, 2, 6, 7, 1, 2, 3, 4, 3, 5, 7, 0, 1, 2, 7, 0, 1, 2, 5, 6, 1, 2, 3, 7, 3, 4, 6, 7};
    std::vector<float> hsource_val
        = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 3.0f, 5.0f, 7.0f, 1.0f, 3.0f, 9.0f, 1.0f, 2.0f, 3.0f,
           4.0f, 1.0f, 2.0f, 1.0f, 5.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 1.0f, 2.0f};

    int32_t* dsource_row_ptr;
    int32_t* dsource_col_ind;
    float*   dsource_val;
    HIP_CHECK(hipMalloc(&dsource_row_ptr, sizeof(int32_t) * (M + 1)));
    HIP_CHECK(hipMalloc(&dsource_col_ind, sizeof(int32_t) * nnz));
    HIP_CHECK(hipMalloc(&dsource_val, sizeof(float) * nnz));

    HIP_CHECK(hipMemcpy(
        dsource_row_ptr, hsource_row_ptr.data(), sizeof(int32_t) * (M + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dsource_col_ind, hsource_col_ind.data(), sizeof(int32_t) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dsource_val, hsource_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));

    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Build Source
    rocsparse_spmat_descr source;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&source,
                                               M,
                                               N,
                                               nnz,
                                               dsource_row_ptr,
                                               dsource_col_ind,
                                               dsource_val,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r));

    // Build target
    void* dtarget_row_ptr;
    HIP_CHECK(hipMalloc(&dtarget_row_ptr, sizeof(int32_t) * (M + 1)));
    rocsparse_spmat_descr target;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&target,
                                               M,
                                               N,
                                               0,
                                               dtarget_row_ptr,
                                               nullptr,
                                               nullptr,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r));

    const rocsparse_fill_mode fill_mode = rocsparse_fill_mode_lower;
    const rocsparse_diag_type diag_type = rocsparse_diag_type_non_unit;

    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        target, rocsparse_spmat_fill_mode, &fill_mode, sizeof(fill_mode)));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        target, rocsparse_spmat_diag_type, &diag_type, sizeof(diag_type)));

    // Create descriptor
    rocsparse_extract_descr descr;
    ROCSPARSE_CHECK(
        rocsparse_create_extract_descr(&descr, source, target, rocsparse_extract_alg_default));

    // Analysis phase
    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_extract_buffer_size(
        handle, descr, source, target, rocsparse_extract_stage_analysis, &buffer_size));
    void* dbuffer;
    HIP_CHECK(hipMalloc(&dbuffer, buffer_size));
    ROCSPARSE_CHECK(rocsparse_extract(
        handle, descr, source, target, rocsparse_extract_stage_analysis, buffer_size, dbuffer));
    HIP_CHECK(hipFree(dbuffer));

    // The user is responsible to allocate target arrays after the analysis phase.
    int64_t target_nnz;
    ROCSPARSE_CHECK(rocsparse_extract_nnz(handle, descr, &target_nnz));

    std::cout << "target_nnz: " << target_nnz << std::endl;

    void* dtarget_col_ind;
    void* dtarget_val;
    HIP_CHECK(hipMalloc(&dtarget_col_ind, sizeof(int32_t) * target_nnz));
    HIP_CHECK(hipMalloc(&dtarget_val, sizeof(float) * target_nnz));
    ROCSPARSE_CHECK(
        rocsparse_csr_set_pointers(target, dtarget_row_ptr, dtarget_col_ind, dtarget_val));

    // Calculation phase
    ROCSPARSE_CHECK(rocsparse_extract_buffer_size(
        handle, descr, source, target, rocsparse_extract_stage_compute, &buffer_size));
    HIP_CHECK(hipMalloc(&dbuffer, buffer_size));
    ROCSPARSE_CHECK(rocsparse_extract(
        handle, descr, source, target, rocsparse_extract_stage_compute, buffer_size, dbuffer));
    HIP_CHECK(hipFree(dbuffer));

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    HIP_CHECK(hipFree(dsource_row_ptr));
    HIP_CHECK(hipFree(dsource_col_ind));
    HIP_CHECK(hipFree(dsource_val));

    HIP_CHECK(hipFree(dtarget_row_ptr));
    HIP_CHECK(hipFree(dtarget_col_ind));
    HIP_CHECK(hipFree(dtarget_val));

    return 0;
}
//! [doc example]
