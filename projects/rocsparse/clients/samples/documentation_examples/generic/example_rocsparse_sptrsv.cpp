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

    std::vector<int>   hcsr_row_ptr = {0, 1, 3, 5, 6};
    std::vector<int>   hcsr_col_ind = {0, 0, 1, 1, 2, 3};
    std::vector<float> hcsr_val     = {1, 4, 2, 3, 7, 1};
    std::vector<float> hx(m, 1.0f);
    std::vector<float> hy(m, 0.0f);

    // Scalar alpha
    float alpha = 1.0f;

    int nnz = hcsr_row_ptr[m] - hcsr_row_ptr[0];

    // Offload data to device
    int*   dcsr_row_ptr;
    int*   dcsr_col_ind;
    float* dcsr_val;
    float* dx;
    float* dy;
    HIP_CHECK(hipMalloc(&dcsr_row_ptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc(&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc(&dcsr_val, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc(&dx, sizeof(float) * m));
    HIP_CHECK(hipMalloc(&dy, sizeof(float) * m));

    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(float) * m, hipMemcpyHostToDevice));

    rocsparse_handle      handle;
    rocsparse_spmat_descr matA;
    rocsparse_dnvec_descr vecX;
    rocsparse_dnvec_descr vecY;

    rocsparse_indextype  row_idx_type = rocsparse_indextype_i32;
    rocsparse_indextype  col_idx_type = rocsparse_indextype_i32;
    rocsparse_datatype   data_type    = rocsparse_datatype_f32_r;
    rocsparse_datatype   compute_type = rocsparse_datatype_f32_r;
    rocsparse_index_base idx_base     = rocsparse_index_base_zero;
    rocsparse_operation  trans        = rocsparse_operation_none;

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

    // Create dense vector X
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecX, m, dx, data_type));

    // Create dense vector Y
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecY, m, dy, data_type));

    rocsparse_sptrsv_descr sptrsv_descr;
    ROCSPARSE_CHECK(rocsparse_create_sptrsv_descr(&sptrsv_descr));

    const rocsparse_sptrsv_alg sptrsv_alg = rocsparse_sptrsv_alg_default;
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_alg,
                                               &sptrsv_alg,
                                               sizeof(sptrsv_alg),
                                               nullptr));

    const rocsparse_operation sptrsv_operation = rocsparse_operation_none;
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_operation,
                                               &sptrsv_operation,
                                               sizeof(sptrsv_operation),
                                               nullptr));

    const rocsparse_datatype sptrsv_scalar_datatype = rocsparse_datatype_f32_r;
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_scalar_datatype,
                                               &sptrsv_scalar_datatype,
                                               sizeof(sptrsv_scalar_datatype),
                                               nullptr));

    const rocsparse_datatype sptrsv_compute_datatype = rocsparse_datatype_f32_r;
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_compute_datatype,
                                               &sptrsv_compute_datatype,
                                               sizeof(sptrsv_compute_datatype),
                                               nullptr));

    const rocsparse_analysis_policy sptrsv_analysis_policy = rocsparse_analysis_policy_reuse;
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_analysis_policy,
                                               &sptrsv_analysis_policy,
                                               sizeof(sptrsv_analysis_policy),
                                               nullptr));

    size_t buffer_size;
    void*  temp_buffer;

    // Analysis phase
    ROCSPARSE_CHECK(rocsparse_sptrsv_buffer_size(handle,
                                                 sptrsv_descr,
                                                 matA,
                                                 vecX,
                                                 vecY,
                                                 rocsparse_sptrsv_stage_analysis,
                                                 &buffer_size,
                                                 nullptr));

    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    ROCSPARSE_CHECK(rocsparse_sptrsv(handle,
                                     sptrsv_descr,
                                     matA,
                                     vecX,
                                     vecY,
                                     rocsparse_sptrsv_stage_analysis,
                                     buffer_size,
                                     temp_buffer,
                                     nullptr));

    HIP_CHECK(hipFree(temp_buffer));
    temp_buffer = nullptr;

    int64_t zero_pivot;
    ROCSPARSE_CHECK(rocsparse_sptrsv_get_output(handle,
                                                sptrsv_descr,
                                                rocsparse_sptrsv_output_zero_pivot_position,
                                                &zero_pivot,
                                                sizeof(zero_pivot),
                                                nullptr));
    if(zero_pivot != -1)
    {
        std::cout << "zero pivot detected during analysis at position " << zero_pivot << std::endl;
    }

    //
    // Compute phase.
    //
    ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_scalar_alpha,
                                               &alpha,
                                               sizeof(&alpha),
                                               nullptr));
    ROCSPARSE_CHECK(rocsparse_sptrsv_buffer_size(handle,
                                                 sptrsv_descr,
                                                 matA,
                                                 vecX,
                                                 vecY,
                                                 rocsparse_sptrsv_stage_compute,
                                                 &buffer_size,
                                                 nullptr));

    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));
    ROCSPARSE_CHECK(rocsparse_sptrsv(handle,
                                     sptrsv_descr,
                                     matA,
                                     vecX,
                                     vecY,
                                     rocsparse_sptrsv_stage_compute,
                                     buffer_size,
                                     temp_buffer,
                                     nullptr));

    // Device synchronization
    hipStream_t stream;
    ROCSPARSE_CHECK(rocsparse_get_stream(handle, &stream));

    HIP_CHECK(hipStreamSynchronize(stream));
    ROCSPARSE_CHECK(rocsparse_sptrsv_get_output(handle,
                                                sptrsv_descr,
                                                rocsparse_sptrsv_output_zero_pivot_position,
                                                &zero_pivot,
                                                sizeof(zero_pivot),
                                                nullptr));
    if(zero_pivot != -1)
    {
        std::cout << "zero pivot detected during compute phase at position " << zero_pivot
                  << std::endl;
    }

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matA));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vecX));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vecY));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    ROCSPARSE_CHECK(rocsparse_destroy_sptrsv_descr(sptrsv_descr));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hy.data(), dy, sizeof(float) * m, hipMemcpyDeviceToHost));

    // Clear device memory
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));
    HIP_CHECK(hipFree(temp_buffer));
    return 0;
}
//! [doc example]
