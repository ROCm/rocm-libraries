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
    //
    // Define a symmetric positive definite matrix (lower triangular stored).
    //
    //     4 2 0 0
    // A = 2 8 1 0
    //     0 1 8 2
    //     0 0 2 4
    //
    static constexpr int32_t              m            = 4;
    static constexpr int32_t              nnz          = 7;
    static constexpr rocsparse_index_base idx_base     = rocsparse_index_base_zero;
    static constexpr rocsparse_indextype  row_idx_type = rocsparse_indextype_i32;
    static constexpr rocsparse_indextype  col_idx_type = rocsparse_indextype_i32;
    static constexpr rocsparse_datatype   data_type    = rocsparse_datatype_f64_r;

    // Lower triangular part of A in CSR format
    const int32_t hcsr_row_ptr[m + 1] = {0, 1, 3, 5, 7};

    const int32_t hcsr_col_ind[nnz] = {0, 0, 1, 1, 2, 2, 3};

    const double hcsr_val[nnz] = {4.0, 2.0, 8.0, 1.0, 8.0, 2.0, 4.0};

    const double singularity_tolerance = 1e-10;

    const int32_t boost_enable    = 0;
    const double  boost_tolerance = 1e-10;
    const double  boost_value     = 1.0;

    //
    // Offload data to device
    //
    int32_t* dcsr_row_ptr;
    HIP_CHECK(hipMalloc(&dcsr_row_ptr, sizeof(int32_t) * (m + 1)));
    HIP_CHECK(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr, sizeof(int32_t) * (m + 1), hipMemcpyHostToDevice));

    int32_t* dcsr_col_ind;
    HIP_CHECK(hipMalloc(&dcsr_col_ind, sizeof(int32_t) * nnz));
    HIP_CHECK(hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(int32_t) * nnz, hipMemcpyHostToDevice));

    double* dcsr_val;
    HIP_CHECK(hipMalloc(&dcsr_val, sizeof(double) * nnz));
    HIP_CHECK(hipMemcpy(dcsr_val, hcsr_val, sizeof(double) * nnz, hipMemcpyHostToDevice));

    //
    // Create handle.
    //
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    //
    // Create sparse matrix A
    //
    rocsparse_spmat_descr matA;
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

    //
    // Create the descriptor of the Incomplete LDL^H algorithm of level 0.
    //
    rocsparse_spildlt0_descr spildlt0_descr;
    ROCSPARSE_CHECK(rocsparse_spildlt0_descr_create(handle, &spildlt0_descr, nullptr));

    //
    // Configure the descriptor.
    //
    const rocsparse_spildlt0_alg spildlt0_alg = rocsparse_spildlt0_alg_default;
    ROCSPARSE_CHECK(rocsparse_spildlt0_set_input(handle,
                                                 spildlt0_descr,
                                                 rocsparse_spildlt0_input_alg,
                                                 &spildlt0_alg,
                                                 sizeof(spildlt0_alg),
                                                 nullptr));

    const rocsparse_datatype spildlt0_compute_datatype = rocsparse_datatype_f64_r;
    ROCSPARSE_CHECK(rocsparse_spildlt0_set_input(handle,
                                                 spildlt0_descr,
                                                 rocsparse_spildlt0_input_compute_datatype,
                                                 &spildlt0_compute_datatype,
                                                 sizeof(spildlt0_compute_datatype),
                                                 nullptr));

    const rocsparse_analysis_policy spildlt0_analysis_policy = rocsparse_analysis_policy_reuse;
    ROCSPARSE_CHECK(rocsparse_spildlt0_set_input(handle,
                                                 spildlt0_descr,
                                                 rocsparse_spildlt0_input_analysis_policy,
                                                 &spildlt0_analysis_policy,
                                                 sizeof(spildlt0_analysis_policy),
                                                 nullptr));

    ROCSPARSE_CHECK(rocsparse_spildlt0_set_input(handle,
                                                 spildlt0_descr,
                                                 rocsparse_spildlt0_input_singularity_tolerance,
                                                 &singularity_tolerance,
                                                 sizeof(double),
                                                 nullptr));

    ROCSPARSE_CHECK(rocsparse_spildlt0_set_input(handle,
                                                 spildlt0_descr,
                                                 rocsparse_spildlt0_input_boost_enable,
                                                 &boost_enable,
                                                 sizeof(int32_t),
                                                 nullptr));

    ROCSPARSE_CHECK(rocsparse_spildlt0_set_input(handle,
                                                 spildlt0_descr,
                                                 rocsparse_spildlt0_input_boost_tolerance,
                                                 &boost_tolerance,
                                                 sizeof(double),
                                                 nullptr));

    ROCSPARSE_CHECK(rocsparse_spildlt0_set_input(handle,
                                                 spildlt0_descr,
                                                 rocsparse_spildlt0_input_boost_value,
                                                 &boost_value,
                                                 sizeof(double),
                                                 nullptr));

    hipStream_t stream;
    ROCSPARSE_CHECK(rocsparse_get_stream(handle, &stream));

    //
    // SpILDLT0 Analysis phase
    //
    size_t non_persistent_buffer_size_in_bytes;
    void*  non_persistent_buffer;

    ROCSPARSE_CHECK(rocsparse_spildlt0_buffer_size(handle,
                                                   spildlt0_descr,
                                                   matA,
                                                   matA,
                                                   rocsparse_spildlt0_stage_analysis,
                                                   &non_persistent_buffer_size_in_bytes,
                                                   nullptr));
    HIP_CHECK(hipMalloc(&non_persistent_buffer, non_persistent_buffer_size_in_bytes));

    ROCSPARSE_CHECK(rocsparse_spildlt0(handle,
                                       spildlt0_descr,
                                       matA,
                                       matA,
                                       rocsparse_spildlt0_stage_analysis,
                                       non_persistent_buffer_size_in_bytes,
                                       non_persistent_buffer,
                                       nullptr));

    //
    // Check for any singularities after analysis.
    //
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    rocsparse_singularity post_analysis_singularity;
    ROCSPARSE_CHECK(rocsparse_spildlt0_get_output(handle,
                                                  spildlt0_descr,
                                                  rocsparse_spildlt0_output_singularity,
                                                  &post_analysis_singularity,
                                                  sizeof(rocsparse_singularity),
                                                  nullptr));

    int64_t singularity_position;
    ROCSPARSE_CHECK(rocsparse_spildlt0_get_output(handle,
                                                  spildlt0_descr,
                                                  rocsparse_spildlt0_output_singularity_position,
                                                  &singularity_position,
                                                  sizeof(int64_t),
                                                  nullptr));
    HIP_CHECK(hipStreamSynchronize(stream));

    switch(post_analysis_singularity)
    {
    case rocsparse_singularity_none:
    {
        break;
    }
    case rocsparse_singularity_symbolic:
    {
        std::cout << "symbolic singularity detected at position: " << singularity_position
                  << std::endl;
        ROCSPARSE_CHECK(rocsparse_status_zero_pivot);
        break;
    }
    case rocsparse_singularity_numeric_exact:
    case rocsparse_singularity_numeric_near:
    {
        ROCSPARSE_CHECK(rocsparse_status_internal_error);
        break;
    }
    }

    //
    // Compute phase.
    //
    ROCSPARSE_CHECK(rocsparse_spildlt0_buffer_size(handle,
                                                   spildlt0_descr,
                                                   matA,
                                                   matA,
                                                   rocsparse_spildlt0_stage_compute,
                                                   &non_persistent_buffer_size_in_bytes,
                                                   nullptr));
    HIP_CHECK(hipFree(non_persistent_buffer));
    non_persistent_buffer = nullptr;
    HIP_CHECK(hipMalloc(&non_persistent_buffer, non_persistent_buffer_size_in_bytes));

    ROCSPARSE_CHECK(rocsparse_spildlt0(handle,
                                       spildlt0_descr,
                                       matA,
                                       matA,
                                       rocsparse_spildlt0_stage_compute,
                                       non_persistent_buffer_size_in_bytes,
                                       non_persistent_buffer,
                                       nullptr));

    //
    // Check for any singularities after compute.
    //
    rocsparse_singularity post_compute_singularity;
    ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    ROCSPARSE_CHECK(rocsparse_spildlt0_get_output(handle,
                                                  spildlt0_descr,
                                                  rocsparse_spildlt0_output_singularity,
                                                  &post_compute_singularity,
                                                  sizeof(rocsparse_singularity),
                                                  nullptr));

    ROCSPARSE_CHECK(rocsparse_spildlt0_get_output(handle,
                                                  spildlt0_descr,
                                                  rocsparse_spildlt0_output_singularity_position,
                                                  &singularity_position,
                                                  sizeof(int64_t),
                                                  nullptr));
    HIP_CHECK(hipStreamSynchronize(stream));

    switch(post_compute_singularity)
    {
    case rocsparse_singularity_none:
    {
        break;
    }
    case rocsparse_singularity_symbolic:
    {
        std::cout << "numeric symbolic singularity detected at position: " << singularity_position
                  << std::endl;
        ROCSPARSE_CHECK(rocsparse_status_internal_error);
        break;
    }
    case rocsparse_singularity_numeric_exact:
    {
        std::cout << "numeric exact singularity detected at position: " << singularity_position
                  << std::endl;
        break;
    }
    case rocsparse_singularity_numeric_near:
    {
        std::cout << "numeric near singularity detected at position: " << singularity_position
                  << std::endl;
        break;
    }
    }

    HIP_CHECK(hipFree(non_persistent_buffer));

    //
    // Back-solve  A x = b  with the incomplete factor  A = L D L^H  using three
    // sptrsv phases:
    //   1. L y = b    (lower, unit diagonal)
    //   2. D z = y    (diagonal-only solve, non-unit diagonal)
    //   3. L^H x = z  (transpose of the lower factor, unit diagonal)
    //
    // L (unit-lower) and D (on the diagonal) live in the same factor arrays.
    // Expose one spmat view per solve phase; all three share the same arrays and
    // each keeps its own analysis:
    //   matL  : lower / unit          -> L  y = b
    //   matD  : diagonal / non-unit   -> D  z = y
    //   matLt : lower / unit          -> L^H x = z  (solved with the transpose
    //                                                 operation; same arrays as L)
    //
    rocsparse_spmat_descr matL;
    rocsparse_spmat_descr matD;
    rocsparse_spmat_descr matLt;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matL,
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
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matD,
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
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matLt,
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

    const rocsparse_fill_mode fill_lower    = rocsparse_fill_mode_lower;
    const rocsparse_fill_mode fill_diagonal = rocsparse_fill_mode_diagonal;
    const rocsparse_diag_type diag_unit     = rocsparse_diag_type_unit;
    const rocsparse_diag_type diag_non_unit = rocsparse_diag_type_non_unit;
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        matL, rocsparse_spmat_fill_mode, &fill_lower, sizeof(fill_lower)));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        matL, rocsparse_spmat_diag_type, &diag_unit, sizeof(diag_unit)));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        matD, rocsparse_spmat_fill_mode, &fill_diagonal, sizeof(fill_diagonal)));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        matD, rocsparse_spmat_diag_type, &diag_non_unit, sizeof(diag_non_unit)));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        matLt, rocsparse_spmat_fill_mode, &fill_lower, sizeof(fill_lower)));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        matLt, rocsparse_spmat_diag_type, &diag_unit, sizeof(diag_unit)));

    const double alpha = 1.0;
    const double hb[m] = {1.0, 1.0, 1.0, 1.0};

    double* db;
    double* dy;
    double* dz;
    double* dx;
    HIP_CHECK(hipMalloc(&db, sizeof(double) * m));
    HIP_CHECK(hipMalloc(&dy, sizeof(double) * m));
    HIP_CHECK(hipMalloc(&dz, sizeof(double) * m));
    HIP_CHECK(hipMalloc(&dx, sizeof(double) * m));
    HIP_CHECK(hipMemcpy(db, hb, sizeof(double) * m, hipMemcpyHostToDevice));

    rocsparse_dnvec_descr vecB;
    rocsparse_dnvec_descr vecY;
    rocsparse_dnvec_descr vecZ;
    rocsparse_dnvec_descr vecX;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecB, m, db, data_type));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecY, m, dy, data_type));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecZ, m, dz, data_type));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vecX, m, dx, data_type));

    // The three solves share the same algorithm and reuse their (cached)
    // analysis across right-hand sides.
    const rocsparse_sptrsv_alg      alg             = rocsparse_sptrsv_alg_default;
    const rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;

    size_t buffer_size;
    void*  temp_buffer;

    // 1. Solve L y = b.
    {
        rocsparse_sptrsv_descr sptrsv_descr;
        ROCSPARSE_CHECK(rocsparse_create_sptrsv_descr(&sptrsv_descr));

        const rocsparse_operation operation = rocsparse_operation_none;
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(
            handle, sptrsv_descr, rocsparse_sptrsv_input_alg, &alg, sizeof(alg), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_operation,
                                                   &operation,
                                                   sizeof(operation),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_scalar_datatype,
                                                   &data_type,
                                                   sizeof(data_type),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_compute_datatype,
                                                   &data_type,
                                                   sizeof(data_type),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_analysis_policy,
                                                   &analysis_policy,
                                                   sizeof(analysis_policy),
                                                   nullptr));

        // Analysis stage.
        ROCSPARSE_CHECK(rocsparse_sptrsv_buffer_size(handle,
                                                     sptrsv_descr,
                                                     matL,
                                                     vecB,
                                                     vecY,
                                                     rocsparse_sptrsv_stage_analysis,
                                                     &buffer_size,
                                                     nullptr));
        HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));
        ROCSPARSE_CHECK(rocsparse_sptrsv(handle,
                                         sptrsv_descr,
                                         matL,
                                         vecB,
                                         vecY,
                                         rocsparse_sptrsv_stage_analysis,
                                         buffer_size,
                                         temp_buffer,
                                         nullptr));
        HIP_CHECK(hipFree(temp_buffer));

        // Compute stage.
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_scalar_alpha,
                                                   &alpha,
                                                   sizeof(alpha),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_buffer_size(handle,
                                                     sptrsv_descr,
                                                     matL,
                                                     vecB,
                                                     vecY,
                                                     rocsparse_sptrsv_stage_compute,
                                                     &buffer_size,
                                                     nullptr));
        HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));
        ROCSPARSE_CHECK(rocsparse_sptrsv(handle,
                                         sptrsv_descr,
                                         matL,
                                         vecB,
                                         vecY,
                                         rocsparse_sptrsv_stage_compute,
                                         buffer_size,
                                         temp_buffer,
                                         nullptr));
        HIP_CHECK(hipFree(temp_buffer));

        ROCSPARSE_CHECK(rocsparse_destroy_sptrsv_descr(sptrsv_descr));
    }

    // 2. Solve D z = y (diagonal-only solve).
    {
        rocsparse_sptrsv_descr sptrsv_descr;
        ROCSPARSE_CHECK(rocsparse_create_sptrsv_descr(&sptrsv_descr));

        const rocsparse_operation operation = rocsparse_operation_none;
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(
            handle, sptrsv_descr, rocsparse_sptrsv_input_alg, &alg, sizeof(alg), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_operation,
                                                   &operation,
                                                   sizeof(operation),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_scalar_datatype,
                                                   &data_type,
                                                   sizeof(data_type),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_compute_datatype,
                                                   &data_type,
                                                   sizeof(data_type),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_analysis_policy,
                                                   &analysis_policy,
                                                   sizeof(analysis_policy),
                                                   nullptr));

        // Analysis stage.
        ROCSPARSE_CHECK(rocsparse_sptrsv_buffer_size(handle,
                                                     sptrsv_descr,
                                                     matD,
                                                     vecY,
                                                     vecZ,
                                                     rocsparse_sptrsv_stage_analysis,
                                                     &buffer_size,
                                                     nullptr));
        HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));
        ROCSPARSE_CHECK(rocsparse_sptrsv(handle,
                                         sptrsv_descr,
                                         matD,
                                         vecY,
                                         vecZ,
                                         rocsparse_sptrsv_stage_analysis,
                                         buffer_size,
                                         temp_buffer,
                                         nullptr));
        HIP_CHECK(hipFree(temp_buffer));

        // Compute stage.
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_scalar_alpha,
                                                   &alpha,
                                                   sizeof(alpha),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_buffer_size(handle,
                                                     sptrsv_descr,
                                                     matD,
                                                     vecY,
                                                     vecZ,
                                                     rocsparse_sptrsv_stage_compute,
                                                     &buffer_size,
                                                     nullptr));
        HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));
        ROCSPARSE_CHECK(rocsparse_sptrsv(handle,
                                         sptrsv_descr,
                                         matD,
                                         vecY,
                                         vecZ,
                                         rocsparse_sptrsv_stage_compute,
                                         buffer_size,
                                         temp_buffer,
                                         nullptr));
        HIP_CHECK(hipFree(temp_buffer));

        ROCSPARSE_CHECK(rocsparse_destroy_sptrsv_descr(sptrsv_descr));
    }

    // 3. Solve L^H x = z (transpose of the stored lower factor).
    {
        rocsparse_sptrsv_descr sptrsv_descr;
        ROCSPARSE_CHECK(rocsparse_create_sptrsv_descr(&sptrsv_descr));

        const rocsparse_operation operation = rocsparse_operation_transpose;
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(
            handle, sptrsv_descr, rocsparse_sptrsv_input_alg, &alg, sizeof(alg), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_operation,
                                                   &operation,
                                                   sizeof(operation),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_scalar_datatype,
                                                   &data_type,
                                                   sizeof(data_type),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_compute_datatype,
                                                   &data_type,
                                                   sizeof(data_type),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_analysis_policy,
                                                   &analysis_policy,
                                                   sizeof(analysis_policy),
                                                   nullptr));

        // Analysis stage.
        ROCSPARSE_CHECK(rocsparse_sptrsv_buffer_size(handle,
                                                     sptrsv_descr,
                                                     matLt,
                                                     vecZ,
                                                     vecX,
                                                     rocsparse_sptrsv_stage_analysis,
                                                     &buffer_size,
                                                     nullptr));
        HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));
        ROCSPARSE_CHECK(rocsparse_sptrsv(handle,
                                         sptrsv_descr,
                                         matLt,
                                         vecZ,
                                         vecX,
                                         rocsparse_sptrsv_stage_analysis,
                                         buffer_size,
                                         temp_buffer,
                                         nullptr));
        HIP_CHECK(hipFree(temp_buffer));

        // Compute stage.
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   sptrsv_descr,
                                                   rocsparse_sptrsv_input_scalar_alpha,
                                                   &alpha,
                                                   sizeof(alpha),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_buffer_size(handle,
                                                     sptrsv_descr,
                                                     matLt,
                                                     vecZ,
                                                     vecX,
                                                     rocsparse_sptrsv_stage_compute,
                                                     &buffer_size,
                                                     nullptr));
        HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));
        ROCSPARSE_CHECK(rocsparse_sptrsv(handle,
                                         sptrsv_descr,
                                         matLt,
                                         vecZ,
                                         vecX,
                                         rocsparse_sptrsv_stage_compute,
                                         buffer_size,
                                         temp_buffer,
                                         nullptr));
        HIP_CHECK(hipFree(temp_buffer));

        ROCSPARSE_CHECK(rocsparse_destroy_sptrsv_descr(sptrsv_descr));
    }

    // Copy the solution back to the host and print it.
    double hx[m];
    HIP_CHECK(hipMemcpy(hx, dx, sizeof(double) * m, hipMemcpyDeviceToHost));
    HIP_CHECK(hipStreamSynchronize(stream));
    std::cout << "Solution x of A x = b:";
    for(int32_t i = 0; i < m; ++i)
    {
        std::cout << " " << hx[i];
    }
    std::cout << std::endl;

    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vecB));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vecY));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vecZ));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vecX));
    HIP_CHECK(hipFree(db));
    HIP_CHECK(hipFree(dy));
    HIP_CHECK(hipFree(dz));
    HIP_CHECK(hipFree(dx));

    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matL));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matD));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matLt));

    ROCSPARSE_CHECK(rocsparse_spildlt0_descr_destroy(handle, spildlt0_descr, nullptr));

    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matA));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));

    return 0;
}
//! [doc example]
