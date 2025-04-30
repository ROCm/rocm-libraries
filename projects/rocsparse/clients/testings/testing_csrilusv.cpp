/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing.hpp"
template <typename T>
void testing_csrilusv_bad_arg(const Arguments& arg)
{
}

template <typename T>
void testing_csrilusv(const Arguments& arg)
{
    const bool                  to_int    = arg.timing ? false : true;
    static constexpr bool       full_rank = true;
    rocsparse_matrix_factory<T> matrix_factory(arg, to_int, full_rank);

    rocsparse_int             M    = arg.M;
    rocsparse_int             N    = arg.N;
    rocsparse_analysis_policy apol = arg.apol;
    rocsparse_solve_policy    spol = arg.spol;
    rocsparse_index_base      base = arg.baseA;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descrM;

    // Create matrix info
    rocsparse_local_mat_info info;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrM, base));

    // Allocate host memory for matrix
    host_vector<rocsparse_int> hcsr_row_ptr;
    host_vector<rocsparse_int> hcsr_col_ind;
    host_vector<T>             hcsr_val_gold;
    host_vector<rocsparse_int> h_struct_pivot_gold(1);
    host_vector<rocsparse_int> h_struct_pivot_1(1);
    host_vector<rocsparse_int> h_struct_pivot_2(1);
    host_vector<rocsparse_int> h_numeric_pivot_gold(1);
    host_vector<rocsparse_int> h_numeric_pivot_L_gold(1);
    host_vector<rocsparse_int> h_numeric_pivot_U_gold(1);
    host_vector<rocsparse_int> h_numeric_pivot_1(1);
    host_vector<rocsparse_int> h_numeric_pivot_2(1);
    host_vector<rocsparse_int> h_numeric_pivot_L_1(1);
    host_vector<rocsparse_int> h_numeric_pivot_L_2(1);
    host_vector<rocsparse_int> h_numeric_pivot_U_1(1);
    host_vector<rocsparse_int> h_numeric_pivot_U_2(1);

    host_vector<rocsparse_int> h_singular_pivot_gold(1);
    host_vector<rocsparse_int> h_singular_pivot_1(1);
    host_vector<rocsparse_int> h_singular_pivot_2(1);

    // Sample matrix
    rocsparse_int nnz;
    matrix_factory.init_csr(hcsr_row_ptr, hcsr_col_ind, hcsr_val_gold, M, N, nnz, base);

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnz);
    device_vector<T>             dcsr_val(nnz);
    device_vector<rocsparse_int> d_struct_pivot_2(1);
    device_vector<rocsparse_int> d_numeric_pivot_2(1);
    device_vector<rocsparse_int> d_numeric_pivot_L_2(1);
    device_vector<rocsparse_int> d_numeric_pivot_U_2(1);

    device_vector<rocsparse_int> d_singular_pivot_2(1);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !d_struct_pivot_2 || !d_numeric_pivot_2
       || !d_numeric_pivot_L_2 || !d_numeric_pivot_U_2)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val_gold, sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Compute reference incomplete LU factorization on host
    {
        double tol = 0;
        CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_get_tolerance(handle, info, &tol));

        host_csrilu0<T>(M,
                        hcsr_row_ptr,
                        hcsr_col_ind,
                        hcsr_val_gold,
                        base,
                        h_struct_pivot_gold,
                        h_numeric_pivot_gold,
                        h_singular_pivot_gold,
                        tol,
                        false,
                        std::real(static_cast<T>(0)),
                        static_cast<T>(0));
    }

    // Obtain csrilu0 buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_buffer_size<T>(
        handle, M, nnz, descrM, dcsr_val, dcsr_row_ptr, dcsr_col_ind, info, &buffer_size));

    // Allocate buffer
    device_dense_vector<char> dbuffer(buffer_size);

    // csrilu0 analysis
    CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_analysis<T>(
        handle, M, nnz, descrM, dcsr_val, dcsr_row_ptr, dcsr_col_ind, info, apol, spol, dbuffer));

    // Check for structural zero pivot using host pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrilu0_zero_pivot(handle, info, h_struct_pivot_1),
                            (h_struct_pivot_gold[0] != -1) ? rocsparse_status_zero_pivot
                                                           : rocsparse_status_success);

    // Check for structural zero pivot using device pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrilu0_zero_pivot(handle, info, d_struct_pivot_2),
                            (h_struct_pivot_gold[0] != -1) ? rocsparse_status_zero_pivot
                                                           : rocsparse_status_success);

    // Copy output to CPU
    CHECK_HIP_ERROR(hipMemcpy(
        h_struct_pivot_2, d_struct_pivot_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

    // Check pivot results
    h_struct_pivot_gold.unit_check(h_struct_pivot_1);
    h_struct_pivot_gold.unit_check(h_struct_pivot_2);

    // If structural pivot has been found, we are done
    if(h_struct_pivot_gold[0] != -1)
    {
        return;
    }

    // csrilu0
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0<T>(
        handle, M, nnz, descrM, dcsr_val, dcsr_row_ptr, dcsr_col_ind, info, spol, dbuffer));

    // Check for numerical zero pivot using host pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrilu0_zero_pivot(handle, info, h_numeric_pivot_1),
                            (h_numeric_pivot_gold[0] != -1) ? rocsparse_status_zero_pivot
                                                            : rocsparse_status_success);

    {
        auto st = rocsparse_csrilu0_singular_pivot(handle, info, h_singular_pivot_1);
        EXPECT_ROCSPARSE_STATUS(st, rocsparse_status_success);
    }
    // Check for structural zero pivot using device pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrilu0_zero_pivot(handle, info, d_numeric_pivot_2),
                            (h_numeric_pivot_gold[0] != -1) ? rocsparse_status_zero_pivot
                                                            : rocsparse_status_success);

    {

        auto st = rocsparse_csrilu0_singular_pivot(handle, info, d_singular_pivot_2);
        EXPECT_ROCSPARSE_STATUS(st, rocsparse_status_success);
    }

    // Copy output to CPU
    host_vector<T> hcsr_val(nnz);
    CHECK_HIP_ERROR(hipMemcpy(
        h_numeric_pivot_2, d_numeric_pivot_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        h_singular_pivot_2, d_singular_pivot_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hcsr_val, dcsr_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

    // Check pivot results
    h_numeric_pivot_gold.unit_check(h_numeric_pivot_1);
    h_numeric_pivot_gold.unit_check(h_numeric_pivot_2);

    h_singular_pivot_gold.unit_check(h_singular_pivot_1);
    h_singular_pivot_gold.unit_check(h_singular_pivot_2);

    // If numerical pivot has been found, we are done
    if(h_numeric_pivot_gold[0] != -1)
    {
        return;
    }

    // Check ILU factorization
    hcsr_val_gold.near_check(hcsr_val);

    // Create matrix descriptors for csrsv
    rocsparse_local_mat_descr descrL;
    rocsparse_local_mat_descr descrU;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrL, base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrU, base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(descrL, rocsparse_fill_mode_lower));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(descrU, rocsparse_fill_mode_upper));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(descrL, rocsparse_diag_type_unit));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(descrU, rocsparse_diag_type_non_unit));

    // Initialize structures for csrsv
    T h_alpha = static_cast<T>(1);

    host_vector<T> hx(N, static_cast<T>(1));
    host_vector<T> hy_1(M);
    host_vector<T> hy_2(M);
    host_vector<T> hy_gold(M);
    host_vector<T> hz_1(M);
    host_vector<T> hz_2(M);
    host_vector<T> hz_gold(M);

    // Allocate device memory
    device_vector<T> dx(N);
    device_vector<T> dy_1(M);
    device_vector<T> dy_2(M);
    device_vector<T> dz_1(M);
    device_vector<T> dz_2(M);
    device_vector<T> d_alpha(1);

    if(!dx || !dy_1 || !dy_2 || !dz_1 || !dz_2 || !d_alpha)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * N, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    // Compute reference solution on host
    host_csrsv<rocsparse_int, rocsparse_int, T>(rocsparse_operation_none,
                                                M,
                                                nnz,
                                                h_alpha,
                                                hcsr_row_ptr,
                                                hcsr_col_ind,
                                                hcsr_val_gold,
                                                hx,
                                                (int64_t)1,
                                                hz_gold,
                                                rocsparse_diag_type_unit,
                                                rocsparse_fill_mode_lower,
                                                base,
                                                h_struct_pivot_gold,
                                                h_numeric_pivot_L_gold);
    host_csrsv<rocsparse_int, rocsparse_int, T>(rocsparse_operation_none,
                                                M,
                                                nnz,
                                                h_alpha,
                                                hcsr_row_ptr,
                                                hcsr_col_ind,
                                                hcsr_val_gold,
                                                hz_gold,
                                                (int64_t)1,
                                                hy_gold,
                                                rocsparse_diag_type_non_unit,
                                                rocsparse_fill_mode_upper,
                                                base,
                                                h_struct_pivot_gold,
                                                h_numeric_pivot_U_gold);

    // Obtain csrsv buffer sizes
    size_t buffer_size_l;
    size_t buffer_size_u;

    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_buffer_size<T>(handle,
                                                         rocsparse_operation_none,
                                                         M,
                                                         nnz,
                                                         descrL,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         info,
                                                         &buffer_size_l));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_buffer_size<T>(handle,
                                                         rocsparse_operation_none,
                                                         M,
                                                         nnz,
                                                         descrU,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         info,
                                                         &buffer_size_u));

    // Buffer sizes should match with csrilu0 buffer size
    unit_check_scalar(buffer_size, buffer_size_l);
    unit_check_scalar(buffer_size, buffer_size_u);

    // csrsv analysis
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(handle,
                                                      rocsparse_operation_none,
                                                      M,
                                                      nnz,
                                                      descrL,
                                                      dcsr_val,
                                                      dcsr_row_ptr,
                                                      dcsr_col_ind,
                                                      info,
                                                      apol,
                                                      spol,
                                                      dbuffer));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(handle,
                                                      rocsparse_operation_none,
                                                      M,
                                                      nnz,
                                                      descrU,
                                                      dcsr_val,
                                                      dcsr_row_ptr,
                                                      dcsr_col_ind,
                                                      info,
                                                      apol,
                                                      spol,
                                                      dbuffer));

    // Lower part uses unit diagonal, structural pivot not possible

    // Check upper part for structural zero pivot using host pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descrU, info, h_struct_pivot_1),
                            (h_struct_pivot_gold[0] != -1) ? rocsparse_status_zero_pivot
                                                           : rocsparse_status_success);

    // Check upper part for structural zero pivot using device pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descrU, info, d_struct_pivot_2),
                            (h_struct_pivot_gold[0] != -1) ? rocsparse_status_zero_pivot
                                                           : rocsparse_status_success);

    // Copy output to CPU
    CHECK_HIP_ERROR(hipMemcpy(
        h_struct_pivot_2, d_struct_pivot_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

    // Check pivots
    h_struct_pivot_gold.unit_check(h_struct_pivot_1);
    h_struct_pivot_gold.unit_check(h_struct_pivot_2);

    // If structural pivot has been found, we are done
    if(h_struct_pivot_gold[0] != -1)
    {
        return;
    }

    // Solve Lz = x (= 1)

    // Host pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_solve<T>(handle,
                                                   rocsparse_operation_none,
                                                   M,
                                                   nnz,
                                                   &h_alpha,
                                                   descrL,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind,
                                                   info,
                                                   dx,
                                                   dz_1,
                                                   spol,
                                                   dbuffer));

    // Check for numerical zero pivot using host pointer mode
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descrL, info, h_numeric_pivot_L_1),
                            (h_numeric_pivot_L_gold[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

    // Device pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_solve<T>(handle,
                                                   rocsparse_operation_none,
                                                   M,
                                                   nnz,
                                                   d_alpha,
                                                   descrL,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind,
                                                   info,
                                                   dx,
                                                   dz_2,
                                                   spol,
                                                   dbuffer));

    // Check for numerical zero pivot using device pointer mode
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descrL, info, d_numeric_pivot_L_2),
                            (h_numeric_pivot_L_gold[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

    // Copy output to CPU
    CHECK_HIP_ERROR(hipMemcpy(
        h_numeric_pivot_L_2, d_numeric_pivot_L_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hz_1, dz_1, sizeof(T) * M, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hz_2, dz_2, sizeof(T) * M, hipMemcpyDeviceToHost));

    // Check pivot results
    h_numeric_pivot_L_gold.unit_check(h_numeric_pivot_L_1);
    h_numeric_pivot_L_gold.unit_check(h_numeric_pivot_L_2);

    // If numerical pivot has been found, we are done
    if(h_numeric_pivot_L_gold[0] != -1)
    {
        return;
    }

    // Check z
    hz_gold.near_check(hz_1);
    hz_gold.near_check(hz_2);

    // Solve Uy = z

    // Host pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_solve<T>(handle,
                                                   rocsparse_operation_none,
                                                   M,
                                                   nnz,
                                                   &h_alpha,
                                                   descrU,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind,
                                                   info,
                                                   dz_1,
                                                   dy_1,
                                                   spol,
                                                   dbuffer));

    // Check for numerical zero pivot using host pointer mode
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descrU, info, h_numeric_pivot_U_1),
                            (h_numeric_pivot_U_gold[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

    // Device pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_solve<T>(handle,
                                                   rocsparse_operation_none,
                                                   M,
                                                   nnz,
                                                   d_alpha,
                                                   descrU,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind,
                                                   info,
                                                   dz_2,
                                                   dy_2,
                                                   spol,
                                                   dbuffer));

    // Check for numerical zero pivot using device pointer mode
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descrU, info, d_numeric_pivot_U_2),
                            (h_numeric_pivot_U_gold[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

    // Copy output to CPU
    CHECK_HIP_ERROR(hipMemcpy(
        h_numeric_pivot_U_2, d_numeric_pivot_U_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * M, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * M, hipMemcpyDeviceToHost));

    // Check pivot and y
    h_numeric_pivot_U_gold.unit_check(h_numeric_pivot_U_1);
    h_numeric_pivot_U_gold.unit_check(h_numeric_pivot_U_2);

    // If numerical pivot has been found, we are done
    if(h_numeric_pivot_U_gold[0] != -1)
    {
        return;
    }

    // Check y
    hy_gold.near_check(hy_1);
    hy_gold.near_check(hy_2);

    // Clear csrsv meta data
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_clear(handle, descrL, info));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_clear(handle, descrU, info));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_clear(handle, info));
}

#define INSTANTIATE(TYPE)                                       \
    template void testing_csrilusv<TYPE>(const Arguments& arg); \
    template void testing_csrilusv_bad_arg<TYPE>(const Arguments& arg)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csrilusv_extra(const Arguments& arg) {}
